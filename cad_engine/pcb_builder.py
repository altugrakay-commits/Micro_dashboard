"""
CadQuery 3D PCB Builder
=======================
Generates parameterized 3D PCB substrates, thermal via arrays, surface traces, 
and component geometries, with export capabilities to STEP, STL, and OBJ formats.
"""

import json
import cadquery as cq
import trimesh
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, cast

@dataclass
class ComponentSpec:
    """Defines a PCB component's physical dimensions, location, and thermal generation properties."""
    name: str
    footprint_type: str                     # e.g., '0805', 'QFN-32', 'USB-C'
    x: float                                # mm (Origin centered at 0.0)
    y: float                                # mm
    layer: str = "F.Cu"                     # 'F.Cu' or 'B.Cu'
    rotation: float = 0.0                   # Degrees
    height_mm: float = 1.0
    width_mm: float = 2.0
    length_mm: float = 2.0
    power_dissipation_w: float = 0.0        # Heat dissipation source Q
    is_anchor: bool = False                 # Hard-locked edge/connector component.
    keepout_margin_mm: float = 0.5
    color: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0) # Default dark gray

class PCBBuilder:
    """CadQuery engine for generating 3D PCB substrate, 3D components, and coordinate maps."""

    def __init__(
            self,
            preset_name: str="default",
            config_path: str="outputs/design_rules.json",
            output_dir: str="outputs/CAD_3D"
            ) -> None:
        self.preset_name = preset_name
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.components: List[ComponentSpec] = []
        self.total_h: float = 1.6   # Default substrate thickness in mm.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loads rules dynamically on initialization.
        self.rules = self.load_pcb_rules(self.preset_name, self.config_path)

    def load_pcb_rules(self, preset_name: str, config_path: Path) -> dict:
        """Loads design rules from JSON with robust default values."""
        defaults = {
                "min_trace_clearance_um": 100.0,
                "min_bend_radius_mm": 0.3,
                "min_dielectric_thickness_um": 100.0,
                "trace_expansion_factor": 1.0,
                "relief_gap_um": 50.0,
                "via_pitch_mm": 1.5,
                "corner_fillet_um": 50.0
            }

        if not config_path.exists():
            return defaults

        try:    
            with open(config_path, "r", encoding="utf-8") as f:
                rules_data = json.load(f)
                return {**defaults, **rules_data.get(preset_name, {})}
        except Exception:
            return defaults

    def build_dielectric_stackup(
            self, 
            board_size_mm: tuple[float, float], 
            target_thickness_mm: Optional[float] = None, 
            num_layers: int=4
        ) -> tuple[cq.Workplane, float]:
        """Builds substrates stackup geometry given dielectric parameters."""
        width, length = board_size_mm

        if target_thickness_mm is not None:
            self.total_h = float(target_thickness_mm)
        else:
            # Convert dielectric thickness from µm to mm.
            dielectric_h = self.rules.get("min_dielectric_thickness_um", 100.0) / 1000.0
            copper_h = 0.035 # Standard 1 oz copper (35 µm).
            self.total_h = float(num_layers * (dielectric_h + copper_h))

        # Calculate exact total height and construct single-fused base block to avoid coplanar face artifacts.
        stackup = cq.Workplane("XY").box(width, length, self.total_h, centered=(True, True, False))
        return stackup, self.total_h

    def build_die_recess_cutout(self, stackup: cq.Workplane, die_pad_size_mm: float) -> cq.Workplane:
        """Creates a die placement cavity incorporating thermal stress expansion relief gap."""
        relief_gap_mm = self.rules.get("relief_gap_um", 50.0) / 1000.0
        # Pocket dimensions = Die dimensions + calculated CTE relief gap.
        pocket_dim = die_pad_size_mm + (2.0 * relief_gap_mm)
        recess_depth_mm = 0.25 # Defined cavity depth.

        return (
            stackup.faces(">Z")
            .workplane()
            .rect(pocket_dim, pocket_dim)
            .cutBlind(-recess_depth_mm)
            )

    def generate_thermal_via_array(
            self,
            board_size_mm: tuple[float,float],
            total_height_mm: float,
            die_pad_size_mm: float = 10.0,
            via_radius_mm: float = 0.3,
            via_pitch_mm: Optional[float] = None
            ) -> Optional[cq.Workplane]:
        """Generates cylindrical thermal via arrays outside the central die recess pocket."""
        width, length = board_size_mm
        via_pitch = via_pitch_mm if via_pitch_mm is not None else float(self.rules.get("via_pitch_mm", 1.5))
        relief_gap_mm = self.rules.get("relief_gap_um", 50.0) / 1000.0
        pocket_half_dim = (die_pad_size_mm + 2.0 * relief_gap_mm) / 2.0

        x_span = np.arange(-width / 4.0, width / 4.0, via_pitch)
        y_span = np.arange(-length / 4.0, length / 4.0, via_pitch)

        # Generate 2D point coordinates for the grid.
        xx, yy = np.meshgrid(x_span, y_span)

        # Vector Mask: Keep vias outside the central die recess pocket
        mask = (np.abs(xx) > pocket_half_dim) | (np.abs(yy) > pocket_half_dim)

        # Fully vectorized array conversion directly to tuples.
        via_points = np.column_stack((xx[mask], yy[mask])).tolist()

        if not via_points:
            return None

        return (
            cq.Workplane("XY")
            .workplane(offset=0)
            .pushPoints(via_points)
            .circle(via_radius_mm)
            .extrude(total_height_mm)
        )

    def build_component_geometry(self, comp: ComponentSpec) -> cq.Workplane:
        """Generate 3D CadQuery solids for registered components."""
        # Baseline placement on top of board (Z = total_h).
        base_z = self.total_h if comp.layer == "F.Cu" else 0.0

        if comp.footprint_type == "IC_DIE":
            # Recessed central silicon die.
            recess_depth = 0.25
            z_pos = self.total_h - recess_depth
            return (
                cq.Workplane("XY")
                .workplane(offset=z_pos)
                .center(comp.x, comp.y)
                .rect(comp.length_mm, comp.width_mm)
                .extrude(comp.height_mm)
            )

        elif comp.footprint_type == "0805":
            # Passives/Capacitors. 0805 Ceramic Body with Metallic End Caps.
            cap_body = (
                cq.Workplane("XY")
                .workplane(offset=base_z)
                .center(comp.x, comp.y)
                .rect(comp.length_mm, comp.width_mm)
                .extrude(comp.height_mm)
            )
            # Add end terminals (caps).
            cap_width = 0.4
            term1 = (
                cq.Workplane("XY")
                .workplane(offset=base_z)
                .center(comp.x - (comp.length_mm / 2.0) + (cap_width / 2.0), comp.y)
                .rect(cap_width, comp.width_mm * 1.02)
                .extrude(comp.height_mm * 1.02)
            )
            term2 = (
                cq.Workplane("XY")
                .workplane(offset=base_z)
                .center(comp.x + (comp.length_mm / 2.0) - (cap_width / 2.0), comp.y)
                .rect(cap_width, comp.width_mm * 1.02)
                .extrude(comp.height_mm * 1.02)
            )
            return cap_body.union(term1).union(term2)

        elif comp.footprint_type == "CONNECTOR":
            # Edge Connector / USB-C Box. Header Body with Metal Pin Contacts.
            conn_body = (
                cq.Workplane("XY")
                .workplane(offset=base_z)
                .center(comp.x, comp.y)
                .rect(comp.length_mm, comp.width_mm)
                .extrude(comp.height_mm)
            )
            pins = (
                cq.Workplane("XY")
                .workplane(offset=base_z)
                .center(comp.x, comp.y)
                .rarray(2.0, 1.0, 5, 1)
                .circle(0.3)
                .extrude(comp.height_mm + 1.5)
            )
            return conn_body.union(pins)

        else:
            # Generic Component Solid.
            return (
                cq.Workplane("XY")
                .workplane(offset=base_z)
                .center(comp.x, comp.y)
                .rect(comp.length_mm, comp.width_mm)
                .extrude(comp.height_mm)
            )

    def build_copper_traces(self, board_size_mm: tuple[float, float]) -> cq.Workplane:
        """Generates paramterized surface copper traces on top of F.Cu"""
        trace_h = 0.035 # 1 oz copper (mm)
        trace_w = 0.25  # Trace width (mm)
        z_pos = self.total_h

        # Route traces connecting central pad to outer decoupling caps.
        return (
            cq.Workplane("XY")
            .workplane(offset=z_pos)
            .moveTo(-10.0, 0.0).lineTo(10.0, 0.0)
            .moveTo(0.0, -10.0).lineTo(0.0, 10.0)
            .moveTo(-8.0, -8.0).lineTo(8.0, 8.0)
            .moveTo(-8.0, 8.0).lineTo(8.0, -8.0)
            .rect(trace_w, trace_w)
            .extrude(trace_h)
        )

    def populate_default_components(self, die_pad_size_mm: float) -> None:
        """Populates the PCB with default ICs, passive, and connector components."""
        self.components.clear()

        # 1. Main Silicon Die Component (Center).
        self.add_component(ComponentSpec(
            name="Silicon_Die_U1",
            footprint_type="IC_DIE",
            x=0.0, y=0.0,
            length_mm=die_pad_size_mm,
            width_mm=die_pad_size_mm,
            height_mm=0.8,
            power_dissipation_w=15.0,
            color=(0.15, 0.15, 0.18, 1.0)   # Shiny Dark Metallic Gray
        ))

        # 2. Surrounding Decoupling Capacitors (0805).
        offset = (die_pad_size_mm / 2.0) * 2.5
        cap_positions = [
            (offset, offset), (-offset, offset),
            (offset, -offset), (-offset, -offset),
            (0.0, offset + 1.5), (0.0, -(offset + 1.5))
        ]
        for idx, (cx, cy) in enumerate(cap_positions, 1):
            self.add_component(ComponentSpec(
                name=f"Capacitor_C{idx}",
                footprint_type="0805",
                x=cx, y=cy,
                length_mm=2.0, width_mm=1.25, height_mm=0.8,
                color=(0.7, 0.5, 0.2, 1.0)  # Ceramic Bronze / Tan.
            ))

        # 3. Main Power Connector / Interface Header (Edge).
        self.add_component(ComponentSpec(
            name="Power_Connector_J1",
            footprint_type="CONNECTOR",
            x=0.0, y=-16.0,
            length_mm=12.0, width_mm=5.0, height_mm=4.0,
            is_anchor=True,
            color=(0.8, 0.8, 0.8, 1.0)  # Metallic Silver.
        ))

    def process_and_clean_mesh(self, stl_path: Path) -> trimesh.Trimesh:
        """Repairs non-watertight geometry and merges duplicates vertices using Trimesh."""
        if not stl_path.exists():
            raise FileNotFoundError(f"Cannot clean missing mesh file: {stl_path}")

        loaded_geom = trimesh.load(str(stl_path))
        mesh = (
            trimesh.util.concatenate(loaded_geom.dump())
            if isinstance(loaded_geom, trimesh.Scene)
            else loaded_geom
        )
        # Explicitly cast to Trimesh to satisfy static analysis.
        mesh = cast(trimesh.Trimesh, mesh)

        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_inversion(mesh)
            trimesh.repair.fix_winding(mesh)

        mesh.merge_vertices()
        # Export processed OBJ mesh alongside STL.
        mesh.export(str(stl_path.with_suffix('.obj')))
        return mesh

    def generate_pcb_substrate(
            self,
            filename: str = "pcb_layout.step",
            board_size_mm: tuple[float, float] = (40.0, 40.0),
            thickness_mm: Optional[float] = None,
            die_pad_size_mm: float = 10.0,
            via_radius_mm: float = 0.3,
            via_pitch_mm: Optional[float] = None,
            num_layers: int = 4
    ) -> Path:
        """Constructs a parameterized 3D PCB substrate and exports individual CAD meshes."""
        if via_pitch_mm is not None:
            self.rules["via_pitch_mm"] = via_pitch_mm

        filepath = self.output_dir / filename
        effective_thickness = (
            thickness_mm
            if thickness_mm is not None
            else float((self.rules.get("min_dielectric_thickness_um", 100.0) / 1000.0 + 0.035) * num_layers)
        )

        try:
            # Build Base Substrate.
            pcb_substrate, total_h = self.build_dielectric_stackup(
                board_size_mm, 
                target_thickness_mm=thickness_mm, 
                num_layers=num_layers
                )

            # Apply Edge Fillets
            safe_fillet_mm = float(np.clip(self.rules.get("corner_fillet_um", 50.0) / 1000.0, 0.0, 0.5))
            if safe_fillet_mm > 0.05:
                pcb_substrate = pcb_substrate.edges("|Z").fillet(safe_fillet_mm)

            # Recess Cutout & Via Subtraction.
            pcb_substrate = self.build_die_recess_cutout(pcb_substrate, die_pad_size_mm)
            vias = self.generate_thermal_via_array(board_size_mm, total_h, die_pad_size_mm, via_radius_mm, via_pitch_mm=via_pitch_mm)
            if vias is not None:
                pcb_substrate = pcb_substrate.cut(vias)

            # Populate default 3D components if none were added manually.
            if not self.components:
                self.populate_default_components(die_pad_size_mm)

            # Export Component & Assembly Mesh Parts.
            parts_dir = self.output_dir / f"{filepath.stem}_parts"
            parts_dir.mkdir(parents=True, exist_ok=True)

            # Export Substrate STL.
            substrate_stl = parts_dir / "fr4_substrate.stl"
            cq.exporters.export(pcb_substrate, str(substrate_stl), exportType=cq.exporters.ExportTypes.STL)
            self.process_and_clean_mesh(substrate_stl)

            if vias is not None:
                via_stl = parts_dir / "thermal_via_array.stl"
                cq.exporters.export(vias, str(via_stl), exportType=cq.exporters.ExportTypes.STL)
                self.process_and_clean_mesh(via_stl)

            for comp in self.components:
                comp_solid = self.build_component_geometry(comp)

                # Format hex color from ComponentSpec tuple.
                r,g,b = [int(c * 255) for c in comp.color[:3]]
                hex_color = f"{r:02x}{g:02x}{b:02x}"

                part_prefix = (
                    "die" if comp.footprint_type == "IC_DIE"
                    else "solder" if comp.footprint_type == "CONNECTOR"
                    else f"component_{comp.footprint_type.lower()}"
                )
                comp_stl = parts_dir / f"{part_prefix}_{comp.name}_color_{hex_color}.stl"

                try:
                    cq.exporters.export(comp_solid, str(comp_stl), exportType=cq.exporters.ExportTypes.STL)
                    self.process_and_clean_mesh(comp_stl)
                except Exception as comp_err:
                    print(f"⚠️ Warning exporting component STL ({comp.name}): {comp_err}")

            # Master STEP export.
            assembly = cq.Assembly()
            assembly.add(pcb_substrate, name="FR4_Substrate")
            for comp in self.components:
                assembly.add(self.build_component_geometry(comp), name=comp.name)
            assembly.export(str(filepath), "STEP")

            # Export Copper Traces Mesh.
            traces = self.build_copper_traces(board_size_mm)
            trace_stl = parts_dir / "trace_copper.stl"
            cq.exporters.export(traces, str(trace_stl), exportType=cq.exporters.ExportTypes.STL)
            self.process_and_clean_mesh(trace_stl)

            return filepath
        
        except Exception as e:
            print(f"⚠️ Warning during 3D CAD assembly generation: {e}")
            fallback_box = cq.Workplane("XY").box(board_size_mm[0], board_size_mm[1], effective_thickness)
            fallback_path = self.output_dir / "pcb_fallback.step"
            cq.exporters.export(fallback_box, str(fallback_path))
            return fallback_path

    def add_component(self, comp: ComponentSpec) -> None:
        """Registers a component into the placement state."""
        self.components.append(comp)

    def export_physics_mesh_grid(self, grid_resolution_mm: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Maps component coordinates and heat dissipation rates into spatial tensors."""
        width, length = self.rules.get("board_size_mm", (40.0, 40.0))
        x_span = np.arange(-width/2.0, width/2.0, grid_resolution_mm)
        y_span = np.arange(-length/2.0, length/2.0, grid_resolution_mm)
        xx, yy = np.meshgrid(x_span, y_span)

        coords = np.column_stack([xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), self.total_h)])
        q_volumetric = np.zeros((coords.shape[0], 1), dtype=np.float32)

        # Map active heat sources to volumetric heat generation Q(x,y,z).
        for comp in self.components:
            if comp.power_dissipation_w > 0:
                dist = np.hypot(coords[:,0] - comp.x, coords[:,1] - comp.y)
                active_mask = dist <= (comp.length_mm / 2.0)
                # Volumetric heat source estimates W/m³
                vol_m3 = (comp.length_mm * comp.width_mm * comp.height_mm) * 1e-9
                q_volumetric[active_mask] = comp.power_dissipation_w / vol_m3

        return coords, q_volumetric