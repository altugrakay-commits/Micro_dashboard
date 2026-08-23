"""
3D PCB Stackup & Gerber Visualizer
==================================
Renders parametric multi-layer substrates, thermal via grids, and active 
surface-mount components using PyVista.
"""

from pathlib import Path
from typing import Any, Dict, Optional, cast
import numpy as np
import pyvista as pv

class GerberBoardVisualizer:
    """3D post-processor driven dynamically by user preference dictionaries."""

    def __init__(self, theme: str = "document") -> None:
        pv.set_plot_theme(theme)

    def generate_board_stackup(self, config: Dict[str, Any]) -> Dict[str, pv.DataSet]:
        """Generates parametric PCB components and substrate geometry."""
        board_w = float(config.get("board_width", 30.0))
        board_h = float(config.get("board_height", 30.0))
        thickness = float(config.get("substrate_thickness", 1.6))
        via_count = int(config.get("via_count", 16))
        via_radius = float(config.get("via_drill_radius", 0.2))
        pad_size = float(config.get("pad_size", 10.0))

        # 1. FR4 Substrate Body.
        substrate = cast(
            pv.DataSet,
            pv.Box(bounds=(-board_w / 2, board_w / 2, -board_h / 2, board_h / 2, -thickness, 0.0)).triangulate()
        )

        # 2. Top Copper Plane (F.Cu).
        top_copper = cast(
            pv.DataSet,
            pv.Box(bounds=(-board_w / 2, board_w / 2, -board_h / 2, board_h / 2, 0.0, 0.035)).triangulate()
        )

        # 3. IC Thermal Pad & Traces.
        p_half = pad_size / 2.0
        ic_pad = cast(
            pv.DataSet,
            pv.Box(bounds=(-p_half, p_half, -p_half, p_half, 0.035, 0.070)).triangulate()
        )

        # 4. Thermal Via Array Generation.
        grid_dim = max(1, int(np.sqrt(via_count)))
        spacing = (pad_size * 0.8) / max(1, grid_dim - 1) if grid_dim > 1 else 0.0
        start_offset = -((grid_dim - 1) * spacing) / 2.0 if grid_dim > 1 else 0.0

        vias_list = [
            pv.Cylinder(
                center=(start_offset + i * spacing, start_offset + j * spacing, -thickness / 2.0),
                direction=(0, 0, 1),
                radius=via_radius,
                height=thickness + 0.07,
                resolution=16
            ).triangulate()
            for i in range(grid_dim)
            for j in range(grid_dim)
        ]

        vias_combined = vias_list[0]
        for v in vias_list[1:]:
            vias_combined = vias_combined.merge(v)

        # 5. IC Package Geometry based on selected package.
        pkg_w = pad_size * 0.9
        pkg_h = float(config.get("package_height", 1.2))
        chip_body = pv.Box(bounds=(-pkg_w / 2, pkg_w / 2, -pkg_w / 2, pkg_w / 2, 0.070, 0.070 + pkg_h))

        die_w = pad_size * 0.6
        silicon_die = pv.Box(bounds=(-die_w / 2, die_w / 2, -die_w / 2, die_w / 2, 0.100, 0.400))

        return {
            "substrate": substrate,
            "top_copper": top_copper,
            "ic_pad": ic_pad,
            "thermal_vias": cast(pv.DataSet, vias_combined),
            "ic_package": cast(pv.DataSet, chip_body.triangulate()),
            "silicon_die": cast(pv.DataSet, silicon_die.triangulate())
        }

    def render_3d_pcb_assembly(
            self,
            layers: Dict[str, pv.DataSet],
            config: Dict[str, Any],
            title: str = "3D PCB & Component Assembly",
            output_png: Optional[Path] = None,
            interactive: bool = True
    ) -> None:
        """Renders extruded PCB stackup, copper features, and thermal via grid."""
        plotter = pv.Plotter(window_size=[1280, 720], off_screen=not interactive)

        # 1. Render FR4 Substrate (Semi-transparent Green).
        plotter.add_mesh(layers["substrate"], color=config.get("substrate_color","#1b4d2e"), opacity=0.65)

        # 2. Render Copper Base Plane (Copper Metallic).
        plotter.add_mesh(layers["top_copper"], color="#b87333", opacity=0.3)

        # 3. Render IC Thermal Ground Pad (Bright Gold).
        plotter.add_mesh(layers["ic_pad"], color="#ffd700", metallic=0.8)

       # 4. Render Thermal Via Array (Plated Copper Cylinders).
        plotter.add_mesh(layers["thermal_vias"], color="#d4af37")

        # 5. Mounted Component (QFN IC Body & Embedded Die).
        plotter.add_mesh(layers["ic_package"], color="#222222", opacity=0.85, smooth_shading=True)

        plotter.add_mesh(layers["silicon_die"], color="#00ffff",metallic=0.9)

        # 6. HUD Text Overlays.
        plotter.add_text(title, position="upper_left", font_size=14, color="black")

        hud_metrics = {
            "Board Size": f"{config.get('board_width', 30.0)} x {config.get('board_height', 30.0)} mm",
            "Package": str(config.get('package_type', 'QFN-16')),
            "Substrate Thickness": f"{config.get('substrate_thickness', 1.6):.2f} mm",
            "Via Array": f"{config.get('via_count', 16)} Vias ({config.get('via_drill_radius', 0.2)*2:.2f}mm Drill)",
            "Pad Area": f"{config.get('pad_size', 10.0)} x {config.get('pad_size', 10.0)} mm"
        }

        hud_text = "--- LAYOUT METRICS ---\n" + "\n".join(f"{k}: {v}" for k, v in hud_metrics.items())
        plotter.add_text(hud_text, position="upper_right", font_size=10, color="darkblue", shadow=True)

        # 6. Scene Settings and Camera.
        plotter.add_axes(interactive=False) # type: ignore
        plotter.show_grid(color="gray")     # type: ignore

        if output_png:
            output_png.parent.mkdir(parents=True, exist_ok=True)

        if interactive:
            plotter.show(interactive_update=True)
            if output_png:
                plotter.screenshot(str(output_png))
                print(f"✅ PCB render saved to: {output_png}")
            plotter.show()
        else:
            # Headless execution.
            plotter.show(screenshot=str(output_png), auto_close=True)
            print(f"✅ PCB render saved to: {output_png}")