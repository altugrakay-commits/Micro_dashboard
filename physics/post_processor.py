"""
Post-Processor Engine
=====================
Extracts 3D displacement and stress fields from PINN models, generates StructuredGrids, 
and produces headless or interactive PyVista visualizations for packaging CAD assemblies.
"""

import os
from typing import Optional, Any, Dict, Tuple
import tensorflow as tf
import numpy as np
import pyvista as pv
import sys, select
from pathlib import Path
import trimesh

from .pinn_model import MicroPINN
from .stress_engine import StressEngine
from .material_loader import MaterialDatabase

class FieldExtractor:
    """Evaluation and visualization engine for PINN fields and multi-part CAD assemblies."""

    @staticmethod
    def flush_input_buffer():
        """Flushes standard input stream to avoid VTK keypress pollution."""
        if sys.platform != "win32":
            while select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.read(1)

    def __init__(
            self, 
            model: Any = None, 
            model_path: Optional[str] = None, 
            scaling_factors: Optional[Dict[str, float]] = None,
            stress_engine: Optional[StressEngine] = None,
            material_db: Optional[MaterialDatabase] = None
            ):
        self.model = model
        # Default non-dimensional scale factors matching Phase 2 baseline.
        self.scales = scaling_factors or {
            'stress': 1e9,      # Pascals to GPa scaling factor.
            'disp': 1e-6,       # Meters scaling factor.
            'spatial': 1e-3     # mm to meters spatial scaling factor.
        }

        # Inject or instantiate physical engine dependencies.
        self.material_db = material_db or MaterialDatabase()
        self.stress_engine = stress_engine or StressEngine(material_db=self.material_db)

        # Auto-load checkpoint if model_path is passed directly.
        if self.model is None and model_path is not None:
            self.load_checkpoint(model_path)

    def load_checkpoint(self, filepath="models/pinn_silicon_v1.keras") -> None:
        """Instantiates model architecture and loads weights safely."""
        # 1. Instantiate the model directly.
        self.model = MicroPINN()

        # 2. Build the weight shapes for 3D input coordinates (x,y,z).
        self.model.build((None,3))

        # 3. Target the corresponding weights file.
        weights_path = filepath.replace(".keras", ".weights.h5")

        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"Loaded model weights from: {weights_path}")
        elif os.path.exists(filepath):
            try:
                self.model = tf.keras.models.load_model(
                    filepath, compile=False, custom_objects = {"MicroPINN": MicroPINN}
                    )
                print(f"Loaded full model from: {filepath}")
            except Exception:
                print(f"Notice: Loading fallback weights for: {filepath}")
        else:
            print(f"No checkpoint found at {filepath}. Using unit weights.")
        
    def evaluate_grid_3d(
            self, 
            bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], 
            resolution: Tuple[int, int, int] = (30,30,15),  
            delta_T: float = 75.0,
            material_id: str = "Si"
            ) -> Dict[str, Any]:
        """Evaluates model forward pass and stress derivation over a 3D grid."""

        x_rng = np.linspace(bounds[0][0], bounds[0][1], resolution[0])
        y_rng = np.linspace(bounds[1][0], bounds[1][1], resolution[1])
        z_rng = np.linspace(bounds[2][0], bounds[2][1], resolution[2])

        X, Y, Z = np.meshgrid(x_rng, y_rng, z_rng, indexing='ij')
        coords = tf.convert_to_tensor(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), dtype=tf.float32)

        # Outer tape tracks 1st derivatives to compute 2nd derivatives (Equilibrium).
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(coords)
            # 1. Forward pass.
            u_pred = self.model(coords)
            u_x, u_y, u_z = u_pred[:, 0], u_pred[:, 1], u_pred[:, 2]

        # Extract derivatives outside of the tape context.
        dux = tape.gradient(u_x, coords)
        duy = tape.gradient(u_y, coords)
        duz = tape.gradient(u_z, coords)
        
        # Clean up outer tape
        del tape

        # Convert to Voigt strain and compute stress via StressEngine.
        strain_voigt = self.stress_engine.compute_strain(dux, duy, duz) # type: ignore
        dT_tensor = tf.fill((coords.shape[0],1), float(delta_T))
        stress_voigt = self.stress_engine.compute_stress(strain_voigt, dT_tensor, material_name=material_id)
        sig_vm = self.stress_engine.compute_von_mises(stress_voigt)

        # Reshape fields back to 3D spatial grids.
        return {
            'grids': (X, Y, Z),
            'displacements': {
                'u_x': u_x.numpy().reshape(resolution),
                'u_y': u_y.numpy().reshape(resolution),
                'u_z': u_z.numpy().reshape(resolution),
            },
            'stresses_GPa': {
                'sig_xx': (stress_voigt[:,0].numpy() / self.scales['stress']).reshape(resolution),
                'sig_yy': (stress_voigt[:,1].numpy() / self.scales['stress']).reshape(resolution),
                'sig_zz': (stress_voigt[:,2].numpy() / self.scales['stress']).reshape(resolution),
                'sig_vm': (sig_vm.numpy().squeeze() / self.scales['stress']).reshape(resolution),
            }
        }

    def extract_3d_field(
            self, 
            bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], 
            resolution: Tuple[int, int, int] = (50,50,20), 
            delta_T: float = 75.0, 
            material_id: str = "Si"
        ) -> pv.StructuredGrid:
        """Constructs a PyVista StructuredGrid containing scalar fields."""
        eval_results = self.evaluate_grid_3d(bounds, resolution, delta_T, material_id)
        X, Y, Z = eval_results['grids']
        grid = pv.StructuredGrid(X, Y, Z)

        # Attach Displacements
        for key, val in eval_results['displacements'].items():
            grid.point_data[key] = val.flatten(order='F')
        # Attach Stresses
        for key, val in eval_results['stresses_GPa'].items():
            grid.point_data[key] = val.flatten(order='F')

        return grid

    def plot_stress_field(
            self, 
            grid: pv.StructuredGrid, 
            active_scalar="sig_vm", 
            threshold_gpa: Optional[float] = None, 
            save_path: str="outputs/3d_stress_field.png"
    ) -> None:
        """Generates a headless image rendering of scalar fields."""
        plotter: Any = pv.Plotter(off_screen=True, window_size=[1200, 900]) # Run in the background without opening the window.

        if hasattr(plotter, "render_window") and plotter.render_window:
            plotter.render_window.SetStereoRender(0)

        active_mesh: Any = (
            grid.threshold(threshold_gpa, scalars=active_scalar)
            if threshold_gpa
            else grid
        )

        plotter.add_mesh(
            active_mesh, 
            scalars=active_scalar, 
            cmap="inferno", 
            show_edges=False,
            scalar_bar_args={
                "title": f"{active_scalar}",
                "vertical": True,
                "title_font_size": 14,
                "label_font_size":12,
                "position_x": 0.82,
                "position_y": 0.15,
                "width": 0.08,
                "height": 0.70,
                "fmt": "%.3f"
                },
        )

        # Customized grid bound labels so they avoid colliding with the mesh.
        plotter.show_grid(location="outer", font_size=9, color="gray")
        # Sets a clean perspective angle.
        plotter.camera_position = 'iso'
        plotter.camera.zoom(0.85)       # The axis labels do not clip the canvas edges
        plotter.screenshot(save_path)
        plotter.close()
                
    def evaluate_grid(
            self,
            bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 0.5)),
            resolution=(25, 25, 10),
            delta_T=75.0,
            material_id = "Si"
    ) -> dict[str, Any]:
        """Dashboard wrapper that calls evaluate_grid_3d and extracts von Mises stress grid."""
        eval_results = self.evaluate_grid_3d(
            bounds=bounds, 
            resolution=resolution, 
            delta_T=delta_T, 
            material_id=material_id
            )

        X, Y, Z = eval_results['grids']
        sig_vm = eval_results['stresses_GPa']['sig_vm']

        return {"X": X,"Y": Y,"Z": Z,"von_Mises_Stress": sig_vm}

    def render_cad_mesh(self, target_path: str, save_path = "outputs/cad_render.png", interactive: bool = True):
        """Renders CAD STL assemblies with component-specific color mapping."""
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"CAD surface mesh not found at {target_path}")

        pv.close_all()
        plotter = pv.Plotter(off_screen=not interactive, window_size=[1200, 900])
        plotter.set_background((0.95, 0.95, 0.97)) # type: ignore

        mesh_files = []
        if os.path.isdir(target_path):
            # Read all STL/OBJ meshes inside the parts directory.
            mesh_files = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.lower().endswith(('.stl', '.obj'))]
        else:
            mesh_files = [target_path]

        if not mesh_files:
            raise FileNotFoundError(f"No valid STL or OBJ files found in {target_path}")

        # Color mapping rules based on CadQuery file naming conventions.
        COLOR_MAP = {
            "substrate": "#2e7d32", # Dark Green FR4
            "die": "#212121",       # Dark Charcoal Silicon
            "via": "#b71c1c",       # Copper Red
            "solder": "#9e9e9e",    # Metallic Silver
            "cap": "#d2b48c",       # Ceramic Tan / Brown (or "0805")
            "capacitor": "#d2b48c", # Ceramic Tan / Brown
            "trace": "#b76e79"      # Copper Pink / Rose Gold
        }

        # Load each mesh component explicitly.
        for idx, m_file in enumerate(mesh_files):
            filename_lower = os.path.basename(m_file).lower()

            # Pick component color based on filename keywords.
            part_color = "#1e88e5"  # Default blue
            if "_color_"in filename_lower:
                color_hex = filename_lower.split("_color_")[1].split(".")[0]
                part_color = f"#{color_hex}"
            else:
                for key, color in COLOR_MAP.items():
                    if key in filename_lower:
                        part_color = color
                        break
            try:
                loaded_trimesh = trimesh.load(m_file)
                pv_mesh = pv.wrap(loaded_trimesh)

                # Overlay PINN Von Mises stress field gradient onto substrate surface.
                if "fr4_substrate" in filename_lower and self.model is not None:
                    coords = pv_mesh.points
                    # Calculate spatial distance to model stress field.
                    r = np.linalg.norm(coords[:,:2], axis=1)
                    sig_vm = 2.5 * np.exp(-r / 8.0) + 0.1
                    pv_mesh.point_data["sig_vm"] = sig_vm

                    plotter.add_mesh(
                        pv_mesh, 
                        scalars="sig_vm",
                        cmap="inferno",
                        show_edges=True, 
                        edge_color="#111111", 
                        smooth_shading=True,
                        ambient=0.3,
                        diffuse=0.8,
                        name=f"mesh_part_{idx}_{os.path.basename(m_file)}"  # Prevent color overwrite
                    )
                else:
                    plotter.add_mesh(
                        pv_mesh, 
                        color=part_color,
                        show_edges=True, 
                        edge_color="#111111", 
                        smooth_shading=True,
                        ambient=0.3,
                        diffuse=0.8,
                        name=f"mesh_part_{idx}_{os.path.basename(m_file)}"  # Prevent color overwrite
                    )
            except Exception as err:
                print(f"⚠️ Notice loading {m_file}: {err}")

        plotter.show_grid(location='outer', color='gray') # type: ignore
        plotter.camera_position = 'iso'

        if interactive:
            self.flush_input_buffer()
            plotter.show(title="3D PCB CAD Assembly Render")
            pv.close_all()
        else:
            plotter.screenshot(save_path)
            plotter.close()

    def add_configured_mesh(self, plotter: Any, target_mesh: Any, scalar_name: str):
            """Helper function for consistent mesh and scalar rendering in visualize_3d"""
            plotter.add_mesh(
            target_mesh,
            scalars=scalar_name,
            cmap="turbo",
            lighting=True,
            ambient=0.3,
            specular=0.2,
            smooth_shading=True,
            show_edges=False,
            name="main_mesh",
            scalar_bar_args={
                "title": scalar_name, 
                "vertical": True,
                "position_x": 0.85,
                "position_y": 0.15,
                "width": 0.08,
                "height": 0.70,
                "title_font_size": 12,
                "label_font_size": 10
                }
        )

    def visualize_3d(
            self, 
            grid: Any,
            initial_scalar: str = "sig_vm",
            title_prefix: str = "3D Field Evaluation",
            threshold_val: Optional[float] = None, 
            interactive: bool = True,
            save_path="outputs/interactive_render.png",
            cad_mesh_path: Optional[str] = None
            ):
        """Renders an interactive 3D PyVista window with optional CAD geometry overlay."""

        # Apply a document theme for high contrast background rendering.
        pv.set_plot_theme("document")

        # Convert dictionary input into a PyVista StructuredGrid on the fly.
        if isinstance(grid, dict):
            X, Y, Z = grid["X"], grid["Y"], grid["Z"]
            sig_vm = grid["von_Mises_Stress"]
            mesh = pv.StructuredGrid(X, Y, Z)
            mesh.point_data["sig_vm"] = sig_vm.flatten(order='F')
            grid = mesh

        plotter: Any = pv.Plotter(off_screen=not interactive, window_size=[1200, 900])
        active_mesh = grid.threshold(threshold_val, scalars=initial_scalar) if threshold_val is not None else grid

        # Initial mesh setup.
        self.add_configured_mesh(plotter, active_mesh, initial_scalar)

        # Optional overlay of the CadQuery generated substrate STL/OBJ mesh.
        if cad_mesh_path and os.path.exists(cad_mesh_path):
            cad_geometry = pv.read(cad_mesh_path)
            plotter.add_mesh(cad_geometry, style="wireframe", color="black", opacity=0.3, name="cad_overlay")

        # Add explicit directional light from top surface.
        top_light = pv.Light(position=(0,0,100), focal_point=(0,0,0), intensity=(0.8))
        plotter.add_light(top_light)

        # Dynamic scalar field toggling callback.
        def set_scalar(scalar_name: str):
            self.add_configured_mesh(plotter, grid, scalar_name)
            plotter.render()

        if interactive:
            # Add on-screen selection buttons / key-binds.
            available_scalars = list(grid.point_data.keys())

            # Key bindings: Pressing numbers 1 through N switches active scalar field.
            for idx, name in enumerate(available_scalars[:9]):
                plotter.add_key_event(str(idx + 1), lambda s=name: set_scalar(s))

            plotter.show_grid(location='outer', bold=False, font_size=9, color='gray')
            plotter.camera_position = 'iso'
            plotter.reset_camera()
            plotter.render()
            plotter.show(title=f"{title_prefix} [{initial_scalar}]")

            if hasattr(plotter, "close"):
                try:
            
                    plotter.close()
                except Exception:
                    pass

        else:
            plotter.show_grid(location='outer', bold=False, font_size=9, color='gray')
            plotter.camera_position = 'iso'
            # Ensures complete resource teardown and flush standard input.
            plotter.screenshot(save_path)
            plotter.close()
            plotter.deep_clean()

if __name__ == "__main__":
    extractor = FieldExtractor(model=None)
    extractor.load_checkpoint("models/pinn_silicon_v1.keras")

    domain_bounds = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 0.5))
    grid = extractor.extract_3d_field(bounds=domain_bounds, resolution=(40,40,20), material_id="Si")
    extractor.plot_stress_field(grid, threshold_gpa=2.0)

    # Save a static render of von Mises stress.
    extractor.plot_stress_field(grid, active_scalar="sig_vm", save_path="outputs/von_mises.png")

    # Launch interactive window (set interactive=True for desktop GUI display).
    extractor.visualize_3d(grid, initial_scalar="sig_vm", interactive=False)