"""
FEA Mesh Visualization Engine
=============================
Provides PyVista rendering routines for post-processing thermo-mechanical 
stress meshes, vector displacement warping, iso-contours, and HUD metrics.
"""

import pyvista as pv
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, cast

class FEAMeshVisualizer:
    """Enhanced PyVista post-processor for thermo-mechanical stress meshes."""

    def __init__(self, theme: str = "document") -> None:
        pv.set_plot_theme(theme)

    def render_enhanced_fea(
            self,
            mesh: pv.UnstructuredGrid,
            user_prefs: Dict[str, Any],
            output_png: Optional[Path] = None,
            interactive: bool = True
    ) -> None:
        """Renders FEA mesh displacement warping, stress contours, and metric overlays."""
        scalar_field = user_prefs.get("scalar_field", "sig_vm")
        warp_factor = float(user_prefs.get("warp_factor", 10.0))
        n_contours = int(user_prefs.get("n_contours", 10))
        cmap = user_prefs.get("colormap", "inferno")
        title = user_prefs.get("title", "Thermo-Mechanical Stress Analysis")

        plotter = pv.Plotter(window_size=[1280,720], off_screen=not interactive)

        # 1. Extract Raw Point Scalars Early.
        raw_scalars = np.array(mesh.point_data.get(scalar_field, [0.0]))
        raw_max = float(np.max(raw_scalars)) if len(raw_scalars) > 0 else 0.0

        # 2. Dynamic Unit Scaling: Convert GPa or Pa to MPa.
        if 0.0 < raw_max < 1e-4:        # Extreme low stress -> kPa
            scale_factor = 1e6          # GPa -> kPa
            unit_label = "kPa"
        elif 1e-4 <= raw_max < 1e-1:    # standard micro-electronics -> MPa
            scale_factor = 1000.0       # GPa -> MPa
            unit_label = "MPa"
        elif raw_max > 1e3:             # Raw Pa input
            scale_factor = 1e-6         # Pa -> MPa
            unit_label = "MPa"
        else:
            scale_factor = 1.0
            unit_label = "MPa"

        scalars_scaled = raw_scalars * scale_factor
        peak_stress_scaled = float(np.max(scalars_scaled)) if len(scalars_scaled) > 0 else 0.0
        min_stress_scaled = float(np.min(scalars_scaled)) if len(scalars_scaled) > 0 else 0.0

        # 3. Vector Displacement Warping.
        render_mesh: pv.DataSet = (
            cast(pv.DataSet, mesh.warp_by_vector("displacement", factor=warp_factor))
            if "displacement" in mesh.point_data
            else mesh.copy()
        )
        render_mesh.point_data["sig_vm_scaled"] = scalars_scaled

        # 4. Base Scalar Mesh Rendering.
        plotter.add_mesh(
            render_mesh,
            scalars="sig_vm_scaled",
            cmap=cmap,
            show_edges=True,
            edge_color="#333333",
            line_width=0.5,
            clim=[min_stress_scaled, max(peak_stress_scaled, 1e-4)],
            scalar_bar_args={
                "title": f"Von Mises Stress ({unit_label})",
                "vertical": True,
                "position_x": 0.85,
                "position_y": 0.15,
                "fmt": "%.2f" if unit_label == "MPa" else "%.1f"
            }
        )

        # 5. Dynamic Iso-Stress Contour Overlay.
        try:
            if peak_stress_scaled > min_stress_scaled:
                # Interpolate cell scalars to point data if point array is missing.
                if "sig_vm_scaled" not in render_mesh.point_data:
                    contour_target = render_mesh.cell_data_to_point_data()
                else:
                    contour_target = render_mesh

                contours = contour_target.contour(
                    isosurfaces=n_contours, 
                    scalars="sig_vm_scaled"
                )
                if contours.n_points > 0:
                    plotter.add_mesh(
                        cast(pv.DataSet, contours),
                        color="cyan",
                        line_width=1.5,
                        label="Iso-Stress Contours",
                        render_lines_as_tubes=True
                    )
        except Exception as e:
            print(f"⚠️ Unable to compute contours: {e}")

        # 6. Interactive HUD Text & Safety Metrics Overlay.
        plotter.add_text(title, position="upper_left", font_size=14, color="black")

        peak_stress_mpa = (peak_stress_scaled / 1000.0 if unit_label == "kPa" else peak_stress_scaled)
        yield_limit = float(user_prefs.get("yield_limit_mpa", 220.0))
        margin = (yield_limit / peak_stress_mpa) - 1.0 if peak_stress_mpa > 1e-6 else 999.0

        hud_metrics={
            r"Peak Stress ($\sigma_{max}$)": f"{peak_stress_scaled:.2f} {unit_label}",
            "Yield Limit": f"{yield_limit:.2f} MPa",
            "Margin of Safety": f"{margin:.2f}",
            "Max Bowing ΔZ": f"{warp_factor:.1f}x",
            "IMC Layer Growth": f"{float(user_prefs.get('imc_thickness_um', 0.0)):.2f} µm",
            "Darveaux Total Life": f"{float(user_prefs.get('fatigue_life_cycles', 0.0)):.0f} cycles"
        }

        hud_text = "--- METRICS HUD ---\n" + "\n".join(f"{k}: {v}" for k, v in hud_metrics.items())
        plotter.add_text(hud_text, position="upper_right", font_size=10, color="darkblue", shadow=True)

        # 7. Scene Settings and Output Export.
        plotter.add_axes(interactive=False) # type: ignore
        plotter.show_grid(color="gray")     # type: ignore

        if output_png:
            output_png.parent.mkdir(parents=True, exist_ok=True)

        if interactive:
            plotter.show(interactive_update=True)
            if output_png:
                plotter.screenshot(str(output_png))
                print(f"✅ FEA render saved to: {output_png}")
            plotter.show()
        else:
            # Headless execution.
            plotter.show(screenshot=str(output_png), auto_close=True)
            print(f"✅ FEA render saved to: {output_png}")