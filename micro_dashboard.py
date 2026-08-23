"""
Micro-Dashboard Orchestrator
============================
Links multi-physics PINN evaluations, dynamic layout stress optimization,
and downstream CAD/ECAD generation pipelines (STEP, GDSII, KiCad, Gerber)
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Suppresses TF INFO, WARNING, and ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import seaborn as sns
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from cad_engine.thermal_stress_reducer import (ThermalStressOptimizer, ThermalStressReducer)
from cad_engine.gerber_kicad_visualizer import GerberBoardVisualizer
from cad_engine.phase3_exporter import Phase3PCBExporter
from cad_engine.fea_visualizer import FEAMeshVisualizer
from cad_engine.layout_builder import KLayoutGenerator
from cad_engine.pcb_builder import PCBBuilder

from physics.degradation_engine import DegradationEngine
from physics.material_loader import MaterialDatabase
from physics.post_processor import FieldExtractor
from physics.stress_engine import StressEngine

# VISUAL STYLING SETUP
sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

# Directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "automotive": {
        "name": "Automotive Engine Bay", 
        "delta_T_k": 115.0, 
        "ambient_temp_c": 85.0, 
        "vibration_g_rms": 12.0, 
        "primary_failure_mode": "Solder Fatigue & thermo-mechanical substrate delamination",
        "barrier_type": "Ni_Au",
        "target_cycles": 3000.0,
        "cycling_frequency_hz": 0.00027778,
    },
    "aerospace": {
        "name": "Aerospace Avionics",
        "delta_T_k": 180.0,
        "ambient_temp_c": 125.0,
        "vibration_g_rms": 12.0,
        "primary_failure_mode": "Die Cracking",
        "barrier_type": "TaN_W",
        "target_cycles": 10000.0,
        "cycling_frequency_hz": 0.00013889,
    },
    "defense": {
        "name": "Defense Missile Guidance",
        "delta_T_k": 210.0,
        "ambient_temp_c": 150.0,
        "vibration_g_rms": 20.0,
        "primary_failure_mode": "Substrate Delamination",
        "barrier_type": "TaN_W",
        "target_cycles": 25000.0,
        "cycling_frequency_hz": 0.001,
    },
    "consumer": {
        "name": "Consumer Mobile Phone",
        "delta_T_k": 45.0,
        "ambient_temp_c": 40.0,
        "vibration_g_rms": 1.5,
        "primary_failure_mode": "Thermal Throttling",
        "barrier_type": "none",
        "target_cycles": 1500.0,
        "cycling_frequency_hz": 0.0001,
    },
    "industrial": {
        "name": "Industrial Power Inverter",
        "delta_T_k": 95.0,
        "ambient_temp_c": 85.0,
        "vibration_g_rms": 3.5,
        "primary_failure_mode": "IMC Creep",
        "barrier_type": "Ni_Au",
        "target_cycles": 5000.0,
        "cycling_frequency_hz": 0.00005,
    },
}

class MicroDashboardOrchestrator:
    """Agentic Orchestrator linking mission profiles, PINN physics evaluation, and CAD synthesis."""

    def __init__(
            self,
            config_path: str = "config/mission_profiles.json",
            model_path: str = "models/pinn_silicon_v1.keras"
    ):
        self.config_path = Path(config_path)
        self.model_path = Path(model_path)
        self.profiles = self._load_profiles()

        # Initialize Physics Engines.
        self.db = MaterialDatabase()
        self.stress_engine = StressEngine(material_db=self.db)
        # Default kinetic activation parameters for solder/Cu interface (D0 in m^2/s, Q in J/mol)
        self.degradation_engine = DegradationEngine(stress_engine=self.stress_engine)

        # Initialize CAD and Optimization Engines.
        self.klayout_gen = KLayoutGenerator()
        self.stress_reducer = ThermalStressReducer(
            stress_engine = self.stress_engine,
            imc_kinetics = self.degradation_engine.imc_kinetics
        )
        self.optimizer = ThermalStressOptimizer()

    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Loads environmental mission profiles with fallback protection."""
        search_paths = [
            Path(__file__).resolve().parent / self.config_path,
            Path.cwd() / self.config_path
        ]

        for path in search_paths:
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        
        print(f"⚠️ Warning: Mission profiles JSON not found at {self.config_path}. Loading defaults.")
        return DEFAULT_PROFILES

    def audit_design_integrity(
        self, pcb_rules: Dict[str, Any], profile: Dict[str, Any]
    ) -> Tuple[bool, list[str]]:
        """Audits mechanical, geometric clearance, and thermal stress constraints."""
        logs: List[str] = []
        is_valid = True

        checks = [
            (
                pcb_rules.get("min_trace_clearance_um", 100.0) < 80.0,
                f"Trace clearance ({pcb_rules.get('min_trace_clearance_um', 100.0):.1f} µm) below safety threshold (80.0 µm).",
            ),
            (
                pcb_rules.get("min_bend_radius_mm", 0.3) < 0.15,
                f"Bend radius ({pcb_rules.get('min_bend_radius_mm', 0.3):.2f} mm) tto sharp; stress risk.",
             ),
             (
                 profile["delta_T_k"] > 150.0
                 and pcb_rules.get("min_dielectric_thickness_um", 100.0) < 75.0,
                 f"Dielectric thickness ({pcb_rules.get('min_dielectric_thickness_um', 100.0):.1f} µm) is insufficient for ∆T = {profile['delta_T_k']} K."
             ),
        ]

        for failed, message in checks:
            if failed:
                logs.append(message)
                is_valid = False

        if is_valid:
            logs.append("All mechanical and layout integrity checks passed successfully.")

        return is_valid, logs

    def run_agentic_co_optimization(self, env_key: str, max_retries: int = 3) -> bool:
        """Executes closed-loop CAD rule synthesis QA checks."""
        profile = self.profiles[env_key]
        print(f"\n🚀 Initializing Agentic Co-Optimization [Preset: {env_key}]")

        delta_t=float(profile["delta_T_k"])
        # Retrieve barrier_type from loaded profile (defaulting to Ni_Au).
        oper_temp_k = float(profile["ambient_temp_c"]) + 273.15
        barrier_type = str(profile.get("barrier_type", "Ni_Au"))

        stackup_rules = self.stress_reducer.calculate_pcb_stack_rules(delta_T=delta_t)
        # Use full degradation evaluation to align IMC growth with step 2.
        deg_res = self.degradation_engine.evaluate_interface_degradation(
            temp_k=oper_temp_k,
            operating_hours=8760.0, # 1 Year
            shear_strain_range=0.005,
            barrier_type=barrier_type
        )
        imc_growth_m = deg_res["imc_thickness_um"] * 1e-6 # type: ignore

        cad_rules = self.stress_reducer.calculate_cad_design_rules(
            mat1="Cu", 
            mat2="Si",
            delta_T=delta_t,
            imc_thickness_m=imc_growth_m
        )
        combined_rules = {**cad_rules, **stackup_rules}

        # Cache dynamic design rules.
        rules_path = Path(OUTPUT_DIR) / "design_rules.json"
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump({env_key: combined_rules}, f, indent=4)

        # Quality Assurance Audit.
        for attempt in range(1, max_retries + 1):
            print(f"\n--- QA Co-Optimization Iteration {attempt}/{max_retries} ---")
            # Instantiate PCBBuilder using current target preset.
            pcb_builder = PCBBuilder(preset_name=env_key)
            pcb_builder.rules = combined_rules

            # Run Quality Control Inspection.
            qa_passed, audit_logs = self.audit_design_integrity(combined_rules, profile)
            for log in audit_logs:
                print(f" {'✅' if qa_passed else '⚠️'}{log}")

            if qa_passed:
                print("\n✅ QA Checks Passed: Design complies with thermal-mechanical bounds.")
                return True
            else:
                print(f"🔧 Agent applying parametric adjustments for iteration {attempt + 1}...")
                # Dynamically expand clearance and dielectric bounds to relax stress.
                combined_rules["min_trace_clearance_um"] += 25.0
                combined_rules["min_dielectric_thickness_um"] += 20.0

        print("❌ Agentic loop reached maximum iterations without full QA convergence.")
        return False

    def run_mission_pipeline(self, env_key: str, render_3d: bool = True) -> None:
        """Executes PINN physics evaluation, CAD synthesis, and export pipelines."""
        profile = self.profiles[env_key]
        delta_t = float(profile["delta_T_k"])
        oper_temp_k = float(profile["ambient_temp_c"]) + 273.15
        barrier_type = str(profile.get("barrier_type", "Ni_Au"))

        profile = self.profiles[env_key]
        print(f"\n============================================")
        print(f" EXECUTING PIPELINE: {profile['name'].upper()}")
        print(f" Failure Mode: {profile['primary_failure_mode']}")
        print(f" ΔT: {'delta_T_k'} K | Ambient: {profile['ambient_temp_c']}°C")
        print(f"\n============================================")

        # Run Agentic QA loop first.
        self.run_agentic_co_optimization(env_key)

        # 1. Physics Evaluation via FieldExtractor & FEA Rendering.
        print("\n1. Evaluating PINN Stress Fields...")
        extractor = FieldExtractor(model_path=str(self.model_path))

        # 1a. Extract PyVista StructureGrid containing scalar fields.
        grid_mesh = extractor.extract_3d_field(
            bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 0.5)),
            resolution=(25, 25, 10),
            delta_T=delta_t
        )

        # Dynamic Stress Scaling: Multiply normalized mesh scalars by environmental Delta T.
        grid_mesh.point_data["sig_vm"] *= delta_t / 100.0
        base_peak_stress = float(np.max(grid_mesh.point_data["sig_vm"]))
        print(f" Baseline Peak Von Mises Stress: {base_peak_stress:.3f} GPa")

        # 2. Stress Optimization Loop & Reduction Charting
        print("\n2. Executing Thermal Stress Optimization Loop...")
        final_stress = self.optimizer.run_optimization_loop(
            base_stress_gpa=base_peak_stress,
            delta_t_k=delta_t,
            preset_name=env_key,
            steps=6
        )

        # Compute stress scaling factor from optimization.
        stress_scale_ratio = final_stress / base_peak_stress

        # 3. Intermetallic Kinetics & Interface Shear Calculation
        print("\n3. Evaluating Interface Kinetics & Low-Cycle Thermal Fatigue...")
        # Compute effective strain mitigation.
        solder_k_v = self.stress_engine.compute_solder_geometry_factor(
            geometry_profile="hourglass", height_um=120.0, diameter_um=150.0
        )
        relief_gap_um = self.stress_reducer.calculate_required_relief_gap(
            mat1="Cu", mat2="Si", delta_T=delta_t, length=10e-3,h_imc_m=1e-6
        )

        # Base strain passed directly.
        mitigated_shear_strain = 0.005 * solder_k_v * stress_scale_ratio
        # Extract profile-specific target cycles (for default dynamically, if not present).
        freq_hz = float(profile.get("cycling_frequency_hz", 1.0 / 3600.0))
        target_cycles = float(profile.get("target_cycles", 1000.0))

        # 2a. Calculate required mechanical relief gap via ThermalStressReducer
        target_hours = (target_cycles / freq_hz) / 3600.0
        deg_results = self.degradation_engine.evaluate_interface_degradation(
            temp_k=oper_temp_k,
            operating_hours=target_hours,
            shear_strain_range=mitigated_shear_strain,  # Dynamic strain range.
            relief_gap_um=relief_gap_um,
            barrier_type=profile.get("barrier_type", "Ni_Au"),
            n_target=target_cycles,
            cycling_frequency_hz=freq_hz
        )

        nl_data = deg_results["norris_landzberg_cycles"]    # Extracts dict returned by evaluate_norris_landzberg
        darveaux_data = deg_results["darveaux_fatigue"]

        if not isinstance(darveaux_data, dict) or not isinstance(nl_data, dict):
            raise TypeError("Expected dictionary outputs for fatigue metrics from degradation engine.")

        # Scale mesh scalar field for HUD display.
        grid_mesh.point_data["sig_vm"] *= stress_scale_ratio

        # FEA Stress Rendering.
        fea_user_config = {
            "scalar_field": "sig_vm",
            "warp_factor": float(np.clip(profile["delta_T_k"] / 15.0, 5.0, 25.0)),
            "colormap": "inferno",
            "yield_limit_mpa": float(profile.get("yield_limit_mpa", 220.0)),
            "title": f"FEA Stress Analysis - {profile['name']}",
            "imc_thickness_um": deg_results["imc_thickness_um"],
            "fatigue_life_cycles": darveaux_data["total_fatigue_life_cycles"]
        }
        fea_png_path = Path(OUTPUT_DIR) / f"fea_stress_{env_key}.png"
        FEAMeshVisualizer().render_enhanced_fea(
            mesh=grid_mesh.cast_to_unstructured_grid(),
            user_prefs=fea_user_config,
            output_png=fea_png_path
        )
        pv.close_all()

        print(f" Calculated IMC Layer Growth (1 Yr):    {deg_results['imc_thickness_um']:.3f} µm")
        print(f" Darveaux Crack Initiation Life:        {darveaux_data['crack_initiation_cycles']:.0f} cycles")
        print(f" Darveaux Total Life to Failure:        {darveaux_data['total_fatigue_life_cycles']:.0f} cycles")
        print(f" Norris-Landzberg Fatigue Life:         {nl_data['norris_landzberg_cycles']:.0f} cycles (Target: {nl_data['target_cycles']:.0f})")
        print(f" Norris-Landzberg Fatigue Penalty:      {nl_data['fatigue_penalty']:.4f}")
        print(f" Applied Dynamic Relief Gap:            {relief_gap_um:.2f} µm")

        # 4. CAD GDSII Layout Generation
        print("\n4. Generating Micro-Structure Layout (KLayout GDSII)...")
        # Adaptive dimensions based on optimized state.
        vibr_g = float(profile["vibration_g_rms"])
        dynamic_buffer_um = float(np.clip(delta_t * 1.25, 80.0, 350.0))
        dynamic_pad_um = float(800.0 + (vibr_g * 25.0))

        layout_file = self.klayout_gen.generate_die_layout(
                filename=f"layout_{env_key}.gds",
                pad_width_um=dynamic_pad_um,
                buffer_margin_um=dynamic_buffer_um
        )

        # 5. STEP 3D CAD Synthesis
        print("\n5. Generating 3D PCB Substrate & Thermal Via Stack (CadQuery STEP)...")
        board_dim_mm = float(40.0 + (profile["vibration_g_rms"] * 0.5))
        dynamic_thickness = float(np.clip(1.6 - (delta_t * 0.002), 1.0, 1.6))

        # 5a. Instantiate PCBBuilder for final execution.
        design_rules = self.stress_reducer.calculate_cad_design_rules(
            mat1="Cu", mat2="Si", delta_T=delta_t, imc_thickness_m=deg_results["imc_thickness_um"] * 1e-6 # type: ignore
        )
        target_via_pitch = design_rules.get("via_pitch_mm", 1.5)

        pcb_builder = PCBBuilder(preset_name=env_key)
        pcb_builder.rules = design_rules
        pcb_builder.populate_default_components(die_pad_size_mm=dynamic_pad_um / 1000.0)

        pcb_file = os.path.normpath(
            pcb_builder.generate_pcb_substrate(
                filename=f"pcb_{env_key}.step",
                board_size_mm = (board_dim_mm, board_dim_mm),
                thickness_mm=dynamic_thickness,
                die_pad_size_mm = dynamic_pad_um / 1000.0,      # Convert µm to mm.
                via_pitch_mm = target_via_pitch,                # Driven by thermal delta
                via_radius_mm = 0.3
            )
        )

        # 5b. Auto-display Pop-Up Window for the populated CAD assembly.
        parts_dir = Path(pcb_builder.output_dir) / f"{Path(pcb_file).stem}_parts"
        stl_mesh_file = Path(pcb_builder.output_dir) / f"{Path(pcb_file).stem}.stl"

        # 3D Model Rendering.
        parts_dir = Path(pcb_builder.output_dir) / f"{Path(pcb_file).stem}_parts"
        stl_file = Path(pcb_builder.output_dir) / f"{Path(pcb_file).stem}.stl"
        target_path = str(parts_dir) if parts_dir.exists() and any(parts_dir.glob("*stl")) else (str(stl_file) if stl_file.exists() else None)

        if target_path and render_3d:
            print(f"🖥️ Opening interactive 3D CAD window for {target_path}...")
            extractor.render_cad_mesh(target_path, interactive=True)
            pv.close_all() # Force VTK to destroy active render windows cleanly before ECAD export.
        else:
            print("⚠️ Skipping 3D CAD render: Valid mesh file/directory not found.")

        # 6. ECAD (Exports & Assembly Render.
        print("\n6. Synthesizing KiCad PCB & Gerber Manufacturing Pack...")
        exporter = Phase3PCBExporter(preset_name=env_key, config_path=Path(OUTPUT_DIR) / "design_rules.json")
        fatigue_life = darveaux_data["total_fatigue_life_cycles"] if isinstance(darveaux_data, dict) else float(darveaux_data)

        exporter.rules = {
            **design_rules,
            "imc_thickness_um": deg_results["imc_thickness_um"],
            "fatigue_life_cycles": darveaux_data["total_fatigue_life_cycles"]
        }

        kicad_path = exporter.generate_kicad_pcb(
            board_name=f"pcb_{env_key}", 
            board_size_mm=board_dim_mm
            )
        
        gerber_dir = exporter.export_manufacturing_pack(
            kicad_path,
            export_subdir=f"gerbers_pcb_{env_key}")

        # 6a. Dynamic 3D ECAD PCB Gerber Assembly Rendering.
        ecad_user_config = {
            "board_width": board_dim_mm,
            "board_height": board_dim_mm,
            "substrate_thickness": dynamic_thickness,
            "via_count": max(16, int((10.0 / target_via_pitch) ** 2)),
            "via_drill_radius": 0.3,
            "pad_size": dynamic_pad_um / 1000.0,
            "package_type": f"QFN Package ({profile['name']})",
            "substrate_color": "#1b4d2e"
        }

        ecad_png_path = Path(OUTPUT_DIR) / f"pcb_3d_assembly_{env_key}.png"
        gerber_viz = GerberBoardVisualizer()
        gerber_viz.render_3d_pcb_assembly(
            layers=gerber_viz.generate_board_stackup(ecad_user_config),
            config=ecad_user_config,
            title=f"3D PCB Assembly - {profile['name']}",
            output_png=ecad_png_path
        )
        pv.close_all()

        print("\n======================================\n")
        print(f"✅ PIPELINE RUN COMPLETE FOR:   {env_key}")
        print(f" Final Reduced Stress:          {final_stress:.3f} GPa")
        print(f" Output IC STEP Layout:         {pcb_file}")
        print(f" Output GDSII Layout:           {layout_file}")
        print(f" Output KiCad PCB:              {kicad_path}")
        print(f" Output Gerber Pack:            {gerber_dir}")
        print(f" FEA Stress Plot Render:        {fea_png_path}")
        print(f" 3D PCB Assembly Render:        {ecad_png_path}")
        print("\n=======================================\n")

    def interactive_menu(self) -> None:
        """Main interactive menu prompt."""
        keys = list(self.profiles.keys())
        while True:
            # Flush stdin before prompting to prevent leftover VTK keypresses from skipping input.
            FieldExtractor.flush_input_buffer()

            print("========================================")
            print(" MICRO-DASHBOARD: MISSION PROFILE SELECT")
            print("========================================")
            for idx, key in enumerate(keys, 1):
                p = self.profiles[key]
                print(f" [{idx}] {key.upper():<16} | {p['name']} (ΔT = {p['delta_T_k']} K)")
            print(" [Q] QUIT")
            print("----------------------------------------")

            choice = input("Select a mission profile (1-5 or Q): ").strip().lower()
            if choice == 'q':
                print("Exiting dashboard orchestrator. Goodbye!")
                # Exits interactive_menu immediately.
                break

            if choice.isdigit() and 1 <= int(choice) <= len(keys):
                self.run_mission_pipeline(keys[int(choice) - 1])
                FieldExtractor.flush_input_buffer()
                input("\nPress Enter to return to main menu...")
            else:
                print("⚠️ Invalid selection. Please enter a valid index or 'Q'.")


if __name__ == "__main__":
    # Command Line Interface execution.
    MicroDashboardOrchestrator().interactive_menu()