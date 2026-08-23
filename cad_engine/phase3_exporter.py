"""
Phase 3 PCB Exporter
====================
Exports layout definitions to native KiCad layout files (.kicad_pcb) and calls 
KiCad CLI utilities to produce production-ready Gerber and drill packages.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import subprocess
import numpy as np
import shutil
import sys

def get_kicad_cli() -> Path:
    """Dynamically locates the kicad-cli executable across system PATH and default install directories."""
    # 1. Check system PATH first.
    kicad_path = shutil.which("kicad-cli")
    if kicad_path:
        return Path(kicad_path)

    # 2. Fallback to standard platform paths.
    if sys.platform == "win32":
        default_paths = [
            Path(
                r"C:\Program Files\KiCad\10.0\kicad-cli.exe"
            ),  # KiCad 10 Standard.
            Path(
                r"C:\Program Files\KiCad\9.0\kicad-cli.exe"
            ),  # KiCad 9 Standard.
            Path(
                r"C:\Program Files\KiCad\8.0\kicad-cli.exe"
            ),   # KiCad 8 Standard.
            Path(
                os.path.expanduser(
                    r"~\AppData\Local\Programs\KiCad\10.0\bin\kicad-cli.exe"
                )
            ),  # User AppData
            Path(
                os.path.expanduser(
                    r"~\AppData\Local\Programs\KiCad\8.0\bin\kicad-cli.exe"
                )
            ),
        ]
        for path in default_paths:
            if path.exists():
                return path
            
    return Path("kicad-cli")

class Phase3PCBExporter:
    """Exports dynamic PCB layouts to KiCad format and packages Gerber manufacturing files."""

    def __init__(
            self,
            preset_name: str = "default",
            config_path: Optional[Path] = None,
            rules: Optional[Dict[str, Any]] = None
    ) -> None:
        self.preset_name = preset_name
        self.config_path = config_path
        self.rules = rules or {}

        # Load rules from file if provided and empty.
        if not self.rules and self.config_path and Path(self.config_path).exists():
            with open(self.config_path, "r") as f:
                self.rules = json.load(f).get(preset_name, {})

    def generate_kicad_pcb(self, board_name: str, output_dir: str = "outputs", board_size_mm: float = 40.0) -> str:
        """Generates a KiCad PCB layout file (.kicad_pcb) on mission design rules."""
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{board_name}.kicad_pcb")

        clearance_mm = float(self.rules.get("min_trace_clearance_um", 100.0)) / 1000.0
        via_pitch_mm = float(self.rules.get("via_pitch_mm", 1.5))

        # Dynamic geometry calculations.
        b_max = float(board_size_mm)
        center = b_max / 2.0

        # Dynamic Thermal Via Array Generation.
        via_statements = []
        via_span = np.arange(center - 5.0, center + 5.0 + 1e-5, via_pitch_mm)
        for vx in via_span:
            for vy in via_span:
                via_statements.append(
                    f'  (via (at {vx:.3f} {vy:.3f}) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (net 1))\n'
                )
        vias_str = "".join(via_statements)

        # Dynamic Header & Layers setup.
        kicad_content = (
            f'(kicad_pcb (version 20240108) (generator micro_dashboard)\n'
            f'  (general\n'
            f'    (thickness 1.6)\n'
            f'  )\n'
            f'  (paper "A4")\n'
            f'  (layers\n'
            f'    (0 "F.Cu" signal)\n'
            f'    (31 "B.Cu" signal)\n'
            f'    (32 "B.Adhes" user "B.Adhesive")\n'
            f'    (33 "F.Adhes" user "F.Adhesive")\n'
            f'    (34 "B.Paste" user)\n'
            f'    (35 "F.Paste" user)\n'
            f'    (36 "B.SilkS" user "B.Silkscreen")\n'
            f'    (37 "F.SilkS" user "F.Silkscreen")\n'
            f'    (38 "B.Mask" user)\n'
            f'    (39 "F.Mask" user)\n'
            f'    (44 "Edge.Cuts" user)\n'
            f'  )\n'
            f'  (setup\n'
            f'    (pad_to_mask_clearance {clearance_mm:.4f})\n'
            f'    (pcbplotparams\n'
            f'      (layerselection 0x00010fc_ffffffff)\n'
            f'      (disablegerberextensions false)\n'
            f'      (usegerberattributes true)\n'
            f'      (usegerberadvancedattributes true)\n'
            f'      (creategerberjobfile true)\n'
            f'    )\n'
            f'  )\n'
            f'  (net 0 "")\n'
            f'  (net 1 "GND")\n'
            f'  (net 2 "VCC")\n'
            f'  (net 3 "LED_NET")\n'

            # Board Outline
            f'  (gr_line (start 0 0) (end {b_max:.2f} 0) (layer "Edge.Cuts") (width 0.1))\n'
            f'  (gr_line (start {b_max:.2f} 0) (end {b_max:.2f} {b_max:.2f}) (layer "Edge.Cuts") (width 0.1))\n'
            f'  (gr_line (start {b_max:.2f} {b_max:.2f}) (end 0 {b_max:.2f}) (layer "Edge.Cuts") (width 0.1))\n'
            f'  (gr_line (start 0 {b_max:.2f}) (end 0 0) (layer "Edge.Cuts") (width 0.1))\n'

            # Components
            f'  (footprint "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical" (layer "F.Cu")\n'
            f'    (at 5 {center:.2f})\n'
            f'    (property "Reference" "J1" (at 0 -2.5 0) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))\n'
            f'    (property "Value" "Power_In" (at 0 5 0) (layer "F.Fab") (effects (font (size 1 1) (thickness 0.15))))\n'
            f'    (pad "1" thru_hole rect (at 0 0) (size 1.7 1.7) (drill 1.0) (layers "*.Cu" "*.Mask") (net 2 "VCC"))\n'
            f'    (pad "2" thru_hole circle (at 0 2.54) (size 1.7 1.7) (drill 1.0) (layers "*.Cu" "*.Mask") (net 1 "GND"))\n'
            f'  )\n'

            # Decoupling Capacitor C1 (0805)
            f'  (footprint "Capacitor_SMD:C_0805_2012Metric" (layer "F.Cu")\n'
            f'    (at {center - 7:.2f} {center:.2f})\n'
            f'    (property "Reference" "C1" (at 0 -1.5 0) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))\n'
            f'    (property "Value" "100nF" (at 0 1.5 0) (layer "F.Fab") (effects (font (size 1 1) (thickness 0.15))))\n'
            f'    (pad "1" smd rect (at -1.0 0) (size 1.0 1.3) (layers "F.Cu" "F.Paste" "F.Mask") (net 2 "VCC"))\n'
            f'    (pad "2" smd rect (at 1.0 0) (size 1.0 1.3) (layers "F.Cu" "F.Paste" "F.Mask") (net 1 "GND"))\n'
            f'  )\n'

            # IC U1 (Centrally Placed)
            f'  (footprint "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm" (layer "F.Cu")\n'
            f'    (at {center:.2f} {center:.2f})\n'
            f'    (property "Reference" "U1" (at 0 -3.5 0) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))\n'
            f'    (property "Value" "IC" (at 0 3.5 0) (layer "F.Fab") (effects (font (size 1 1) (thickness 0.15))))\n'
            f'    (pad "1" smd rect (at -2.47 -1.905) (size 1.55 0.6) (layers "F.Cu" "F.Paste" "F.Mask") (net 2 "VCC"))\n'
            f'    (pad "4" smd rect (at -2.47 1.905) (size 1.55 0.6) (layers "F.Cu" "F.Paste" "F.Mask") (net 1 "GND"))\n'
            f'  )\n'

            # Traces & Vias
            f'  (segment (start 5 {center:.2f}) (end {center - 7:.2f} {center:.2f}) (width 0.5) (layer "F.Cu") (net 2))\n'
            f'  (segment (start {center - 7:.2f} {center:.2f}) (end {center - 2.47:.2f} {center - 1.905:.2f}) (width 0.5) (layer "F.Cu") (net 2))\n'
            f'{vias_str}'

            # Bottom Copper GND Plane
            f'  (zone (net 1) (net_name "GND") (layer "B.Cu") (hatch edge 0.5)\n'
            f'    (connect_pads (clearance {clearance_mm:.4f}))\n'
            f'    (min_thickness 0.25)\n'
            f'    (polygon\n'
            f'      (pts\n'
            f'        (xy 0 0) (xy {b_max:.2f} 0) (xy {b_max:.2f} {b_max:.2f}) (xy 0 {b_max:.2f})\n'
            f'      )\n'
            f'    )\n'
            f'  )\n'
            f')\n'
        )
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(kicad_content)

        print(f"✅ KiCad PCB layout generated: {file_path}")
        return file_path

    def export_manufacturing_pack(
            self,
            kicad_path_path: str,
            export_subdir: str = "gerbers_pcb",
            output_dir: str = "outputs",
    ) -> str:
        """Exports Gerber layers and drill files via KiCad CLI if executable is present."""
        gerber_path = (Path(output_dir) / export_subdir).resolve()
        gerber_path.mkdir(parents=True, exist_ok=True)
        pcb_file = Path(kicad_path_path).resolve()

        # Generate manifest summary for manufacturing verification.
        manifest_path = gerber_path / "manifest.json"
        manifest_data = {
            "preset": self.preset_name,
            "source_pcb": str(pcb_file),
            "applied_rules": self.rules,
            "Gerber_layers": ["F.Cu", "B.Cu", "F.SilkS", "B.SilkS", "F.Mask", "B.Mask", "Edge.Cuts"]
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=4)

        # Dynamic path detection for KiCad CLI.
        kicad_cli = get_kicad_cli()

        if (kicad_cli.exists() or shutil.which(str(kicad_cli))) and pcb_file.exists():
            try:
                # 1. Export Gerber layers.
                subprocess.run([
                    str(kicad_cli), "pcb", "export", "gerbers",
                    str(pcb_file),
                    "-o", f"{gerber_path}{os.sep}"], 
                    check=True)

                # 2. Export Drill files.
                subprocess.run([
                    str(kicad_cli), "pcb", "export", "drill",
                    str(pcb_file),
                    "-o", f"{gerber_path}{os.sep}"], 
                    check=True)

                print(f"✅ Gerber Manufacturing pack exported via KiCad CLI to: {gerber_path}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ KiCad CLI export failed: {e}")
        else:
            print(f"⚠️ KiCad CLI executable or PCB file not found. Skipping Gerber plotting.")

        return str(gerber_path)

if __name__ == "__main__":
    # CLI accepts the profile key dynamically from user input or orchestration scripts.
    parser = argparse.ArgumentParser(description="Micro-Dashboard Phase 3 PCB Exporter")
    parser.add_argument("--preset", type=str, required=True, help="Mission profile key")
    parser.add_argument("--rules_file", type=str, default="outputs/design_rules.json", help="Path to rules JSON")
    args = parser.parse_args()

    rules_file = Path(args.rules_file)
    active_rules = {}

    if rules_file.exists():
        with open(rules_file, "r", encoding="utf-8") as f:
            active_rules = json.load(f).get(args.preset, {})

    exporter = Phase3PCBExporter(
        preset_name=args.preset,
        rules=active_rules,
        config_path=rules_file
        )
    pcb = exporter.generate_kicad_pcb(f"pcb_{args.preset}")
    exporter.export_manufacturing_pack(pcb, f"gerbers_pcb_{args.preset}")