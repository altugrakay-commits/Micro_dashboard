"""
KLayout GDSII Pattern Generator
===============================
Generates micro-feature GDSII layouts (thermal relief pads, die boundaries) 
based on thermal-stress optimization parameters.
"""

from pathlib import Path
from typing import Union
import os
try:
    import klayout.db as kdb
    KLAYOUT_AVAILABLE = True
except ImportError:
    KLAYOUT_AVAILABLE = False

class KLayoutGenerator:
    """Generates micro-feature GDSII Layouts based on stress hotspot metrics."""

    def __init__(self, output_dir: str="outputs/Layouts") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_die_layout(
            self,
            filename: str="die_thermal_relief.gds",
            pad_width_um: float=1000.0,
            buffer_margin_um: float=100.0
    ) -> str:
        """Creates a GDSII die pad with adaptive thermal relief margins."""
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if not KLAYOUT_AVAILABLE:
            # Fallback mock generator if klayout engine isn't installed in the environment.
            txt_path = filepath.with_suffix(".gds.txt")
            txt_path.write_text(
                f"Mock GDSII Layout: Pad Width={pad_width_um}um, Buffer={buffer_margin_um}um\n",
                        encoding = "utf-8")
            print(f"⚠️ KLayout module not found. Created fallback layout text file at: {filepath}")
            return str(txt_path)

        db = kdb.Layout()
        db.dbu = 0.001 # 1 unit = 1nm
        top_cell = db.create_cell("TOP")

        # Layer 1: Silicon Die Boundary.
        l_buffer = db.insert_layer(kdb.LayerInfo(2,0))
        margin_dbu = int(buffer_margin_um * 1000)
        pad_dbu = int(pad_width_um * 1000)

        buffer_box = kdb.Box(-margin_dbu, -margin_dbu, pad_dbu + margin_dbu, pad_dbu + margin_dbu)
        top_cell.shapes(l_buffer).insert(buffer_box)

        db.write(str(filepath))
        print(f"✅ GDSII Layout exported successfully to: {filepath}")
        return str(filepath)