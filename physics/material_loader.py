"""
Material Engine Loader
======================
Handles loading, caching, and evaluation of mechanical and thermal properties 
for packaging materials from a structured CSV database or fallback dictionary.
"""

from pathlib import Path
from typing import Dict, Union, cast
import numpy as np                  # Handles numerical arrays
import pandas as pd                 # Loads CSVs into DataFrames

class MaterialDatabase:
    """Manages material constants, Voigt stiffness matrices, and CTE vectors."""

    def __init__(self, csv_path: Union[str, Path, None] = None) -> None:
        """Initializes the database from a CSV path or uses default fallbacks."""
        if csv_path is None:
            # Deterministically points to project root: micro_dashboard/materials_db.csv
            base_dir = Path(__file__).resolve().parent
            candidates = [
                base_dir / "materials_db.csv",
                base_dir.parent / "materials_db.csv",
                base_dir.parent / "materials" / "materials_db.csv"
            ]
            csv_path = next((p for p in candidates if p.is_file()), candidates[0])
        else:
            csv_path = Path(csv_path)

        if not csv_path.is_file():
            # Fallback inline default dataset creation if CSV is absent.
            self.df = pd.DataFrame({
                'Name': ["Si", "FR4", "Cu", "SAC305", "Cu6Sn%"],
                'YM_xy_Gpa_Max': [130.0, 24.0, 110.0, 50.0, 85.0],
                'YM_xy_Gpa_Min': [130.0, 18.0, 110.0, 40.0, 85.0],
                'YM_z_Gpa_Max': [130.0, 10.0, 110.0, 50.0, 85.0],
                'YM_z_Gpa_Min': [130.0, 7.0, 110.0, 40.0, 85.0],
                'Poissons_Ratio_Max': [0.28, 0.18, 0.34, 0.36, 0.31],
                'Poissons_Ratio_Min': [0.28, 0.11, 0.34, 0.36, 0.31],
                'CTE_xy_Max': [2.6e-6, 15e-6, 17e-6, 20e-6, 16.3e-6],
                'CTE_xy_Min': [2.6e-6, 12e-6, 17e-6, 20e-6, 16.3e-6],
                'CTE_z_Max': [2.6e-6, 50e-6, 17e-6, 20e-6, 16.3e-6],
                'CTE_z_Min': [2.6e-6, 45e-6, 17e-6, 20e-6, 16.3e-6],
                }).set_index('Name')
        else:
            # Reads CSV using semicolon delimiters and indexes on 'Name.'
            self.df = pd.read_csv(csv_path, sep=";").set_index("Name")
    
    # A helper method that takes a material string and returns all of its CSV columns as a key-value Python dictionary.
    def get_properties(self, material_name: str) -> Dict[str, float]:
        """Extracts property dictionary for a given material identifier."""
        if material_name not in self.df.index:
            raise KeyError(f"Material '{material_name}' not found. Available: {list(self.df.index)}")
        return cast(Dict[str, float], self.df.loc[material_name].to_dict())
    
    def get_stiffness_matrix(self, material_name: str, use_worst_case: bool=True) -> np.ndarray:
        """Generates the 6x6 Voigt Constitutive Stiffness Matrix [C] in Pascals
        Supports isotropic and transversely isotropic (orthotropic) materials."""
        props = self.get_properties(material_name)

        # Extract Young Moduli (converts GPa to Pascals).
        suffix = 'Max' if use_worst_case else 'Min'
        e_xy = props[f'YM_xy_Gpa_{suffix}'] * 1e9   # GPa converted to Pa
        e_z = props[f'YM_z_Gpa_{suffix}'] * 1e9
        nu = props[f'Poissons_Ratio_{suffix}']

        # Checks for Isotropic vs. Transversely Isotropic (Anisotropic)
        if np.isclose(e_xy, e_z):
            # --- ISOTROPIC CASE (Si, GaN, Cu, SAC305, Cu6Sn5) ---
            lam = (e_xy * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = e_xy / (2.0 * (1.0 + nu))
            return np.array(
            [
                [lam + 2*mu,    lam,        lam,        0, 0, 0],
                [lam,           lam + 2*mu, lam,        0, 0, 0],
                [lam,           lam,        lam + 2*mu, 0, 0, 0],
                [0,             0,          0,         mu, 0, 0],
                [0,             0,          0,         0, mu, 0],
                [0,             0,          0,         0, 0, mu]
                ], dtype=np.float64)
        
        # --- TRANSVERSELY ISOTROPIC CASE (FR4 Substrates) ---
        # Approximating in-plane (x,y) symmetry with distinct out-of-plane (z) compliance.
        nu_xy = nu
        nu_zb = nu * (e_z / e_xy) # Coupling ratio adjustments.

        g_xy = e_xy / (2.0 * (1.0 + nu_xy))
        g_zb = e_z / (2.0 * (1.0 + nu_zb)) # Out-of-plane shear modulus.

        compliance = np.zeros((6, 6), dtype=np.float64)
        compliance[0,0] = 1.0 / e_xy
        compliance[0,1] = -nu_xy / e_xy
        compliance[0,2] = -nu_zb / e_z
        compliance[1,0] = -nu_xy / e_xy
        compliance[1,1] = 1.0 / e_xy
        compliance[1,2] = -nu_zb / e_z
        compliance[2,0] = -nu_zb / e_z
        compliance[2,1] = -nu_zb / e_z
        compliance[2,2] = 1.0 / e_z
        compliance[3,3] = 1.0 / g_zb
        compliance[4,4] = 1.0 / g_zb
        compliance[5,5] = 1.0 / g_xy

        return np.linalg.inv(compliance)

    def get_cte_vector(self, material_name: str, use_worst_case: bool=True) -> np.ndarray:
        """Returns the 6-element Voigt CTE expansion vector [alpha]."""
        props = self.get_properties(material_name)
        suffix = 'Max' if use_worst_case else 'Min'

        alpha_xy = props[f'CTE_xy_{suffix}']
        alpha_z = props[f'CTE_z_{suffix}']

        # Displays [alpha_x, alpha_y, alpha_z, shear_xy, shear_yz, shear_zx]
        return np.array([alpha_xy, alpha_xy, alpha_z, 0.0, 0.0, 0.0], dtype=np.float64)

if __name__ == "__main__":
    # Test script locally
    db = MaterialDatabase()
    print("✅ MaterialLoader initialized successfully!")
    print(f"FR4 Stiffness Matrix Shape: {db.get_stiffness_matrix('FR4').shape}")
    print(f"Cu CTE Vector: {db.get_cte_vector('Cu')}")