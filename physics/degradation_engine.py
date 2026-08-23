"""
Degradation Engine
==================
Evaluates viscoplastic damage, Darveaux crack propagation, Norris-Landzberg thermal 
fatigue, intermetallic layer kinetics, and spatial multiaxial Morrow fatigue mapping.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Union, Optional
from dataclasses import dataclass

@dataclass
class MaterialProperties:
    """Material definition including Anand viscoplasticity and physical parameters."""
    name: str = "SAC305"
    elastic_modulus_GPa: float = 40.0
    poisson_ratio: float = 0.36
    cte_ppm_K: float = 20.0

    # Anand Model Parameters (SAC305).
    A: float = 1.0e7    # Pre-exponential factor (l/s)
    Q_R: float = 9000.0 # Activation Energy / Universal gas constant (K)
    xi: float = 4.0     # Stress Multiplier
    m: float = 0.25     # Strain rate sensitivity exponent
    s_0: float = 30.0   # Initial deformation resistance (MPa)

    # Intermetallic kinetics (Cu6Sn5 / Cu35n).
    D_eff_0: float = 2.5e-5     # Pre-exponential diffusion factor (mm^2/s)
    E_a_imc: float = 68000.0    # Activation energy (J/mol)

class IntermetallicKinetics:
    """Calculates temperature-driven intermetallic compound (IMC) growth."""

    def __init__(self, d0: float = 2.5e-11, q_activation: float = 68000.0) -> None:
        self.d0 = d0            # Pre-exponential diffusion coefficient (m^2/s)
        self.q = q_activation   # Initial IMC layer thickness (meters)
        self.r_gas = 8.314      # Universal gas constant (J/mol·K)

    def calculate_growth(self, temp_k: float, time_s: float, barrier_type: str = "Ni_AU") -> float:
        """Calculates current IMC thickness layer in meters"""
        # Attenuation factor for barrier layers.
        barrier_factors = {
            "none": 1.0,
            "Ni_Au": 0.05,  # Reduces diffusion rate by 95%
            "TaN_W": 0.01   # Reduces diffusion rate by 99%
        }
        k_barrier = barrier_factors.get(barrier_type, 1.0)
        d_eff = self.d0 * k_barrier * np.exp(-self.q / (self.r_gas * temp_k))
        return float(np.sqrt(d_eff * time_s))

class DegradationEngine:
    """Damage accumulation engine for semiconductor packaging interconnects."""

    def __init__(self, stress_engine = None, material: Optional[MaterialProperties] = None) -> None:
        self.stress_engine = stress_engine
        self.material = material or MaterialProperties()
        self.imc_kinetics = IntermetallicKinetics()

    def calculate_anand_equivalent_stress(self, plastic_strain_rate: float, temp_k: float) -> float:
        """Calculates flow stress using Anand Constitutive Model."""
        term1 = plastic_strain_rate / self.material.A
        term2 = np.exp(self.material.Q_R / temp_k)
        inner = (term1 * term2) ** self.material.m
        inner_clipped = np.clip(inner, -1.0, 1.0)
        return float((1.0 / self.material.xi) * np.arcsin(inner_clipped) * self.material.s_0)

    def evaluate_darveaux_fatigue(
            self, 
            shear_strain_range: float, 
            shear_stress_mpa: float, 
            solder_joint_diameter_m: float = 150e-6
            ) -> Dict[str, float]:
        """Calculates crack initiation and growth cycles using Darveaux's energy formulation."""

        # Inelastic strain energy density per cycle (Delta W) in MPa.
        # Delta W - 0.5 * delta_gamma * delta_tau
        delta_W_mpa = 0.5 * shear_strain_range * shear_stress_mpa
        delta_W_safe = max(delta_W_mpa, 1e-4)

        # Standard SAC305 Darveaux Constants (Constants calibrated for Delta W in MPa)
        K1, K2 = 22400.0, -1.52
        K3, K4 = 1.10e-7, 1.36

        # 1. Cycles to Crack Initiation (N_0).
        n_0 = K1 * (delta_W_safe ** K2)

        # 2. Crack Growth Rate per Cycle (da/dN) in meters/cycle.
        da_dn = K3 * (delta_W_safe ** K4)

        # 3. Total Fatigue Life to Failure (N_f) across joint characteristic diameter.
        a_critical = solder_joint_diameter_m / 2.0  # Characteristic crack path length.
        n_g = a_critical / max(da_dn, 1e-12)
        n_f = n_0 + n_g
        
        return {
            "delta_W_mpa": delta_W_mpa,
            "crack_initiation_cycles": float(n_0),
            "crack_propagation_cycles": float(da_dn),
            "total_fatigue_life_cycles": float(n_f)
        }

    def evaluate_norris_landzberg(
            self, 
            shear_strain_range: float, 
            temp_max_k: float, 
            cycling_frequency_hz: float = 1.0/3600.0, 
            n_target: float = 1000.0
            ) -> Dict[str, Union[float, bool]]:
        """Evaluates empirical Norris-Landzberg thermal cycling model."""
        if shear_strain_range <= 0:
            return{"norris_landzberg_cycles": 1e7, "target_cycles": n_target, "fatigue_penalty": 0.0, "target_met": True}

        n_exp = 2.1
        a_freq_exp = 0.332
        ea_over_kb = 3130.0
        
        frequency_factor = cycling_frequency_hz ** a_freq_exp
        thermal_factor = np.exp(ea_over_kb / temp_max_k)
        strain_factor = shear_strain_range ** (-n_exp)

        cycles = float(1.2e-3 * strain_factor * frequency_factor * thermal_factor)
        fatigue_penalty = float(np.square(np.maximum(0.0, 1.0 - (cycles / n_target))))

        return {
            "norris_landzberg_cycles": cycles,
            "target_cycles": n_target,
            "fatigue_penalty": fatigue_penalty,
            "target_met": cycles >= n_target
        }

    def evaluate_interface_degradation(
            self, 
            temp_k: float, 
            operating_hours: float, 
            shear_strain_range: float,
            relief_gap_um: float = 50.0,
            barrier_type: str = "Ni_Au",
            n_target: float = 1000.0,
            cycling_frequency_hz: float = 1/3600.0
    ) -> Dict[str, Union[float, Dict]]:
        """Runs multi-physics degradation pipeline combining Anand stress, Darveaux fatigue, and Norris-Landzberg."""
        time_s = operating_hours * 3600.0
        imc_growth_m = self.imc_kinetics.calculate_growth(temp_k=temp_k, time_s=time_s, barrier_type=barrier_type)

        # Apply strain reduction directly via relief gap compliance.
        damping_factor = 0.85 + 0.15 * np.exp(-relief_gap_um / 400)
        effective_strain = shear_strain_range * damping_factor

        # Compute plastic strain rate dynamically based on cycling frequency.
        ramp_time_s = max(1.0 / (2.0 * cycling_frequency_hz), 60.0)
        effective_strain_rate = max(effective_strain / ramp_time_s, 1e-16)

        # Energy density delta_W calculation via Anand model.
        flow_stress_mpa = self.calculate_anand_equivalent_stress(
            plastic_strain_rate=effective_strain_rate, 
            temp_k=temp_k
        )

        # Darveaux fatigue life using damped inelastic strain energy density.
        darveaux = self.evaluate_darveaux_fatigue(
            shear_strain_range=effective_strain,
            shear_stress_mpa=flow_stress_mpa
        )

        # Norris-Landzberg thermal cycling using damped effective strain and explicit kwargs.
        norris_landzberg = self.evaluate_norris_landzberg(
            shear_strain_range=effective_strain, 
            temp_max_k=temp_k, 
            cycling_frequency_hz=cycling_frequency_hz,
            n_target=n_target
        )

        return {
            "imc_thickness_um": imc_growth_m * 1e6,
            "darveaux_fatigue": darveaux,
            "norris_landzberg_cycles": norris_landzberg
        }

    def evaluate_mesh_fatigue_map(
            self,
            stress_engine_outputs: Dict[str, Union[tf.Tensor, np.ndarray]],     # Von Mises & Shear Tensors from StressEngine.
            temp_k_field: float,                                                # Spatial temperature field from PINN
            operating_hours: float = 8760.0                                     # Default to 1 operational year
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Maps multiaxial Morrow stress-life across a 3D spatial mesh."""
        sig_vm = stress_engine_outputs['sig_vm']
        sig_vm_arr = (sig_vm.numpy() if isinstance(sig_vm, tf.Tensor) else np.asarray(sig_vm))

        # Morrow Stress-Life Parameters for interconnect materials (Pa).
        sigma_f_prime = 250e6   # Fatigue strength coefficient.
        b_exp = -0.12           # Fatigue strength exponent.

        # Zero-to-max cycling stress amplitude and mean stress components.
        sigma_amplitude = np.maximum(sig_vm_arr / 2.0, 1.0e3)
        sigma_mean = sig_vm_arr / 2.0

        # Morrow multiaxial fatigue limit calculation: N_f = 0.5 * (sigma_a / (sigma_f_prime - sigma_m)) ^ (1/b)
        effective_fatigue_limit = np.maximum(sigma_f_prime - sigma_mean, 1e5)
        with np.errstate(over='ignore', invalid='ignore'):
            base_ratio = sigma_amplitude / effective_fatigue_limit
            cycles_map = 0.5 * (base_ratio ** (1 / b_exp))
            cycles_map = np.nan_to_num(cycles_map, nan=1.0e7, posinf=1.0e7)
        cycles_map = np.clip(cycles_map, 10.0, 1.0e7)

        return {
            "min_cycles_to_failure": float(np.min(cycles_map)),
            "mean_cycles_to_failure": float(np.mean(cycles_map)),
            "fatigue_cycles_map": cycles_map
        }
if __name__ == "__main__":
    engine = DegradationEngine()
    res = engine.evaluate_interface_degradation(
        temp_k=398.15,
        operating_hours=1000.0,
        shear_strain_range=0.005
    )
    print("✅ DegradationEngine Phase executed successfully.")
    print(f"IMC Layer Thickness: {res['imc_thickness_um']:.3f} µm")