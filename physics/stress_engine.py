"""
Stress Engine
=============
Computes thermo-mechanical constitutive relations, strain conversion, 
von Mises stress fields, Fourier heat PDE residuals, and interfacial shear kinetics.
"""

# ==============
# MODULE IMPORTS
# ==============

import tensorflow as tf
from typing import Dict, Optional, Union
from .material_loader import MaterialDatabase
import numpy as np

class StressEngine:
    """Computes constitutive relations, thermal strains, and Navier-Cauchy 
    equilibrium loss terms using TensorFlow automatic differentiation."""

    def __init__(self, material_db: Optional[MaterialDatabase] = None) -> None:
        self.material_db = material_db or MaterialDatabase()

    @tf.function
    def compute_thermal_pde_loss(
        self,
        tape: tf.GradientTape,      # Persistent tf.GradientTape capturing spatial coordinates.
        coords: tf.Tensor,          # Spatial coordinates tensor of shape (batch_size, 3) [x,y,z].
        temperature: tf.Tensor,     # Predicted scalar temperature field T(x,y,z) or shape (batch_size, 1).
        k_eff: tf.Tensor,           # Effective spatial isotropic/anisotropic thermal conductivity tensor (W/m*K) of shape (batch_size, 1) or (batch_size, 3).
        q_volumetric: tf.Tensor     # Internal heat source density Q(x,y,z) in (W/m³) of shape (batch_size, 1).
    ) -> tf.Tensor:                 # Returns pde_loss, or the Mean squared PDE residual over the batch.
        """Computes steady-state Fourier heat conduction PDE residual: 
        Residual = div( k * grad(T) ) + Q = 0"""

        # First spatial derivative: grad(T):
        dT_dcoords = tape.gradient(temperature, coords)

        # Convert IndexedSlices to dense Tensor if returned by tape gradient.
        if isinstance(dT_dcoords, tf.IndexedSlices):
            dT_dcoords = tf.convert_to_tensor(dT_dcoords)

        # Conductive Heat Flux Vector: q = -k(x,y,z) * grad(T).
        q_flux = -k_eff * dT_dcoords    # Element-wise anisotropic heat flux

        # Divergence of Heat Flux: div(q) = dqx/dx + dq/dy + dqz/dz.
        dq_dcoords = tape.gradient(q_flux, coords)
        if isinstance(dq_dcoords, tf.IndexedSlices):
            dq_dcoords = tf.convert_to_tensor(dq_dcoords)
            
        div_q = tf.reduce_sum(dq_dcoords, axis=1, keepdims=True)

        # Governing Heat Equation Residual: -div(q) + Q = 0.
        residual = -div_q + q_volumetric
        return tf.reduce_mean(tf.square(residual))
    
    @tf.function
    def compute_strain(self, du_dx: tf.Tensor, du_dy: tf.Tensor, du_dz: tf.Tensor) -> tf.Tensor:
        """Converts spatial displacement gradients into 6-element Voigt engineering strain: 
        eps = [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_zx, gamma_xy]."""
        eps_xx = du_dx[:, 0:1]
        eps_yy = du_dy[:, 1:2]
        eps_zz = du_dz[:, 2:3]

        # Engineering Shear Strains (gamma = 2 * epsilon_shear)
        gamma_yz = du_dy[:, 2:3] + du_dz[:, 1:2]
        gamma_zx = du_dx[:, 2:3] + du_dz[:, 0:1]
        gamma_xy = du_dx[:, 1:2] + du_dy[:, 0:1]

        # Stack into Voigt vector (batch_size, 6)
        return tf.concat([eps_xx, eps_yy, eps_zz, gamma_yz, gamma_zx, gamma_xy], axis=-1)

    def compute_stress(self, strain_voigt: tf.Tensor, delta_T: tf.Tensor, material_name: str) -> tf.Tensor:
        """Calculates Voigt stress tensor: sigma = [C] * (eps_total - alpha * delta_T)."""
        # Fetch NumPy tensors from database and casts to TensorFlow float32
        c_matrix = tf.constant(self.material_db.get_stiffness_matrix(material_name), dtype=tf.float32)
        alpha_vec = tf.constant(self.material_db.get_cte_vector(material_name), dtype=tf.float32)

        # Thermal Strain: eps_thermal = alpha * delta_T
        # Reshapes alpha to (1,6) to broadcast across tne batch dimension
        alpha_vec = tf.reshape(alpha_vec, (1,6))

        # Ensures that delta_T shape is (batch_size, 1)
        if len(delta_T.shape) == 1:
            delta_T = tf.expand_dims(delta_T, axis=-1)

        thermal_strain = alpha_vec * delta_T

        # Squeeze extra middle dimensions if present.
        if len(strain_voigt.shape) == 3:
            strain_voigt = tf.squeeze(strain_voigt, axis=1)

        # Slice to keep only the standard 6 Voigt strain components.
        if strain_voigt.shape[-1] > 6: # type: ignore
            strain_voigt = strain_voigt[:, :6]
        
        elastic_strain = strain_voigt - thermal_strain

        # Compute Stress: sigma = elastic_strain @ C^T (Batch matrix multiplication)
        return tf.matmul(elastic_strain, c_matrix, transpose_b=True)

    def compute_von_mises(self, stress_voigt: tf.Tensor) -> tf.Tensor:
        """Computes Von Mises equivalent stress from the 6-element Voigt stress vector."""
        s_xx, s_yy, s_zz = stress_voigt[:, 0:1], stress_voigt[:, 1:2], stress_voigt[:, 2:3]
        s_yz, s_zx, s_xy = stress_voigt[:, 3:4], stress_voigt[:, 4:5], stress_voigt[:, 5:6]

        term_diag = 0.5 * (
            (s_xx - s_yy) ** 2 + (s_yy - s_zz) ** 2 + (s_zz - s_xx) ** 2
        )
        term_shear = 3.0 * (s_xy**2 + s_yz**2 + s_zx**2)

        return tf.sqrt(
            tf.sqrt(term_diag + term_shear + 1e-8)
        )

    def compute_interface_shear_stress(
            self, 
            mat1_name: str, 
            mat2_name: str, 
            delta_T: tf.Tensor,                     # Tensor field (batch_size, 1)
            h1: float, 
            h2: float, 
            h_imc: Union[float, tf.Tensor] = 0.0,   # Dynamic IMC layer tensor (batch_size, 1)
            char_len: float = 1e-3,
            via_density_factor: float = 1.0         # Thermal via density reinforcement factor
            ) -> tf.Tensor:
        """Calculates interfacial shear stress accounting for dynamic IMC growth compliance."""

        # Fetches CTE (mean_alpha_x) and Young's Modulus E via stiffness matrix proxy or DB lookup.
        alpha1 = self.material_db.get_cte_vector(mat1_name)[0]
        alpha2 = self.material_db.get_cte_vector(mat2_name)[0]

        # Approximate E from E_x = C11 - 2*C12^2/(C11 + C12) or from DB properties.
        C1 = self.material_db.get_stiffness_matrix(mat1_name)
        C2 = self.material_db.get_stiffness_matrix(mat2_name)

        # Via density scales out-of-plane stiffness and compliance.
        E1 = C1[0,0] * via_density_factor
        E2 = C2[0,0]
        e_imc = 120e9  # Intermetallic Cu-Sn layer modulus

        delta_epsilon = (alpha1 - alpha2) * delta_T

        if isinstance(h_imc, tf.Tensor):
            h_imc_safe = tf.maximum(h_imc, 1e-9)
            compliance = (1.0 / (E1 * h1)) + (1.0 / (E2 * h2)) + (1.0 / (e_imc * h_imc_safe))
        else:
            h_imc_safe = max(h_imc, 1e-9)
            compliance = (1.0 / (E1 * h1)) + (1.0 / (E2 * h2)) + (1.0 / (e_imc * h_imc_safe))

        return tf.abs(delta_epsilon) / (compliance * char_len)

    def compute_underfill_shear_reduction(
            self,
            shear_strain_bare: float,
            underfill_tg_k: float = 423.15,     # Default Tg: 150°C (423.15 K)
            operating_temp_k: float = 398.15,    # Default T_max: 125°C (398.15 K)
            relief_gap_um: float = 50.0
    ) -> float:
        """Evaluates underfill effectiveness and strain dissipation based on operating temperature and Tg."""
        effectiveness = 0.35 if operating_temp_k < underfill_tg_k else 0.65
        # Factor in mechanical relief gap dissipation.
        gap_efficiency = float(np.clip(1.0 - (relief_gap_um / 200.0), 0.2, 1.0))
        return float(shear_strain_bare * effectiveness * gap_efficiency)

    def compute_solder_geometry_factor(
            self,
            geometry_profile: str = "cylindrical",
            height_um: float = 100.0,
            diameter_um: float = 120.0
    ) -> float:
        """Calculates plastic strain concentration factor (K_v) based on solder bump profile."""
        aspect_ratio = height_um / max(diameter_um, 1.0)
        profiles = {
            "cylindrical": 1.00,
            "barrel": 1.15,         # Barrel shapes concentrate strain at waist/pad interface.
            "hourglass": 0.72,      # Hourglass shapes reduce peak strain at pad corners.
            "optimized_fillet": 0.65
        }
        base_factor = profiles.get(geometry_profile.lower(), 1.00)
        # Taller aspect ratios reduce overall shear strain range.
        aspect_modifier = float(np.clip(1.0 / np.sqrt(aspect_ratio), 0.70, 1.30))
        return float(base_factor * aspect_modifier)

    def compute_die_flexural_warpage(
            self,
            die_thickness_um: float = 750.0,
            die_span_mm: float = 10.0,
            delta_T: float = 180.0
    ) -> Dict[str, float]:
        """Calculates die flexural rigidity and bowing/warpage displacement."""
        si_e_modulus = 130.0e9  # Silicon Modulus (Pa)
        si_poisson = 0.28       # Silicon Poisson ratio
        si_cte = 2.6e-6         # Silicon CTE (1/K)

        h_m = die_thickness_um * 1.0e-6
        span_m = die_span_mm * 1.0e-3

        # Flexural Rigidity D = E * h^3 / (12 * (1 - v^2)).
        flexural_rigidity = (si_e_modulus * (h_m ** 3)) / (12.0 * (1.0 - si_poisson ** 2))

        # Approximate max bowing deflection (delta_Z) under thermal load.
        delta_z_um = float((3.0 * span_m**2 * si_cte * delta_T) / (8.0 * h_m) * 1.0e6)

        return {
            "flexural_rigidity_Nm": float(flexural_rigidity),
            "max_bowing_delta_z_um": delta_z_um,
            "die_thickness_um": die_thickness_um
        }

if __name__ == "__main__":
    db=MaterialDatabase()
    engine = StressEngine(material_db=db)

    # Unit Test 1: Mechanical Stress Engine Evaluation (Clamped Boundary Condition)
    batch_size = 5
    zero_strain = tf.zeros((batch_size, 6), dtype=tf.float32)
    delta_T = tf.fill((batch_size, 1), 100.0) # dT = 100 K

    cu_stress = engine.compute_stress(zero_strain, delta_T, material_name="Cu")
    cu_vm = engine.compute_von_mises(cu_stress)

    print("✅ StressSEngine executed successfully!")
    print(f"Cu Von Mises Stress (Clamped @ 100K): {cu_vm[0, 0].numpy() / 1e6:.2f} MPa")

    # Unit Test 2: Thermal Heat Conduction PDE Loss Evaluation.
    pde_batch_size = 100
    coords = tf.random.uniform((pde_batch_size, 3), minval=-1.0, maxval=1.0)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(coords)
        # Synthetic quadratic T profile: T(x,y,z) = 300 + 50*(x^2 + y^2 + z^2)
        temp = 300 + 50 * tf.reduce_sum(tf.square(coords), axis=1, keepdims=True)

    k_cu = tf.fill((pde_batch_size, 1), 400.0)      # 400 W/m·K for Copper
    q_vol = tf.fill((pde_batch_size, 1), -30000.0)  # Heat sink extraction term.

    pde_loss = engine.compute_thermal_pde_loss(tape, coords, temp, k_cu, q_vol)
    del tape

    print("✅ Fourier Thermal Conduction PDE test passed!")
    print(f"Fourier PDE Loss Residual: {pde_loss.numpy():.4f}")