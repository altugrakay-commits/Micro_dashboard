"""
PINN Hybrid Optimization Engine
===============================
Trains Physics-Informed Neural Networks (PINNs) using a two-stage
hybrid scheme: Global Adam optimization followed by L-BFGS Quasi-Newton fine-tuning.
"""

import logging
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Project root directory added dynamically to sys.path.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import scipy.optimize as sopt
import tensorflow as tf
from typing import cast

from physics.material_loader import MaterialDatabase
from physics.stress_engine import StressEngine
from physics.pinn_model import MicroPINN, PINNLossCalculator
from physics.post_processor import FieldExtractor

class DomainSampler:
    """Generates 3D spatial collocation points for domain interior and boundary surfaces."""
    
    def __init__(self, bounds: tuple[float, float, float] = (1e-3, 1e-3, 0.2e-3)):
        self.bounds = tf.constant(bounds, dtype=tf.float32)
        self.lx, self.ly, self.lz = bounds

    def sample_domain(self, n_points: int) -> tf.Tensor:
        """Samples interior volume in meters [0, lx] x [0, ly] x [0, lz]."""
        return tf.random.uniform((n_points, 3), dtype=tf.float32) * self.bounds

    def sample_bottom_boundary(self, n_points: int) -> tf.Tensor:
        """Samples clamped bottom boundary face (z = 0)."""
        xy = tf.random.uniform((n_points, 2), dtype=tf.float32) * self.bounds[:2]
        return tf.concat([xy, tf.zeros((n_points, 1), dtype=tf.float32)], axis=1)

    def sample_top_boundary(self, n_points: int) -> tf.Tensor:
        """Samples traction-free top surface face (z = lz)."""
        xy = tf.random.uniform((n_points, 2), dtype=tf.float32) * self.bounds[:2]
        z = tf.fill((n_points, 1), self.lz)
        return tf.concat([xy, z], axis=1)

class PINNTrainer:
    """Orchestrates hybrid Adam + L-BFGS multi-stage PINN training."""

    def __init__(
            self, 
            model: MicroPINN, 
            loss_calc: PINNLossCalculator, 
            sampler: DomainSampler, 
            learning_rate: float = 1e-3
        ):
        self.model = model
        self.loss_calc = loss_calc
        self.sampler = sampler

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm = 1.0)
        # Scaling factors for unit non-dimensionalization (MPa scaling).
        self.stress_scale = 1e-9    # Scale stress to GPa
        self.pde_scale = 1e-6       # Scale second-derivatives

    def compute_total_loss(
            self, 
            coords_domain: tf.Tensor, 
            coords_bottom: tf.Tensor, 
            coords_top: tf.Tensor, 
            dT_domain: tf.Tensor,
            dT_top: tf.Tensor,
            material_name: str, 
            weights: tuple[float,float,float] = (1.0, 1e2, 1.0)
            ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Computes weighted multi-objective loss (Interior PDE + Boundary Conditions)."""
        w_pde, w_dirichlet, w_neumann = weights

        # 1. Interior PDE Residual Loss (Scaled)
        pde_loss = self.loss_calc.compute_pde_loss(self.model, coords_domain, dT_domain, material_name) * self.pde_scale

        # 2. Dirichlet Boundary Loss: Clamped Bottom Face u(x,y,0) = 0
        u_bottom = self.model(coords_bottom)
        dirichlet_loss = tf.reduce_mean(tf.square(u_bottom[:, 2:3]))

        # 3. Neumann Boundary Loss: Traction-free Top Surface (sigma_zz = tau_zx = tau_yz = 0)
        with tf.GradientTape() as tape_top:
            tape_top.watch(coords_top)
            u_top = self.model(coords_top)

        J_top = tape_top.jacobian(u_top, coords_top)
        strain_top = self.loss_calc.engine.compute_strain(J_top[:,:,0], J_top[:,:,1], J_top[:,:,2])
        stress_top_gpa = self.loss_calc.engine.compute_stress(strain_top, dT_top, material_name) * self.stress_scale

        # Extraction of sig_zz, sig_yz, sig_zx (indices 2, 3, 4):
        neumann_loss = tf.reduce_mean(tf.square(stress_top_gpa[:, 2:5]))
        # Total Weighted Physics Loss
        total_loss = (w_pde * pde_loss) + (w_dirichlet * dirichlet_loss) + (w_neumann * neumann_loss)

        return total_loss, pde_loss, dirichlet_loss, neumann_loss

    @tf.function
    def train_adam_step(
            self,
            coords_domain: tf.Tensor,
            coords_bottom: tf.Tensor,
            coords_top: tf.Tensor,
            dT_domain: tf.Tensor,
            dT_top: tf.Tensor,
            material_name: str,
    ) -> dict[str, tf.Tensor]:
        """Performs a single graph-compiled Adam iteration."""
        with tf.GradientTape() as tape:
            total_loss, pde, dirichlet, neumann = self.compute_total_loss(
                coords_domain, coords_bottom, coords_top, dT_domain, dT_top, material_name
            )

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return {
            "total_loss": total_loss,
            "pde_loss": pde,
            "dirichlet_loss": dirichlet,
            "neumann_loss": neumann, 
        }

    def train_lbfgs(
            self,
            coords_domain: tf.Tensor,
            coords_bottom: tf.Tensor,
            coords_top: tf.Tensor,
            delta_T_val: float,
            material_name: str,
            max_iter: int = 200,
    ) -> None:
        """Fine-tune model weights using second-order L-BFGS optimization."""
        print("\n⚡Switching to Stage 2: L-BFGS Quasi-Newton Fine Tuning...")

        dT_domain = tf.fill((coords_domain.shape[0], 1), delta_T_val)
        dT_top = tf.fill((coords_top.shape[0], 1), delta_T_val)

        # Helper functions to pack/unpack model weights into a 1D vector for SciPy
        shapes = [v.shape for v in self.model.trainable_variables]
        sizes = [v.shape.num_elements() for v in self.model.trainable_variables]

        def set_flat_weights(flat_weights: np.ndarray) -> None:
            idx = 0
            for v, shape, size in zip(self.model.trainable_variables, shapes, sizes):
                v.assign(tf.reshape(flat_weights[idx: idx + size], shape))
                idx += size

        def loss_and_grads(flat_weights: np.ndarray) -> tuple[float, np.ndarray]:
            set_flat_weights(flat_weights)
            with tf.GradientTape() as tape:
                total_loss, _,_,_ = self.compute_total_loss(
                    coords_domain, coords_bottom, coords_top, dT_domain, dT_top, material_name
                    )
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            flat_grads = np.concatenate([g.numpy().flatten() for g in grads]).astype(np.float64) # type: ignore
            return float(total_loss.numpy()), flat_grads

        init_weights = np.concatenate([v.numpy().flatten() for v in self.model.trainable_variables]).astype(np.float64)
        iter_count = 0

        def callback(x: np.ndarray) -> None:
            nonlocal iter_count
            iter_count += 1
            if iter_count % 20 == 0:
                loss_val, _ = loss_and_grads(x)
                print(f" L-BFGS Iter {iter_count:03d} | Total Loss: {loss_val:.4e}")

        # Execute L-BFGS optimization.
        res = sopt.minimize(
            fun=loss_and_grads,
            x0=init_weights,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={"maxiter": max_iter, "gtol": 1e-8, "ftol": 1e-8}
        )

        set_flat_weights(res.x)
        print(f"✅ L-BFGS Complete | Status: {res.message}")

if __name__ == "__main__":
    db = MaterialDatabase()
    engine = StressEngine(material_db=db)
    pinn = cast(MicroPINN, MicroPINN())
    loss_calc = PINNLossCalculator(engine=engine)
    sampler = DomainSampler(bounds = (1e-3, 1e-3, 0.2e-3))

    trainer = PINNTrainer(model=pinn, loss_calc=loss_calc, sampler=sampler)

    # Pre-sample discrete collocation grid.
    coords_domain = sampler.sample_domain(n_points=5000)
    coords_bottom = sampler.sample_bottom_boundary(n_points=100)
    coords_top = sampler.sample_top_boundary(n_points=100)

    delta_T_val, material_name = 75.0, "Si"
    dT_domain = tf.fill((coords_domain.shape[0], 1), delta_T_val)
    dT_top = tf.fill((coords_top.shape[0], 1), delta_T_val)

    # Stage 1: Adam.
    print("🚀 Stage 1: Adam Global Optimization (Max 300 Epochs)...")
    best_loss, patience, patience_counter = float("inf"), 30, 0

    for epoch in range(1, 301):
        metrics = {
            k: float(v.numpy())
            for k, v in trainer.train_adam_step(
                coords_domain, coords_bottom, coords_top, dT_domain, dT_top, material_name
            ).items()
        }

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"Total Loss: {metrics['total_loss']:.4e} | "
                f"PDE Residual: {metrics['pde_loss']:.4e} | "
                f"Dirichlet (Bottom): {metrics['dirichlet_loss']:.4e} | "
                f"Neumann (Top): {metrics['neumann_loss']:.4e} | "
            )

        # Early Stopping Trigger Logic
        current_loss = metrics["pde_loss"]
        if current_loss < best_loss - 1e-3:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch >= 100 and (current_loss <= 5.0 or patience_counter >= patience):
            print(f"\n 🎯 Early Stopping Triggered at Epoch {epoch}! (PDE Residual: {current_loss:.4f})")
            break

    # Stage 2: L-BFGS
    trainer.train_lbfgs(
        coords_domain, 
        coords_bottom, 
        coords_top, 
        delta_T_val=delta_T_val, 
        material_name=material_name, 
        max_iter=1000
    )

    # Model Export.
    pinn.build((None, 3))
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    pinn.save(models_dir / "models/pinn_silicon_v1.keras")
    pinn.save_weights(models_dir / "models/pinn_silicon_v1.weights.h5")

    print(f"✅ Model saved to: {models_dir / 'pinn_silicon_v1.keras'}")

    # Field Analysis Check.
    print("\n📊 Stage 3: Extracting 3D Stress and Displacement Fields...")
    extractor = FieldExtractor(model=pinn)
    field_data = extractor.evaluate_grid_3d(
        bounds = ((0.0, 1.0e-3), (0.0, 1.0e-3), (0.0, 0.2e-3)),
        resolution=(20,20,10),
        material_id=material_name
    )

    print(f"Peak predicted von Mises stress: {np.max(field_data['stresses_GPa']['sig_vm']):.4f} GPa")