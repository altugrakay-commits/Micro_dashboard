"""
Physics-Informed Neural Network (PINN) Model
============================================
Defines the MicroPINN Keras architecture and PINNLossCalculator for enforcing 3D 
Navier-Cauchy mechanical equilibrium loss terms: div(sigma) + f = 0."""

import keras
import tensorflow as tf
from physics.material_loader import MaterialDatabase
from physics.stress_engine import StressEngine
from typing import List, cast

@keras.saving.register_keras_serializable(package="physics")
class MicroPINN(tf.keras.Model):
    """Deep Neural Network (PINN) for mapping spatial coordinates (x,y,z) to displacement u(x,y,z)."""

    def __init__(self, layers: list[int] = [3, 64, 64, 64, 3], **kwargs) -> None:
        super().__init__(**kwargs)
        self.layer_sizes = layers

        # Characteristic physical dimensions for non-dimensionalization.
        self.L0 = tf.constant([1e-3, 1e-3, 0.2e-3], dtype=tf.float32)   # [m]
        self.U0 = tf.constant(1e-6, dtype=tf.float32)                   # 1 micron reference displacement [m]

        # Explicitly name layers to prevent variable name divergence on load.
        net_layers = []
        for i, out_dim in enumerate(layers[1:]):
            activation = "tanh" if i < len(layers) - 2 else None
            name = f"pinn_dense_{i}" if i < len(layers) - 2 else "pinn_output"
            net_layers.append(
                tf.keras.layers.Dense(
                    out_dim,
                    activation=activation,
                    kernel_initializer="glorot_normal",
                    name=name
                    )
            )
        self.net = tf.keras.Sequential(net_layers, name="pinn_backbone")

    def build(self, input_shape: tf.TensorShape) -> None:
        """Initializes model weights upon calling build."""
        super().build(input_shape)
        self.net.build((None, self.layer_sizes[0]))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass predicting physical displacement field in meters."""
        # Map physical coordinates [0, L] to [-1, 1] for stable neural activation.
        x_norm = 2.0 * (inputs / self.L0) - 1.0
        return self.net(x_norm) * self.U0

    def get_config(self) -> dict:
        """Serializes network architectural configuration."""
        config = super().get_config()
        config.update({"layers": self.layer_sizes})
        return config

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

class PINNLossCalculator:
    """Calculates second-order partial differential loss residuals via automatic differentiation."""

    def __init__(self, engine: StressEngine) -> None:
        self.engine = engine

    @tf.function
    def compute_pde_loss(
        self, 
        model: MicroPINN, 
        coords: tf.Tensor, 
        delta_T: tf.Tensor, 
        material_name: str
        ) -> tf.Tensor:
        """Evaluates Navier-Cauchy static equilibrium residual loss: div(sigma) = 0."""

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(coords)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(coords)
                # Forward pass through PINN to predict u = [u_x, u_y, u_z]
                u_pred = model(coords)  # Shape: (N,3)

                # 1. First-order displacement gradients w.r.t. coordinates [x,y,z]
                dux = tf.convert_to_tensor(tape1.gradient(u_pred[:, 0], coords))  # (N, 3) -> [dux_dx, dux_dy, dux_dz]
                duy = tf.convert_to_tensor(tape1.gradient(u_pred[:, 1], coords))  # (N, 3) -> [duy_dx, duy_dy, duy_dz]
                duz = tf.convert_to_tensor(tape1.gradient(u_pred[:, 2], coords))  # (N, 3) -> [duz_dx, duz_dy, duz_dz]

            # Spatial derivative vectors.
            du_dx = tf.stack([dux[:, 0], duy[:, 0], duz[:, 0]], axis=-1)
            du_dy = tf.stack([dux[:, 1], duy[:, 1], duz[:, 1]], axis=-1)
            du_dz = tf.stack([dux[:, 2], duy[:, 2], duz[:, 2]], axis=-1)

            # Computes Voigt Strain (N,6) and Stress Tensors (N,6)
            strain_voigt = self.engine.compute_strain(du_dx, du_dy, du_dz)
            stress_voigt = self.engine.compute_stress(strain_voigt, delta_T, material_name)

            # 2. Second-order stress divergence derivatives w.r.t. coordinates
            # Extract stress components: [s_xx, s_yy, s_zz, s_yz, s_zx, s_xy]
            s_xx, s_yy, s_zz = stress_voigt[:, 0], stress_voigt[:, 1], stress_voigt[:, 2]
            s_yz, s_zx, s_xy = stress_voigt[:, 3], stress_voigt[:, 4], stress_voigt[:, 5]

            ds_xx = tf.convert_to_tensor(tape2.gradient(s_xx, coords))  # (N, 3) -> [ds_xx_dx, ds_xx_dy, ds_xx_dz]
            ds_yy = tf.convert_to_tensor(tape2.gradient(s_yy, coords))  # (N, 3) -> [ds_yy_dx, ds_yy_dy, ds_yy_dz]
            ds_zz = tf.convert_to_tensor(tape2.gradient(s_zz, coords))  # (N, 3) -> [ds_zz_dx, ds_zz_dy, ds_zz_dz]
            ds_yz = tf.convert_to_tensor(tape2.gradient(s_yz, coords))
            ds_zx = tf.convert_to_tensor(tape2.gradient(s_zx, coords))
            ds_xy = tf.convert_to_tensor(tape2.gradient(s_xy, coords))

        # Explicitly release persistent tape resources to prevent memory leakage
        del tape1
        del tape2

        # Navier-Cauchy Equilibrium Residuals: div(sigma) = 0
        res_x = ds_xx[:, 0] + ds_xy[:, 1] + ds_zx[:, 2]
        res_y = ds_xy[:, 0] + ds_yy[:, 1] + ds_yz[:, 2]
        res_z = ds_zx[:, 0] + ds_yz[:, 1] + ds_zz[:, 2]

        # Mean-Squared Error of physical residuals
        residuals = tf.stack([res_x, res_y, res_z], axis=-1)
        return tf.reduce_mean(tf.square(residuals))

if __name__ == "__main__":
    db = MaterialDatabase()
    engine = StressEngine(material_db=db)
    pinn = cast(MicroPINN, MicroPINN())
    loss_calc = PINNLossCalculator(engine=engine)

    # Unit Test: Evaluates PDE loss over 10 sample spatial coordinates in a Silicon die.
    coords = tf.random.uniform((10, 3), minval=0.0, maxval=1e-3, dtype=tf.float32) # 1 mm die
    delta_T = tf.fill((10, 1), 75.0) # dT = 75 K operating rise
    pde_loss = loss_calc.compute_pde_loss(pinn, coords, delta_T, material_name="Si")

    print("✅ MicroPINN & LossCalculator initialized successfully.")
    print(f"Navier-Cauchy Loss Residual: {pde_loss.numpy():.4e}")