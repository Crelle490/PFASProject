from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from predictor.ode_runtime import build_model_from_config


class HPINNMovingHorizonEstimator:
    def __init__(
        self,
        dt: float,
        horizon_steps: int,
        *,
        measurement_indices=(7,),
        max_iters: int = 200,
        learning_rate: float = 1e-2,
    ):
        self.dt = float(dt)
        self.horizon_steps = int(horizon_steps)
        self.measurement_indices = tuple(int(i) for i in measurement_indices)
        self.max_iters = int(max_iters)
        self.learning_rate = float(learning_rate)
        

        project_root = Path(__file__).resolve().parents[1]
        cfg_dir = project_root / "config"
        trained_k_yaml = cfg_dir / "trained_params.yaml"

        t_sim = np.arange(self.horizon_steps, dtype=np.float32) * self.dt
        self.model, self.dummy, initial_states = build_model_from_config(
            cfg_dir=cfg_dir,
            trained_k_yaml=trained_k_yaml,
            t_sim=t_sim,
            dt=self.dt,
        )
        self.rk_cell = None
        for layer in self.model.layers:
            if hasattr(layer, "cell"):
                self.rk_cell = layer.cell
                break
        if self.rk_cell is None:
            raise RuntimeError("Could not locate RK cell in HPINN model.")
        # Keep constant inputs as tensors so graph execution can place them on GPU.
        self._cell_inputs = tf.constant([self.rk_cell.c_cl, self.rk_cell.c_so3], dtype=tf.float32)

        self._x0_guess = initial_states.astype(np.float32)
        self.scale_factor = tf.constant(1.0e-06, dtype=tf.float32)
        self._window = []
        self.last_state = self._x0_guess[0].copy()

        cov_path = cfg_dir / "covariance_params.yaml"
        if cov_path.exists():
            cov = yaml.safe_load(open(cov_path, "r"))
            R = np.array(cov["measurement_noise_covariance"], dtype=np.float32)
            if R.ndim == 2 and R.shape[0] >= max(self.measurement_indices) + 1:
                R_sel = R[np.ix_(self.measurement_indices, self.measurement_indices)]
                self._inv_R = np.linalg.pinv(R_sel)
            else:
                self._inv_R = None
        else:
            self._inv_R = None

    @tf.function
    def _rollout_tf(self, x0, n_steps):
        """Compiled rollout to reduce Python overhead and allow GPU placement."""
        state = tf.cast(x0, tf.float32)
        if tf.rank(state) == 1:
            state = tf.reshape(state, (1, -1))

        n = tf.cast(n_steps, tf.int32)
        outputs = tf.TensorArray(tf.float32, size=n)
        for i in tf.range(n):
            y, [state] = self.rk_cell(self._cell_inputs, [state])
            outputs = outputs.write(i, y[0])
        return outputs.stack()

    def _rollout(self, x0, n_steps):
        x0_tf = tf.convert_to_tensor(x0, dtype=tf.float32)
        return self._rollout_tf(x0_tf, tf.convert_to_tensor(n_steps, dtype=tf.int32))

    def simulate(self, x0, steps):
        y_pred = self._rollout(x0, steps)
        return y_pred.numpy()

    def advance_prior(self, steps: int):
        """Propagate the internal prior forward by a number of steps (between measurements)."""
        steps = int(max(0, steps))
        if steps == 0:
            return self.last_state
        traj = self._rollout_tf(self.last_state.reshape(1, -1), steps)
        self.last_state = traj[steps - 1].numpy()
        self._x0_guess = np.asarray(self.last_state, dtype=np.float32).reshape(1, -1)
        return self.last_state

    def update_measurement(self, z):
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        if len(z) != len(self.measurement_indices):
            raise ValueError(
                f"measurement length {len(z)} does not match indices {self.measurement_indices}"
            )
        self._window.append(z)
        if len(self._window) > self.horizon_steps:
            self._window = self._window[-self.horizon_steps :]

    def estimate(self):
        if not self._window:
            return self.last_state, None

        z = np.asarray(self._window, dtype=np.float32)
        n_steps = z.shape[0]
        z_tf = tf.convert_to_tensor(z, dtype=tf.float32)

        x0 = tf.Variable(self._x0_guess, dtype=tf.float32)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        for _ in range(self.max_iters):
            with tf.GradientTape() as tape:
                y_pred = self._rollout_tf(x0, n_steps)
                y_sel = tf.gather(y_pred, self.measurement_indices, axis=-1)
                
                err = (y_sel - z_tf)/self.scale_factor
                if self._inv_R is not None:
                    inv_R = tf.convert_to_tensor(self._inv_R, dtype=tf.float32)
                    err = tf.einsum("ti,ij->tj", err, inv_R)
                loss = tf.reduce_mean(tf.square(err))
            grads = tape.gradient(loss, [x0])
            if grads[0] is None:
                break
            opt.apply_gradients(zip(grads, [x0]))

        y_pred = self._rollout_tf(x0, n_steps)
        self.last_state = y_pred[n_steps - 1].numpy()
        self._x0_guess = np.asarray(self.last_state, dtype=np.float32).reshape(1, -1)

        return self.last_state, y_pred.numpy()

    def step(self, z):
        self.update_measurement(z)
        return self.estimate()


if __name__ == "__main__":
    # Example usage
    mhe = HPINNMovingHorizonEstimator(dt=1.0, horizon_steps=20, measurement_indices=(7,))
    dummy_meas = 0.0
    state, traj = mhe.step([dummy_meas])
    print("Estimated state:", state)
