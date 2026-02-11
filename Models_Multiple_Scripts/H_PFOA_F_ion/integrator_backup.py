import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class RungeKuttaIntegratorCell(Layer):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7,  betaj, k_cl, k_so3,
                 constants,
                 c_cl_default, c_so3_default, pH_default,
                 dt,
                 dummy_initial_state,
                 for_prediction=False,
                 nF=15,
                 exo_map=None,
                 output_mode="defluorination_pct",
                 **kwargs):
        super().__init__(**kwargs)
        self.dt = float(dt)
        self.initial_state = np.asarray(dummy_initial_state, dtype=np.float32)
        self.state_size = 8
        self.for_prediction = bool(for_prediction)

        self.c_cl_default = float(c_cl_default)
        self.c_so3_default = float(c_so3_default)
        self.pH_default = float(pH_default)

        self.nF = float(nF)
        self.output_mode = str(output_mode)

        if exo_map is None:
            exo_map = {"c_cl": 0, "c_so3": 1, "pH": 2, "c_pfoa0": 3}
        self.exo_map = dict(exo_map)

        self.constants = {k: tf.constant(float(v), dtype=tf.float32) for k, v in constants.items()}
        self.ten = tf.constant(10.0, dtype=tf.float32)

        self._log_k_init = np.log10([k1, k2, k3, k4, k5, k6, k7, betaj, k_cl, k_so3])

    def build(self, input_shape):
        # --- global kinetic params (shared) ---
        k_names = ['k1','k2','k3','k4','k5','k6','k7', 'betaj', 'k_cl', 'k_so3']
        self.log_k_values = {
            name: self.add_weight(
                name=f'log_{name}', shape=(),
                initializer=tf.constant_initializer(val),
                trainable=True
            )
            for name, val in zip(k_names, self._log_k_init)
        }

        # ==========================================================
        # NEW: SO3-conditioned correction on hydrated-electron steady state
        # alpha(c) = 10^(a0 + a1*s),  s = c_mM/(K + c_mM)
        # gamma(c) = 10^g0 + 10^g1 * s
        # Here c is SO3 in mM (we convert from M by *1000).
        # ==========================================================

        self.log_Kh_mM = self.add_weight("log_Kh_mM", shape=(),
                                         initializer=tf.constant_initializer(5.0),  # Kh=1 mM
                                         trainable=True)

        # alpha(h)=exp(theta0+theta1*h+theta2*h^2)
        self.theta0 = self.add_weight("theta0", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.theta1 = self.add_weight("theta1", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.theta2 = self.add_weight("theta2", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)

        # gamma(h)=softplus(phi0+phi1*h+phi2*h^2)  (init small to avoid blowing denom)
        self.phi0 = self.add_weight("phi0", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.phi1 = self.add_weight("phi1", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.phi2 = self.add_weight("phi2", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)

        self.built = True

    def _get_channel_or_default(self, u, key, default_scalar):
        idx = self.exo_map.get(key, None)
        if idx is None:
            return tf.fill([tf.shape(u)[0], 1], tf.constant(default_scalar, tf.float32))
        if u.shape.rank is not None and u.shape.rank >= 2 and u.shape[-1] is not None:
            if idx >= int(u.shape[-1]):
                return tf.fill([tf.shape(u)[0], 1], tf.constant(default_scalar, tf.float32))
        return u[:, idx:idx+1]

    def call(self, inputs, states):
        params = {name: tf.pow(self.ten, log_v) for name, log_v in self.log_k_values.items()}
        y = states[0]
        u = tf.cast(inputs, tf.float32)

        c_cl   = self._get_channel_or_default(u, "c_cl",  self.c_cl_default)
        c_so3  = self._get_channel_or_default(u, "c_so3", self.c_so3_default)  # in M
        pH     = self._get_channel_or_default(u, "pH",    self.pH_default)
        c_pfoa0 = self._get_channel_or_default(u, "c_pfoa0", float(self.initial_state[0,0]))

        k1 = self._fun(y, params, c_cl, c_so3, pH, c_pfoa0) * self.dt
        k2 = self._fun(y + 0.5*k1, params, c_cl, c_so3, pH, c_pfoa0) * self.dt
        k3 = self._fun(y + 0.5*k2, params, c_cl, c_so3, pH, c_pfoa0) * self.dt
        k4 = self._fun(y + k3,     params, c_cl, c_so3, pH, c_pfoa0) * self.dt
        y_next = y + (k1 + 2*k2 + 2*k3 + k4)/6.0

        if self.for_prediction:
            output = y_next
        else:
            F = y_next[:, 7:8]
            if self.output_mode == "fluoride":
                output = F
            else:
                eps = tf.constant(1e-12, tf.float32)
                output = 100.0 * F / (self.nF * c_pfoa0 + eps)

        return output, [y_next]

    def _fun(self, y, params, c_cl, c_so3, pH, c_pfoa0):
        # ---------------- states (placeholder: 7 PFAS + F-) ----------------
        y_vars = [y[:, i:i+1] for i in range(7)]  # PFAS chain
        # F = y[:,7:8]

        # ---------------- base e_aq generation (your existing physics) ----------------
        numerator = self.generation_of_eaq_tf(c_cl, c_so3, pH, c_pfoa0)

        # ---------------- base denominator (your existing scavenging structure) -------
        # k_so3_eaq = tf.constant(1.5e6, dtype=tf.float32)
        # k_cl_eaq  = tf.constant(1.0e6, dtype=tf.float32)
        # beta_j    = tf.constant(1.57e4, dtype=tf.float32)

        # k_so3_eaq = tf.pow(self.ten, self.k_so3_)
        # beta_j = tf.pow(self.ten, self.betaj_)
        k_cl_eaq = params['k_cl']
        k_so3_eaq = params['k_so3']
        beta_j = params['betaj']

        denom_base = params['k1'] * c_pfoa0 + beta_j + k_so3_eaq * c_so3 + k_cl_eaq * c_cl

        # ==========================================================
        # NEW: SO3-only correction to e_aq steady state
        # Convert SO3 from M -> mM for nicer scaling in 0~10 mM
        # s = c_mM/(K + c_mM)
        # alpha = 10^(a0 + a1*s) (dimensionless)
        # gamma = 10^g0 + 10^g1*s (same units as denom)
        # ==========================================================
        # --- build h(c_so3) using mM scale ---
        c_so3_mM = 1000.0 * c_so3
        Kh_mM = tf.pow(self.ten, self.log_Kh_mM)  # >0
        h = c_so3_mM / (Kh_mM + c_so3_mM + 1e-12)  # [0,1)

        alpha = tf.exp(self.theta0 + self.theta1 * h + self.theta2 * tf.square(h))  # >0
        gamma = tf.nn.softplus(self.phi0 + self.phi1 * h + self.phi2 * tf.square(h))  # >=0


        c_eaq = (alpha * numerator) / (denom_base + gamma + 1e-12)

        # ---------------- PFAS-eaq kinetics (global k_i shared) ----------------
        rates = [params[f'k{i+1}'] * c_eaq * y_vars[i] for i in range(7)]

        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]
        dy8 = 2.0 * tf.reduce_sum(rates, axis=0, keepdims=False)  # or sum(nu_i * rates[i])

        return tf.concat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], axis=-1)

    def generation_of_eaq_tf(self, c_cl, c_so3, pH, c_pfoa0):

        p = self.constants
        ten = self.ten
        c_oh_m = tf.pow(ten, (-14.0 + pH))
        Sigma_f_185 = (
            p["epsilon_h2o_185"] * p["c_h2o"] +
            p["epsilon_oh_m_185"] * c_oh_m +
            p["epsilon_cl_185"]   * c_cl +
            p["epsilon_so3_185"]  * c_so3 +
            p["epsilon_pfas_185"] * c_pfoa0
        )
        Sigma_f_254 = (
            p["epsilon_h2o_254"] * p["c_h2o"] +
            p["epsilon_so3_254"] * c_so3 +
            p["epsilon_pfas_254"] * c_pfoa0
        )
        f_h2o_185  = (p["epsilon_h2o_185"] * p["c_h2o"]) / Sigma_f_185
        f_oh_m_185 = (p["epsilon_oh_m_185"] * c_oh_m)     / Sigma_f_185
        f_cl_185   = (p["epsilon_cl_185"]   * c_cl)       / Sigma_f_185
        f_so3_185  = (p["epsilon_so3_185"]  * c_so3)      / Sigma_f_185
        f_so3_254  = (p["epsilon_so3_254"]  * c_so3)      / Sigma_f_254

        def absorb(eps, c):
            return (1.0 - tf.pow(ten, -(eps * p["l"] * c)))

        term_h2o_185  = f_h2o_185  * p["phi_h2o_185"]  * absorb(p["epsilon_h2o_185"],  p["c_h2o"])
        term_oh_m_185 = f_oh_m_185 * p["phi_oh_m_185"] * absorb(p["epsilon_oh_m_185"], c_oh_m)
        term_cl_185   = f_cl_185   * p["phi_cl_185"]   * absorb(p["epsilon_cl_185"],   c_cl)
        term_so3_185  = f_so3_185  * p["phi_so3_185"]  * absorb(p["epsilon_so3_185"],  c_so3)

        numerator_185 = p["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)
        numerator_254 = p["I0_254"] * f_so3_254 * p["phi_so3_254"] * absorb(p["epsilon_so3_254"], c_so3)
        return numerator_185 + numerator_254
