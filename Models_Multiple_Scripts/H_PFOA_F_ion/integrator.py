import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class RungeKuttaIntegratorCell(Layer):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7, betaj, k_cl, k_so3,
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

        # log10 init for global params
        self._log_k_init = np.log10([k1, k2, k3, k4, k5, k6, k7, betaj, k_cl, k_so3])

        p = self.constants
        tf32 = tf.float32
        self.epsilon_h2o_185  = tf.constant(float(p["epsilon_h2o_185"]), dtype=tf32)
        self.epsilon_oh_m_185   = tf.constant(float(p["epsilon_oh_m_185"]), dtype=tf32)
        self.epsilon_cl_185   = tf.constant(float(p["epsilon_cl_185"]), dtype=tf32)
        self.epsilon_so3_185  = tf.constant(float(p["epsilon_so3_185"]), dtype=tf32)
        self.epsilon_pfas_185 = tf.constant(float(p["epsilon_pfas_185"]), dtype=tf32)

        self.epsilon_h2o_254  = tf.constant(float(p["epsilon_h2o_254"]), dtype=tf32)
        self.epsilon_so3_254  = tf.constant(float(p["epsilon_so3_254"]), dtype=tf32)
        self.epsilon_pfas_254 = tf.constant(float(p["epsilon_pfas_254"]), dtype=tf32)

        self.phi_h2o_185  = tf.constant(float(p["phi_h2o_185"]), dtype=tf32)
        self.phi_oh_m_185   = tf.constant(float(p["phi_oh_m_185"]), dtype=tf32)
        self.phi_cl_185   = tf.constant(float(p["phi_cl_185"]), dtype=tf32)
        self.phi_so3_185  = tf.constant(float(p["phi_so3_185"]), dtype=tf32)
        self.phi_so3_254  = tf.constant(float(p["phi_so3_254"]), dtype=tf32)

        self.I0_185 = tf.constant(float(p["I0_185"]), dtype=tf32)
        self.I0_254 = tf.constant(float(p["I0_254"]), dtype=tf32)

        self.c_h2o = tf.constant(float(p["c_h2o"]), dtype=tf32)
        self.l = tf.constant(float(p["l"]), dtype=tf32)
        

    def build(self, input_shape):
        # ---- global trainable kinetics (log10-space) ----
        names = ['k1','k2','k3','k4','k5','k6','k7','betaj','k_cl','k_so3']
        self.log_k_values = {
            n: self.add_weight(name=f'log_{n}', shape=(),
                               initializer=tf.constant_initializer(v),
                               trainable=True)
            for n, v in zip(names, self._log_k_init)
        }

        # ==========================================================
        # Baseline (SO3-independent) generation gain: alpha_other > 0
        # This is the key to fitting the "no SO3" sequence.
        # alpha_other = exp(theta_other)
        # ==========================================================
        self.theta_other = self.add_weight(name="theta_other", shape=(),initializer=tf.constant_initializer(0.0),trainable=True)

        # ==========================================================
        # SO3 modifier: alpha_so3(x) = exp(theta1*x + theta2*x^2), alpha_so3(0)=1
        # Use x = log1p(c_mM / Kh_mM) to avoid early saturation.
        # Kh_mM is log10-parametrized to stay positive.
        # ==========================================================
        self.log_Kh_mM = self.add_weight(name="log_Kh_mM", shape=(),initializer=tf.constant_initializer(np.log10(5.0)), trainable=True)
        self.theta1 = self.add_weight(name="theta1", shape=(),initializer=tf.constant_initializer(0.0),trainable=True)
        self.theta2 = self.add_weight(name="theta2", shape=(),initializer=tf.constant_initializer(0.0),trainable=True)

        # Optional extra sink gamma(x) (kept small at x=0)
        self.log_gamma_scale = self.add_weight(name="log_gamma_scale", shape=(),initializer=tf.constant_initializer(4.0),trainable=True)
        self.phi0 = self.add_weight(name="phi0", shape=(),initializer=tf.constant_initializer(-6.0),trainable=True)
        self.phi1 = self.add_weight(name="phi1", shape=(),initializer=tf.constant_initializer(0.0),trainable=True)
        self.phi2 = self.add_weight(name="phi2", shape=(),initializer=tf.constant_initializer(0.0),trainable=True)

        # Quadratic termination for eaq: k_rec >= 0
        self.log_krec = self.add_weight(name="log_krec", shape=(),initializer=tf.constant_initializer(8.0),trainable=True)

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
        params = {name: tf.pow(self.ten, logv) for name, logv in self.log_k_values.items()}
        y = states[0]
        u = tf.cast(inputs, tf.float32)

        c_cl   = self._get_channel_or_default(u, "c_cl",  self.c_cl_default)
        c_so3  = self._get_channel_or_default(u, "c_so3", self.c_so3_default)  # M
        pH     = self._get_channel_or_default(u, "pH",    self.pH_default)
        c_pfoa0 = self._get_channel_or_default(u, "c_pfoa0", float(self.initial_state[0, 0]))

        k1 = self._fun(y, params, c_cl, c_so3, pH, c_pfoa0) * self.dt
        k2 = self._fun(y + 0.5*k1, params, c_cl, c_so3, pH, c_pfoa0) * self.dt
        k3 = self._fun(y + 0.5*k2, params, c_cl, c_so3, pH, c_pfoa0) * self.dt
        k4 = self._fun(y + k3,     params, c_cl, c_so3, pH, c_pfoa0) * self.dt
        y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        # Keep concentrations physically meaningful and avoid numerical sign flips.
        y_next = tf.maximum(y_next, 0.0)

        if self.for_prediction:
            F = y_next[:, 7:8]
            if self.output_mode == "fluoride":
                output = y_next
            else:
                eps = tf.constant(1e-12, tf.float32)
                output = 100.0 * F / (self.nF * c_pfoa0 + eps)
            
        else:
            F = y_next[:, 7:8]
            if self.output_mode == "fluoride":
                output = F
            else:
                eps = tf.constant(1e-12, tf.float32)
                output = 100.0 * F / (self.nF * c_pfoa0 + eps)

        return output, [y_next]

    def _fun(self, y, params, c_cl, c_so3, pH, c_pfoa0):
        eps = tf.constant(1e-12, tf.float32)

        # states: 7 PFAS + F-
        y_vars = [tf.maximum(y[:, i:i+1], 0.0) for i in range(7)]

        # ---- generation split ----
        G_other, G_so3 = self.generation_of_eaq_parts_tf(c_cl, c_so3, pH, y_vars)

        # ---- baseline gain (critical for SO3=0 case) ----
        alpha_other = tf.exp(self.theta_other)  # >0

        # ---- SO3 modifier ----
        c_so3_mM = 1000.0 * c_so3
        Kh_mM = tf.pow(self.ten, self.log_Kh_mM)
        x = tf.math.log1p(c_so3_mM / (Kh_mM + eps))  # x=0 at so3=0

        alpha_so3 = tf.exp(self.theta1 * x + self.theta2 * tf.square(x))  # alpha_so3(0)=1

        # total generation
        G = alpha_other * G_other + alpha_so3 * G_so3

        # ---- linear consumption coefficient ----
        beta_j   = params['betaj']
        k_cl_eaq = params['k_cl']
        k_so3_eaq = params['k_so3']

        # PFAS sink coefficient: sum_i k_i * C_i(t)
        D_pfas = tf.add_n([params[f'k{i+1}'] * y_vars[i] for i in range(7)])

        # optional extra sink gamma(x)
        gamma_scale = tf.pow(self.ten, self.log_gamma_scale)
        gamma_raw = tf.nn.softplus(self.phi0 + self.phi1 * x + self.phi2 * tf.square(x))
        gamma = gamma_scale * gamma_raw

        D_lin = beta_j + k_so3_eaq * c_so3 + k_cl_eaq * c_cl + D_pfas + gamma + eps

        # ---- quadratic termination ----
        k_rec = tf.pow(self.ten, self.log_krec)
        G_pos = tf.maximum(G, 0.0)
        disc = tf.sqrt(tf.square(D_lin) + 4.0 * k_rec * G_pos + 1e-24)
        # Stable positive root of: k_rec*c_eaq^2 + D_lin*c_eaq - G_pos = 0
        c_eaq = (2.0 * G_pos) / (D_lin + disc + 1e-24)
        c_eaq = tf.maximum(c_eaq, 0.0)

        # ---- PFAS-eaq reaction rates ----
        rates = [params[f'k{i+1}'] * c_eaq * y_vars[i] for i in range(7)]

        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]

        # F- generation (keep your factor 2)
        dy8 = 2.0 * tf.add_n(rates) + rates[6]

        return tf.concat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], axis=-1)

    # -------------------------------------------------------------------------
    # Split generation into "other" and "SO3-driven"
    # -------------------------------------------------------------------------
    def generation_of_eaq_parts_tf(self, c_cl, c_so3, pH, y_vars):
        c_pfas_total = tf.add_n(y_vars)
        
        eps = tf.constant(1e-12, tf.float32)
        c_oh_m = tf.pow(self.ten, (-14.0 + pH))

        Sigma_f_185 = (
            self.epsilon_h2o_185 * self.c_h2o +
            self.epsilon_oh_m_185 * c_oh_m +
            self.epsilon_cl_185   * c_cl +
            self.epsilon_so3_185  * c_so3 +
            self.epsilon_pfas_185 * c_pfas_total
        ) + eps

        Sigma_f_254 = (
            self.epsilon_h2o_254 * self.c_h2o +
            self.epsilon_so3_254 * c_so3 +
            self.epsilon_pfas_254 * c_pfas_total
        ) + eps

        f_h2o_185  = (self.epsilon_h2o_185 * self.c_h2o) / Sigma_f_185
        f_oh_m_185 = (self.epsilon_oh_m_185 * c_oh_m)     / Sigma_f_185
        f_cl_185   = (self.epsilon_cl_185   * c_cl)       / Sigma_f_185
        f_so3_185  = (self.epsilon_so3_185  * c_so3)      / Sigma_f_185
        f_so3_254  = (self.epsilon_so3_254  * c_so3)      / Sigma_f_254

        def absorb(eps_abs, c):
            return (1.0 - tf.pow(self.ten, -(eps_abs * self.l * c)))

        term_h2o_185  = f_h2o_185  * self.phi_h2o_185  * absorb(self.epsilon_h2o_185,  self.c_h2o)
        term_oh_m_185 = f_oh_m_185 * self.phi_oh_m_185 * absorb(self.epsilon_oh_m_185, c_oh_m)
        term_cl_185   = f_cl_185   * self.phi_cl_185   * absorb(self.epsilon_cl_185,  c_cl)
        term_so3_185  = f_so3_185  * self.phi_so3_185  * absorb(self.epsilon_so3_185, c_so3)

        # 185nm: split
        G_other_185 = self.I0_185 * (term_h2o_185 + term_oh_m_185 + term_cl_185)
        G_so3_185   = self.I0_185 * term_so3_185

        # 254nm: sulfite only in your current model
        G_so3_254 = self.I0_254 * f_so3_254 * self.phi_so3_254 * absorb(self.epsilon_so3_254, c_so3)

        G_other = G_other_185
        G_so3 = G_so3_185 + G_so3_254
        return G_other, G_so3

    def debug_extract(self, inputs):
        u = tf.cast(inputs, tf.float32)
        c_cl   = self._get_channel_or_default(u, "c_cl",  self.c_cl_default)
        c_so3  = self._get_channel_or_default(u, "c_so3", self.c_so3_default)
        pH     = self._get_channel_or_default(u, "pH",    self.pH_default)
        c_pfoa0 = self._get_channel_or_default(u, "c_pfoa0", float(self.initial_state[0, 0]))
        return c_cl, c_so3, pH, c_pfoa0