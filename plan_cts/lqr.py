import numpy as np
import matplotlib.pyplot as plt
import utils

class LQR:
    def __init__(self, dt=0.05, resolution=0.03, lin_speed_range=np.array([0.0, 1.0]) / 0.03, ang_speed_range=np.array([-0.4, 0.4])):
        self.angle_dim = 2
        self.x_dim = 3
        self.resolution = resolution
        self.lin_speed_range = lin_speed_range
        self.ang_speed_range = ang_speed_range
        self.dt = dt
        self.C = np.diag([1.0, 1.0, 1.0, 1e-10, 1e-10])
        self.c = np.zeros(shape=(5,))
        self.dldxx = self.C[:self.x_dim, :self.x_dim]
        self.dlduu = self.C[self.x_dim:, self.x_dim:]
        self.dldux = self.C[self.x_dim:, :self.x_dim]
        self.dldx = self.c[:self.x_dim][:, np.newaxis]
        self.dldu = self.c[self.x_dim:][:, np.newaxis]
        # self.T = T
    
    def clip_lin_speed(self, speed):
        return np.clip(speed, self.lin_speed_range[0], self.lin_speed_range[1])
    
    def clip_ang_speed(self, speed):
        return np.clip(speed, self.ang_speed_range[0], self.ang_speed_range[1])

    def forward_one_step(self, x, u):
        dx = np.stack([
            self.clip_lin_speed(u[:, 0]) * np.cos(x[:, 2]),
            self.clip_lin_speed(u[:, 0]) * np.sin(x[:, 2]),
            self.clip_ang_speed(u[:, 1])
        ], axis=-1)
        
        return x + self.dt * dx

    def jac_x(self, xs, us):
        dfdtheta = np.stack([
            -self.clip_lin_speed(us[:, 0]) * np.sin(xs[:, 2]),
            self.clip_lin_speed(us[:, 0]) * np.cos(xs[:, 2]),
            np.zeros(shape=xs.shape[:-1])
        ], axis=-1)

        dfdx = np.stack([
            np.zeros_like(xs),
            np.zeros_like(xs),
            dfdtheta
        ], axis=-1)

        return np.eye(3) + self.dt * dfdx
    
    def jac_u(self, xs, us):
        vtilde_prime_nk = self.clip_lin_speed(us[:, 0])
        wtilde_prime_nk = self.clip_ang_speed(us[:, 1])
        zeros_nk = np.zeros(shape=xs.shape[:1])

        # Columns
        b1_nk3 = np.stack([vtilde_prime_nk * np.cos(xs[:, 2]),
                            vtilde_prime_nk * np.sin(xs[:, 2]),
                            zeros_nk], axis=-1)
        b2_nk3 = np.stack([zeros_nk,
                            zeros_nk,
                            wtilde_prime_nk], axis=-1)

        B_nk32 = np.stack([b1_nk3, b2_nk3], axis=-1)
        return B_nk32 * self.dt

    def backward_pass(self, xs, us):
        T = us.shape[0]
        xs_no_T = xs[:-1, :]
        dfdx = self.jac_x(xs_no_T, us)
        dfdu = self.jac_u(xs_no_T, us)

        ks = [None] * T
        Ks = [None] * T

        Vxx_t = self.dldxx
        Vx_t = self.dldx
        xs_next = self.forward_one_step(xs_no_T, us)

        for t in reversed(range(T)):
            f_t = xs[t + 1] - xs_next[t]
            f_t = np.concatenate([
                f_t[: self.angle_dim],
                utils.angle_normalize(f_t[self.angle_dim: self.angle_dim + 1]),
                f_t[self.angle_dim + 1:]
            ], axis=-1)
            f_t = f_t[:, np.newaxis]

            dfdx_t = dfdx[t]
            dfdu_t = dfdu[t]
            dfdx_t_T = dfdx_t.T
            dfdu_t_T = dfdu_t.T

            dfdx_T_dot_Vxx = dfdx_t_T @ Vxx_t
            dfdu_T_dot_Vxx = dfdu_t_T @ Vxx_t

            qx_t = self.dldx + dfdx_t_T @ Vx_t + dfdx_T_dot_Vxx @ f_t
            qu_t = self.dldu + dfdu_t_T @ Vx_t + dfdu_T_dot_Vxx @ f_t

            Qxx_t = self.dldxx + dfdx_T_dot_Vxx @ dfdx_t
            Qux_t = self.dldux + dfdu_T_dot_Vxx @ dfdx_t
            Quu_t = self.dlduu + dfdu_T_dot_Vxx @ dfdu_t

            inv_Quu_t = np.linalg.pinv(Quu_t)

            K_t_2d = -inv_Quu_t @ Qux_t
            k_t_2d = -inv_Quu_t @ qu_t
            Ks[t] = K_t_2d
            ks[t] = k_t_2d.squeeze()
            K_t_T = Ks[t].T

            Vxx_t = Qxx_t - K_t_T @ Quu_t @ K_t_2d
            Vx_t = qx_t - K_t_T @ Quu_t @ k_t_2d

        Ks = np.stack(Ks)
        ks = np.stack(ks)

        return Ks, ks
