#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, './Utilities/')

import numpy as np
import matplotlib.pyplot as plt
from pyDOE2 import lhs
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1234)
tf.set_random_seed(1234)


def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def generate_reference_solution(Nx=200, Nt=150, L=1.0, gamma=1.4, K=1.0, cfl=0.4):
    dx = L / Nx
    x = np.linspace(0, L, Nx)

    rho = np.zeros(Nx)
    m = np.zeros(Nx)
    mid = Nx // 2
    rho[:mid] = 1.0
    rho[mid:] = 0.125
    m[:] = 0.0

    rho_hist = np.zeros((Nt + 1, Nx))
    m_hist = np.zeros((Nt + 1, Nx))
    t_hist = np.zeros(Nt + 1)
    rho_hist[0] = rho
    m_hist[0] = m

    t = 0.0
    for n in range(Nt):
        u = m / rho
        p = K * (rho ** gamma)
        c = np.sqrt(K * gamma * (rho ** (gamma - 1)))
        max_wave_speed = np.max(np.abs(u) + c)
        dt = cfl * dx / max_wave_speed
        t += dt

        d_rho_left = np.zeros(Nx); d_rho_right = np.zeros(Nx)
        d_m_left = np.zeros(Nx);   d_m_right = np.zeros(Nx)
        d_rho_left[1:] = rho[1:] - rho[:-1]
        d_rho_right[:-1] = rho[1:] - rho[:-1]
        d_m_left[1:] = m[1:] - m[:-1]
        d_m_right[:-1] = m[1:] - m[:-1]

        sigma_rho = minmod(d_rho_left, d_rho_right)
        sigma_m = minmod(d_m_left, d_m_right)

        rho_L = rho + 0.5 * sigma_rho
        rho_R = np.roll(rho - 0.5 * sigma_rho, -1)
        m_L = m + 0.5 * sigma_m
        m_R = np.roll(m - 0.5 * sigma_m, -1)

        u_L = m_L / rho_L; p_L = K * (rho_L ** gamma); c_L = np.sqrt(K * gamma * (rho_L ** (gamma - 1)))
        u_R = m_R / rho_R; p_R = K * (rho_R ** gamma); c_R = np.sqrt(K * gamma * (rho_R ** (gamma - 1)))

        f1_L = m_L; f2_L = (m_L ** 2 / rho_L) + p_L
        f1_R = m_R; f2_R = (m_R ** 2 / rho_R) + p_R

        s_max = np.maximum(np.abs(u_L) + c_L, np.abs(u_R) + c_R)
        flux1 = 0.5 * (f1_L + f1_R) - 0.5 * s_max * (rho_R - rho_L)
        flux2 = 0.5 * (f2_L + f2_R) - 0.5 * s_max * (m_R - m_L)

        rho_next = np.copy(rho); m_next = np.copy(m)
        rho_next[1:-1] = rho[1:-1] - (dt / dx) * (flux1[1:-1] - flux1[:-2])
        m_next[1:-1]   = m[1:-1]   - (dt / dx) * (flux2[1:-1] - flux2[:-2])
        rho_next[0], rho_next[-1] = rho_next[1], rho_next[-2]
        m_next[0], m_next[-1] = m_next[1], m_next[-2]

        rho, m = rho_next, m_next
        rho_hist[n + 1] = rho
        m_hist[n + 1] = m
        t_hist[n + 1] = t

    return x, t_hist, rho_hist, m_hist


class ConservativeEulerPINN:
    def __init__(self, X_u_list, U_u_list, X_f_list, X_fi_list,
                 layers_list, bounds_list, gamma, K):
        self.gamma = gamma
        self.K = K
        self.n_sub = len(X_u_list)
        self.n_int = len(X_fi_list)
        assert self.n_int == self.n_sub - 1

        self.bounds_list = bounds_list

        self.x_u = [Xu[:, 0:1] for Xu in X_u_list]
        self.t_u = [Xu[:, 1:2] for Xu in X_u_list]
        self.u_data = U_u_list

        self.x_f = [Xf[:, 0:1] for Xf in X_f_list]
        self.t_f = [Xf[:, 1:2] for Xf in X_f_list]

        self.x_fi = [Xfi[:, 0:1] for Xfi in X_fi_list]
        self.t_fi = [Xfi[:, 1:2] for Xfi in X_fi_list]

        self.weights, self.biases, self.a_var = [], [], []
        for layers in layers_list:
            W, b, a = self.initialize_NN(layers)
            self.weights.append(W)
            self.biases.append(b)
            self.a_var.append(a)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.x_u_tf, self.t_u_tf, self.u_tf = [], [], []
        for Xu, Ud in zip(self.x_u, U_u_list):
            self.x_u_tf.append(tf.placeholder(tf.float64, shape=[None, 1]))
            self.t_u_tf.append(tf.placeholder(tf.float64, shape=[None, 1]))
            self.u_tf.append(tf.placeholder(tf.float64, shape=[None, 2]))

        self.x_f_tf, self.t_f_tf = [], []
        for Xf in self.x_f:
            self.x_f_tf.append(tf.placeholder(tf.float64, shape=[None, 1]))
            self.t_f_tf.append(tf.placeholder(tf.float64, shape=[None, 1]))

        self.x_fi_tf, self.t_fi_tf = [], []
        for Xfi in self.x_fi:
            self.x_fi_tf.append(tf.placeholder(tf.float64, shape=[None, 1]))
            self.t_fi_tf.append(tf.placeholder(tf.float64, shape=[None, 1]))

        self.u_pred = [self.net_U(self.x_u_tf[i], self.t_u_tf[i], i) for i in range(self.n_sub)]
        self.f_pred = [self.net_f(self.x_f_tf[i], self.t_f_tf[i], i) for i in range(self.n_sub)]

        self.fi_pred = []
        self.avg_pred = []
        self.left_at_i = []
        self.right_at_i = []
        for k in range(self.n_int):
            xk, tk = self.x_fi_tf[k], self.t_fi_tf[k]
            U_left = self.net_U(xk, tk, k)
            U_right = self.net_U(xk, tk, k + 1)
            self.left_at_i.append(U_left)
            self.right_at_i.append(U_right)

            rho_L, m_L = U_left[:, 0:1], U_left[:, 1:2]
            rho_R, m_R = U_right[:, 0:1], U_right[:, 1:2]
            p_L = self.K * rho_L ** self.gamma
            p_R = self.K * rho_R ** self.gamma
            flux1_L, flux2_L = m_L, (m_L ** 2) / rho_L + p_L
            flux1_R, flux2_R = m_R, (m_R ** 2) / rho_R + p_R

            fi_rho = flux1_L - flux1_R
            fi_m = flux2_L - flux2_R
            self.fi_pred.append(tf.concat([fi_rho, fi_m], axis=1))
            self.avg_pred.append(0.5 * (U_left + U_right))

        self.loss = []
        for i in range(self.n_sub):
            data_loss = 20.0 * tf.reduce_mean(tf.square(self.u_tf[i] - self.u_pred[i]))
            pde_loss = tf.reduce_mean(tf.square(self.f_pred[i]))

            iface_loss = 0.0
            if i - 1 >= 0:
                iface_loss += 20.0 * tf.reduce_mean(tf.square(self.fi_pred[i - 1]))
                iface_loss += 20.0 * tf.reduce_mean(
                    tf.square(self.right_at_i[i - 1] - self.avg_pred[i - 1]))
            if i < self.n_int:
                iface_loss += 20.0 * tf.reduce_mean(tf.square(self.fi_pred[i]))
                iface_loss += 20.0 * tf.reduce_mean(
                    tf.square(self.left_at_i[i] - self.avg_pred[i]))

            self.loss.append(data_loss + pde_loss + iface_loss)

        self.optimizer_Adam = tf.train.AdamOptimizer(0.0008)
        self.train_op = [self.optimizer_Adam.minimize(L) for L in self.loss]

        self.sess.run(tf.global_variables_initializer())

    def initialize_NN(self, layers):
        weights, biases = [], []
        for l in range(len(layers) - 1):
            in_dim, out_dim = layers[l], layers[l + 1]
            xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
            W = tf.Variable(tf.cast(tf.random.truncated_normal([in_dim, out_dim],
                             stddev=xavier_stddev), tf.float64), dtype=tf.float64)
            b = tf.Variable(tf.zeros([1, out_dim], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        a = tf.Variable(0.05, dtype=tf.float64)
        return weights, biases, a

    def neural_net(self, X, lb, ub, weights, biases, a):
        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        for l in range(len(weights) - 1):
            H = tf.tanh(20.0 * a * tf.add(tf.matmul(H, weights[l]), biases[l]))
        Y = tf.add(tf.matmul(H, weights[-1]), biases[-1])
        return Y

    def net_U(self, x, t, sub_idx):
        lb, ub = self.bounds_list[sub_idx]
        raw = self.neural_net(tf.concat([x, t], axis=1), lb, ub,
                               self.weights[sub_idx], self.biases[sub_idx], self.a_var[sub_idx])
        rho_raw, m_raw = raw[:, 0:1], raw[:, 1:2]
        rho = tf.nn.softplus(rho_raw) + 1e-6
        return tf.concat([rho, m_raw], axis=1)

    def net_f(self, x, t, sub_idx):
        U = self.net_U(x, t, sub_idx)
        rho, m = U[:, 0:1], U[:, 1:2]
        p = self.K * rho ** self.gamma

        rho_t = tf.gradients(rho, t)[0]
        m_t = tf.gradients(m, t)[0]
        m_x = tf.gradients(m, x)[0]

        flux2 = (m ** 2) / rho + p
        flux2_x = tf.gradients(flux2, x)[0]

        f_continuity = rho_t + m_x
        f_momentum = m_t + flux2_x
        return tf.concat([f_continuity, f_momentum], axis=1)

    def train(self, n_iter, print_every=200):
        tf_dict = {}
        for i in range(self.n_sub):
            tf_dict[self.x_u_tf[i]] = self.x_u[i]
            tf_dict[self.t_u_tf[i]] = self.t_u[i]
            tf_dict[self.u_tf[i]] = self.u_data[i]
            tf_dict[self.x_f_tf[i]] = self.x_f[i]
            tf_dict[self.t_f_tf[i]] = self.t_f[i]
        for k in range(self.n_int):
            tf_dict[self.x_fi_tf[k]] = self.x_fi[k]
            tf_dict[self.t_fi_tf[k]] = self.t_fi[k]

        loss_history = [[] for _ in range(self.n_sub)]
        start = time.time()
        for it in range(n_iter):
            for i in range(self.n_sub):
                self.sess.run(self.train_op[i], tf_dict)
            if it % print_every == 0:
                losses = [self.sess.run(L, tf_dict) for L in self.loss]
                for i, Lval in enumerate(losses):
                    loss_history[i].append(Lval)
                print("it %5d | " % it + " | ".join(
                    "loss_%d=%.3e" % (i + 1, Lval) for i, Lval in enumerate(losses)) +
                    " | %.1fs" % (time.time() - start))
        return loss_history

    def predict(self, x, t, sub_idx):
        U = self.sess.run(self.net_U(tf.constant(x, dtype=tf.float64),
                                      tf.constant(t, dtype=tf.float64), sub_idx))
        return U[:, 0:1], U[:, 1:2]


def build_training_data(x, t_hist, rho_hist, m_hist, x_interface,
                         N_u_per_subdomain=200, N_f=2000, N_f_interface=100,
                         layer_width=20, layer_depth=4):
    Nx = len(x)
    T_final = t_hist[-1]
    idx_interface = np.array([np.argmin(np.abs(x - xi)) for xi in x_interface])
    n_sub = len(x_interface) - 1
    n_int = n_sub - 1

    X_u_list, U_u_list, X_f_list, bounds_list, layers_list = [], [], [], [], []

    for sd in range(n_sub):
        i0, i1 = idx_interface[sd], idx_interface[sd + 1]
        x_lo, x_hi = x[i0], (x[i1] if i1 < Nx else x[-1])
        bounds_list.append((np.array([x_lo, 0.0]), np.array([x_hi, T_final])))
        layers_list.append([2] + [layer_width] * layer_depth + [2])

        x_ic = x[i0:i1]
        t_ic = np.zeros_like(x_ic)
        rho_ic = rho_hist[0, i0:i1]
        m_ic = m_hist[0, i0:i1]

        edge_pts, edge_vals = [], []
        if sd == 0:
            edge_pts.append(np.stack([np.full_like(t_hist, x[0]), t_hist], axis=1))
            edge_vals.append(np.stack([rho_hist[:, 0], m_hist[:, 0]], axis=1))
        if sd == n_sub - 1:
            edge_pts.append(np.stack([np.full_like(t_hist, x[-1]), t_hist], axis=1))
            edge_vals.append(np.stack([rho_hist[:, -1], m_hist[:, -1]], axis=1))

        X_ic = np.stack([x_ic, t_ic], axis=1)
        U_ic = np.stack([rho_ic, m_ic], axis=1)
        if edge_pts:
            X_ic = np.vstack([X_ic] + edge_pts)
            U_ic = np.vstack([U_ic] + edge_vals)

        n_avail = X_ic.shape[0]
        n_pick = min(N_u_per_subdomain, n_avail)
        pick = np.random.choice(n_avail, n_pick, replace=False)
        X_u_list.append(X_ic[pick])
        U_u_list.append(U_ic[pick])

        lhs_pts = lhs(2, N_f)
        x_f = x_lo + (x_hi - x_lo) * lhs_pts[:, 0]
        t_f = T_final * lhs_pts[:, 1]
        X_f_list.append(np.stack([x_f, t_f], axis=1))

    X_fi_list = []
    for k in range(n_int):
        x_int = x[idx_interface[k + 1]]
        t_f = T_final * lhs(1, N_f_interface).flatten()
        X_fi_list.append(np.stack([np.full_like(t_f, x_int), t_f], axis=1))

    return X_u_list, U_u_list, X_f_list, X_fi_list, bounds_list, layers_list


if __name__ == "__main__":

    gamma = 1.4
    K = 1.0
    L = 1.0

    x, t_hist, rho_hist, m_hist = generate_reference_solution(
        Nx=200, Nt=150, L=L, gamma=gamma, K=K, cfl=0.4)

    x_interface = np.array([0.0, 0.375, 0.5, 0.625, 1.0])

    (X_u_list, U_u_list, X_f_list, X_fi_list,
     bounds_list, layers_list) = build_training_data(
        x, t_hist, rho_hist, m_hist, x_interface,
        N_u_per_subdomain=200, N_f=2000, N_f_interface=100,
        layer_width=20, layer_depth=4)

    model = ConservativeEulerPINN(X_u_list, U_u_list, X_f_list, X_fi_list,
                                   layers_list, bounds_list, gamma, K)

    Max_iter = 5000
    loss_history = model.train(Max_iter, print_every=200)

    n_sub = len(bounds_list)
    snapshot_t = t_hist[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(x, rho_hist[-1], 'k--', linewidth=2, label='FV reference (density)')
    colors = ['r', 'b', 'g', 'm', 'c', 'y']
    for sd in range(n_sub):
        lb, ub = bounds_list[sd]
        x_plot = np.linspace(lb[0], ub[0], 200).reshape(-1, 1)
        t_plot = np.full_like(x_plot, snapshot_t)
        rho_p, m_p = model.predict(x_plot, t_plot, sd)
        plt.plot(x_plot, rho_p, colors[sd % len(colors)] + '-',
                 linewidth=2, label=f'cPINN subdomain {sd + 1}')
    for xi in x_interface[1:-1]:
        plt.axvline(xi, color='gray', linestyle=':', alpha=0.6)
    plt.xlabel('x'); plt.ylabel(r'$\rho$')
    plt.title(f'Isentropic Euler density at t = {snapshot_t:.3f} (cPINN vs. finite volume)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('cPINN_euler_density.png', dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))
    for i, hist in enumerate(loss_history):
        plt.plot(hist, label=f'Sub-PINN-{i + 1}')
    plt.yscale('log')
    plt.xlabel('iterations (x200)')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Per-subdomain training loss')
    plt.tight_layout()
    plt.savefig('cPINN_euler_loss.png', dpi=150)
    plt.show()