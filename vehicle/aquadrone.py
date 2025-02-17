#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import numpy as np
import math
# import random if you need it inside this file, but now we won't do random here

from lib.control import PIDpolePlacement
from lib.gnc import Smtrx, Hmtrx, Rzyx, Tzyx, m2c, crossFlowDrag, sat, attitudeEuler

class Aquadrone:
    """    
    Aquadrone USV model, now with an external interface for updating mp and rp.
    """

    def __init__(self, V_current=0, beta_current=0):
        # --- Some basic constants ---
        D2R = math.pi / 180
        self.g = 9.81
        self.rho = 1025

        # Environment
        self.V_c = V_current
        self.beta_c = beta_current * D2R

        # Initialize Aquadrone USV states
        self.T_n = 0.1
        self.thrust_actual = np.zeros(2)
        self.L = 2.0
        self.B = 1.08
        self.st = np.zeros(12, float)    
        self.u_actual = np.array([0, 0], float) 
        self.name = "Aquadrone USV (time-varying payload)"

        self.controls = ["Left propeller shaft speed (rad/s)",
                         "Right propeller shaft speed (rad/s)"]
        self.dimU = len(self.controls)

        # Base hull parameters
        self.m_hull = 55.0
        self.rg_hull = np.array([0.2, 0, -0.2], float)

        # Payload (initially):
        self.mp = 25.0
        self.rp = np.array([0.05, 0, -0.35], float) # location of payload

        # geometry for moment of inertia calculations
        self.R44 = 0.4 * self.B
        self.R55 = 0.25 * self.L
        self.R66 = 0.25 * self.L

        # For damping
        self.T_sway = 1.0 
        self.T_yaw  = 1.0
        self.Umax   = 6 * 0.5144

        # geometry for one pontoon
        self.B_pont = 0.25
        self.y_pont = 0.395
        self.Cw_pont = 0.75
        self.Cb_pont = 0.4

        # prop data
        self.l1 = -self.y_pont
        self.l2 =  self.y_pont
        self.k_pos = 0.02216 / 2
        self.k_neg = 0.01289 / 2
        self.n_max = math.sqrt((0.5 * 24.4 * self.g) / self.k_pos)
        self.n_min = -math.sqrt((0.5 * 13.6 * self.g) / self.k_neg)

        # Initialize placeholders for mass/inertia/hydro & damping
        self.M    = np.zeros((6,6))
        self.Minv = np.zeros((6,6))
        self.G    = np.zeros((6,6))
        self.D    = np.zeros((6,6))
        self.Binv = np.zeros((2,2))

        # PID states for heading control (optional usage)
        self.e_int = 0  
        self.wn    = 2.5    
        self.zeta  = 1
        self.r_max = 10*math.pi/180
        self.psi_d = 0
        self.r_d   = 0
        self.a_d   = 0
        self.wn_d  = 0.5
        self.zeta_d= 1

        # store new added mass and linear drag
        self.MA_coef = np.zeros(6)
        self.D_coef = np.zeros(6)
        

        # Compute initial parameters
        self.update_parameters()

    def update_parameters(self):
        """
        Recompute mass, inertia, and hydrodynamic properties
        whenever mp or rp changes.
        """
        # (1) total mass
        self.m_total = self.m_hull + self.mp

        # (2) new CG
        rg = (self.m_hull*self.rg_hull + self.mp*self.rp) / self.m_total
        self.S_rg = Smtrx(rg)
        self.S_rp = Smtrx(self.rp)
        self.H_rg = Hmtrx(rg)
        self.S_rp = Smtrx(self.rp)

        R44 = 0.4 * self.B  # radii of gyration (m)
        R55 = 0.25 * self.L
        R66 = 0.25 * self.L

        # Inertia dyadic, volume displacement and draft
        nabla = (self.m_hull + self.mp) / self.rho  # volume
        self.T = nabla / (2 * self.Cb_pont * self.B_pont * self.L)  # draft
        Ig_CG_hull = self.m_hull * np.diag(np.array([R44 ** 2, R55 ** 2, R66 ** 2]))
        self.Ig = Ig_CG_hull - self.m_hull * self.S_rg @ self.S_rg - self.mp * self.S_rp @ self.S_rp


        # MRB_CG = [ (m+mp) * I3  O3      (Fossen 2021, Chapter 3)
        #               O3       Ig ]
        MRB_CG = np.zeros((6, 6))
        MRB_CG[0:3, 0:3] = (self.m_hull + self.mp) * np.identity(3)
        MRB_CG[3:6, 3:6] = self.Ig
        MRB = self.H_rg.T @ MRB_CG @ self.H_rg
        
        # Hydrodynamic added mass (best practice)
        Xudot = -0.1 * self.m_total
        Yvdot = -1.5 * self.m_total
        Zwdot = -1.0 * self.m_total
        Kpdot = -0.2 * self.Ig[0, 0]
        Mqdot = -0.8 * self.Ig[1, 1]
        Nrdot = -1.7 * self.Ig[2, 2]
        self.MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])
        self.MA_coef = np.array([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])

        self.M    = MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # (6) Recompute waterline volume, draft T
        nabla = self.m_total/self.rho
        self.T = nabla/(2*self.Cb_pont*self.B_pont*self.L)

        # Hydrostatic quantities (Fossen 2021, Chapter 4)
        Aw_pont = self.Cw_pont * self.L * self.B_pont  # waterline area, one pontoon
        I_T = (
            2
            * (1 / 12)
            * self.L
            * self.B_pont ** 3
            * (6 * self.Cw_pont ** 3 / ((1 + self.Cw_pont) * (1 + 2 * self.Cw_pont)))
            + 2 * Aw_pont * self.y_pont ** 2
        )
        I_L = 0.8 * 2 * (1 / 12) * self.B_pont * self.L ** 3
        KB = (1 / 3) * (5 * self.T / 2 - 0.5 * nabla / (self.L * self.B_pont))
        BM_T = I_T / nabla  # BM values
        BM_L = I_L / nabla
        KM_T = KB + BM_T    # KM values
        KM_L = KB + BM_L
        KG = self.T - rg[2]
        GM_T = KM_T - KG    # GM values
        GM_L = KM_L - KG

        G33 = self.rho * self.g * (2 * Aw_pont)  # spring stiffness
        G44 = self.rho * self.g * nabla * GM_T
        G55 = self.rho * self.g * nabla * GM_L
        G_CF = np.diag([0, 0, G33, G44, G55, 0])  # spring stiff. matrix in CF
        LCF = -0.2
        H = Hmtrx(np.array([LCF, 0.0, 0.0]))  # transform G_CF from CF to CO
        self.G = H.T @ G_CF @ H

        # Natural frequencies
        w3 = math.sqrt(G33 / self.M[2, 2])
        w4 = math.sqrt(G44 / self.M[3, 3])
        w5 = math.sqrt(G55 / self.M[4, 4])

        # Linear damping terms (hydrodynamic derivatives)
        Xu = -24.4 *self. g / self.Umax   # specified using the maximum speed
        Xu = Xu - 1 * self.mp
        Yv = -self.M[1, 1]  / self.T_sway # specified using the time constant in sway
        Zw = -2 * 0.3 * w3 * self.M[2, 2]  # specified using relative damping
        Kp = -2 * 0.2 * w4 * self.M[3, 3]
        Mq = -2 * 0.4 * w5 * self.M[4, 4]
        Nr = -self.M[5, 5] / self.T_yaw  # specified by the time constant T_yaw

        self.D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])
        self.D_coef = np.array([Xu, Yv, Zw, Kp, Mq, Nr])

        # Propeller configuration/input matrix
        B = self.k_pos * np.array([[1, 1], [-self.l1, -self.l2]])
        self.Binv = np.linalg.inv(B)

    @staticmethod
    def ssa(angle):
        return (angle + math.pi) % (2*math.pi) - math.pi

    def dynamics(self, st, thrust):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the Aquadrone USV equations of motion using Euler's method.
        """

        eta = st[:6]
        nu = st[6:]
        

        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[5])  # current surge vel.
        v_c = self.V_c * math.sin(self.beta_c - eta[5])  # current sway vel.

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)  # current velocity vector
        Dnu_c = np.array([nu[5]*v_c, -nu[5]*u_c, 0, 0, 0, 0],float) # derivative
        nu_r = nu - nu_c  # relative velocity vector

        # Rigid body and added mass Coriolis and centripetal matrices
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3   (Fossen 2021, Chapter 6)
        #              O3                   -Smtrx(Ig*nu2)  ]
        CRB_CG = np.zeros((6, 6))
        CRB_CG[0:3, 0:3] = self.m_total * Smtrx(nu[3:6])
        CRB_CG[3:6, 3:6] = -Smtrx(np.matmul(self.Ig, nu[3:6]))
        CRB = self.H_rg.T @ CRB_CG @ self.H_rg  # transform CRB from CG to CO

        CA = m2c(self.MA, nu_r)
        # Uncomment to cancel the Munk moment in yaw, if stability problems
        # CA[5, 0] = 0  
        # CA[5, 1] = 0 
        # CA[0, 5] = 0
        # CA[1, 5] = 0

        C = CRB + CA

        # Payload force and moment expressed in BODY
        R = Rzyx(eta[3], eta[4], eta[5])
        T_0 = Tzyx(eta[3], eta[4])
        f_payload = np.matmul(R.T, np.array([0, 0, self.mp * self.g], float))              
        m_payload = np.matmul(self.S_rp, f_payload)
        g_0 = np.array([ f_payload[0],f_payload[1],f_payload[2], 
                         m_payload[0],m_payload[1],m_payload[2] ])

        
        # Control forces and moments
        tau = np.array(
            [
                thrust[0] + thrust[1],
                0,
                0,
                0,
                0,
                -self.l1 * thrust[0] - self.l2 * thrust[1],
            ]
        )

        # print("tau: \n", tau)
        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -np.matmul(self.D, nu_r)
        tau_damp[5] = tau_damp[5] - 10 * self.D[5, 5] * abs(nu_r[5]) * nu_r[5]

        # State derivatives (with dimension)
        tau_crossflow = crossFlowDrag(self.L, self.B_pont, self.T, nu_r)
        sum_tau = (
            tau
            + tau_damp
            + tau_crossflow
            - np.matmul(C, nu_r)
            - np.matmul(self.G, eta)
            + g_0
        )

        nu_dot = Dnu_c + np.matmul(self.Minv, sum_tau)  # USV dynamics

        J_0 = np.block([
            [R, np.zeros((3, 3))],
            [np.zeros((3, 3)), T_0]
        ])
        eta_dot = np.matmul(J_0, nu)
        
        x_dot = np.concatenate((eta_dot, nu_dot))

        return x_dot

    def thruster_dynamics(self, con, sampleTime):
        """
        output thrust for each thruster 
        """
        
        thrust_dot = (con - self.thrust_actual) / self.T_n  # propeller dynamics
        self.thrust_actual = self.thrust_actual + sampleTime * thrust_dot


        return self.thrust_actual
    
    def step(self, st, u_control, dt):
        
        thrust = self.thruster_dynamics(u_control, sampleTime=dt)
        # thrust = u_control
        # print("u_control :\n", u_control)
        print("thrust :\n", thrust)
        k1 = self.dynamics(st, thrust)
        k2 = self.dynamics(st + 0.5 * dt * k1, thrust)
        k3 = self.dynamics(st + 0.5 * dt * k2, thrust)
        k4 = self.dynamics(st + dt * k3, thrust)
        st_next = st + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        st_next[5] = self.ssa(st_next[5])  # wrap yaw angle
        # print("st_next: \n ", st_next)
        return st_next
    

    def controlAllocation(self, tau_X, tau_N):
        """
        [n1, n2] = controlAllocation(tau_X, tau_N)
        """
        tau = np.array([tau_X, tau_N])  # tau = B * u_alloc
        u_alloc = np.matmul(self.Binv, tau)  # u_alloc = inv(B) * tau

        # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)
        n1 = np.sign(u_alloc[0]) * math.sqrt(abs(u_alloc[0]))
        n2 = np.sign(u_alloc[1]) * math.sqrt(abs(u_alloc[1]))

        return n1, n2
    def simulate_trash_collection(self, new_mp, new_rp):
        """
        Overwrite internal mp and rp, then update all mass/inertia/hydro params.
        """
        self.mp = new_mp
        self.rp = new_rp
        # Then call the main parameter update
        self.update_parameters()
