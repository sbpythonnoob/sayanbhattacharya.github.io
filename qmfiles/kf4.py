# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 19:42:51 2025

@author: sayan
"""

# Improved Koch Fractal Model implementation with plotting and better numerical stability
# Executable in this environment. Produces plots for:
#  - Bloch condition allowed energies (spectrum)
#  - Analytical DOS
#  - Commutation error vs energy (verifies commutation across band)
#  - Universality test across many (x,y) on the resonance circle
#
# Saves a script file at /mnt/data/koch_fractal_improved.py for download.

import numpy as np
import matplotlib.pyplot as plt
import math
import os

class KochFractalModelImproved:
    def __init__(self, t=1.0, x=None, y=None, Phi=None, epsilon=0.0, phi0=1.0, small_im=1e-12):
        self.t = float(t)
        self.epsilon = float(epsilon)
        self.phi0 = float(phi0)
        self.small_im = float(small_im)  # small imaginary part to regularize denominators

        # default x,y satisfying x^2 + y^2 = t^2
        if x is None and y is None:
            self.x = self.t / np.sqrt(2.0)
            self.y = self.t / np.sqrt(2.0)
        elif x is None:
            self.y = float(y)
            self.x = math.sqrt(max(0.0, self.t**2 - self.y**2))
        elif y is None:
            self.x = float(x)
            self.y = math.sqrt(max(0.0, self.t**2 - self.x**2))
        else:
            self.x = float(x)
            self.y = float(y)

        # flux -- default resonance phi0/4
        self.Phi = Phi if Phi is not None else self.phi0 / 4.0
        # Peierls phase per bond for triangular plaquette (symmetric choice)
        self.theta = 2.0 * np.pi * self.Phi / (3.0 * self.phi0)

    def _eb(self, E):
        # renormalized beta site energy (regularized with small_im)
        denom = (E - self.epsilon) + 1j * self.small_im
        return self.epsilon + (self.x**2) / denom

    def _ed(self, E):
        denom = (E - self.epsilon) + 1j * self.small_im
        return self.epsilon + 2.0 * (self.x**2) / denom

    def _lambda_F(self, E):
        # Return complex lambda forward using full complex arithmetic (more robust)
        E_eff = (E - self.epsilon) + 1j * self.small_im
        # follow algebraic expression: combine terms as complex numbers
        # lambda^2 = y^2 + (x^4 / E_eff^2) + (2 x^2 y cos(2 theta) / E_eff)
        term1 = (self.y**2)
        term2 = (self.x**4) / (E_eff**2)
        term3 = (2.0 * self.x**2 * self.y * np.cos(2.0 * self.theta)) / E_eff
        lambda_sq = term1 + term2 + term3
        # choose principal sqrt (numpy handles branch cuts consistently)
        lam = np.sqrt(lambda_sq)
        # compute phase robustly
        # numerator and denominator build the complex number whose angle is xi
        num = (self.y * np.sin(self.theta)) - ( (self.x**2 * np.sin(2.0 * self.theta)) / E_eff )
        den = (self.y * np.cos(self.theta)) + ( (self.x**2 * np.cos(2.0 * self.theta)) / E_eff )
        complex_for_angle = den + 1j * num
        xi = np.angle(complex_for_angle)
        return lam * np.exp(1j * xi)

    def get_transfer_matrices(self, E):
        """
        Construct M1, M2, M3 using robust complex arithmetic.
        Returns 2x2 complex numpy arrays.
        """
        eb = self._eb(E)
        #ed = self._ed(E)  # not explicitly used here but retained if needed
        lambdaF = self._lambda_F(E)
        # avoid tiny lambda magnitudes causing numerical blowups
        if np.abs(lambdaF) < 1e-14:
            # small regulator: add a bit to lambdaF's magnitude preserving phase
            lambdaF = (1e-14 + 0j) * np.exp(1j * np.angle(lambdaF))

        lambdaB = np.conjugate(lambdaF)

        # Carefully construct matrices matching the algebraic structure in the paper
        M1a = np.array([[0.0+0j, 1.0+0j], [-1.0+0j, (E - eb)/lambdaB]], dtype=complex)
        M1b = np.array([[(E - eb)/lambdaF, -self.t/lambdaF], [-self.t/lambdaF, 0.0+0j]], dtype=complex)
        M1 = M1a @ M1b

        M2 = np.array([[(E - self.epsilon)/self.t, -1.0+0j], [-1.0+0j, 0.0+0j]], dtype=complex)

        # For M3 we build the β-δ-δ-β block more explicitly to avoid ambiguous squaring
        # Sequence: β -> (β-δ connector) -> δ -> δ -> (δ-β connector) -> β
        # We use the same building blocks but ensure correct order
        M3a = M1a.copy()
        M3b = M1b.copy()
        M3c = M1b.copy()  # repeated block as in the algebra
        M3 = M3a @ M3b @ M3c @ M3b  # order chosen to reflect the four-site composite
        return M1, M2, M3

    def verify_commutation_over_E(self, E_vals, tol=1e-8):
        """
        Verify commutation [M2,M1]=0 and [M2,M3]=0 over an array of energies.
        Returns arrays of Frobenius norms of the commutators for each E.
        """
        errs_21 = np.zeros_like(E_vals, dtype=float)
        errs_23 = np.zeros_like(E_vals, dtype=float)
        for i, E in enumerate(E_vals):
            try:
                M1, M2, M3 = self.get_transfer_matrices(E)
                c21 = M2 @ M1 - M1 @ M2
                c23 = M2 @ M3 - M3 @ M2
                errs_21[i] = np.linalg.norm(c21, ord='fro')
                errs_23[i] = np.linalg.norm(c23, ord='fro')
            except Exception:
                errs_21[i] = np.nan
                errs_23[i] = np.nan
        return errs_21, errs_23

    def calculate_allowed_energies(self, E_vals):
        """
        For each energy, compute the total transfer matrix for one unit cell
        and test the Bloch condition |Tr(M_total)| <= 2. Returns boolean mask.
        """
        allowed_mask = np.zeros_like(E_vals, dtype=bool)
        traces = np.zeros_like(E_vals, dtype=complex)
        for i, E in enumerate(E_vals):
            if np.isclose(E, self.epsilon, atol=1e-12):  # avoid singularity at E=epsilon
                traces[i] = np.nan
                continue
            try:
                M1, M2, M3 = self.get_transfer_matrices(E)
                # total transfer for one "period" (matches the order used in Mathematica in many papers)
                M_total = M2 @ M1 @ M2 @ M3 @ M2 @ M1 @ M2
                tr = np.trace(M_total)
                traces[i] = tr
                allowed_mask[i] = (np.abs(tr) <= 2.0 + 1e-12)
            except Exception:
                traces[i] = np.nan
        return allowed_mask, traces

    def analytical_dos(self, E_vals):
        """
        Analytical 1D DOS for the effective 1D tight-binding chain at resonance.
        ρ(E) = 1/(π sqrt(4 t^2 - (E - ε)^2)) for |E-ε| < 2t
        """
        rho = np.zeros_like(E_vals, dtype=float)
        inside = np.abs(E_vals - self.epsilon) < 2.0 * self.t
        denom = (4.0 * self.t**2) - (E_vals[inside] - self.epsilon)**2
        rho[inside] = 1.0 / (np.pi * np.sqrt(denom + 1e-16))
        return rho

# --- Utility function to run full analysis and produce plots ---
def run_improved_analysis(save_script=True):
    print("Running improved Koch Fractal analysis...")

    # Initialize model with resonance defaults
    model = KochFractalModelImproved(t=1.0, x=1.0/np.sqrt(2.0), y=1.0/np.sqrt(2.0), Phi=0.25, epsilon=0.0)

    # Energy grid
    E_min, E_max = -2.5, 2.5
    num_points = 6000
    E_vals = np.linspace(E_min, E_max, num_points)

    # 1) Verify commutation across energies
    errs_21, errs_23 = model.verify_commutation_over_E(E_vals)
    max_err = np.nanmax(np.vstack([errs_21, errs_23]))
    print(f"Maximum commutator Frobenius norm over grid: {max_err:.3e}")

    # Plot commutation error vs E
    plt.figure(figsize=(8,4))
    plt.title("Commutation error ||[M2,M1]||_F and ||[M2,M3]||_F vs Energy")
    plt.semilogy(E_vals, errs_21, label='||[M2,M1]||_F')
    plt.semilogy(E_vals, errs_23, label='||[M2,M3]||_F', linestyle='--')
    plt.xlabel("Energy E")
    plt.ylabel("Frobenius norm (log scale)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Allowed energies using Bloch condition
    allowed_mask, traces = model.calculate_allowed_energies(E_vals)
    allowed_energies = E_vals[allowed_mask]
    print(f"Found {allowed_energies.size} allowed energy points (out of {E_vals.size}).")

    # Plot allowed energies as a band (scatter to show continuity)
    plt.figure(figsize=(8,4))
    plt.title("Allowed energies satisfying |Tr(M_total)| ≤ 2 (Bloch condition)")
    plt.plot(allowed_energies, np.zeros_like(allowed_energies), marker='.', linestyle='None', markersize=1)
    plt.xlabel("Energy E")
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    # 3) Analytical DOS and numerical histogram of allowed energies (proxy DOS)
    rho_analytic = model.analytical_dos(E_vals)

    # Numerically approximate DOS from allowed energy density (simple histogram smoothing)
    hist_bins = 300
    hist_vals, bin_edges = np.histogram(allowed_energies, bins=hist_bins, range=(E_min, E_max), density=True)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8,4))
    plt.title("Analytical DOS (1D chain) and numerical proxy from allowed energies")
    plt.plot(E_vals, rho_analytic, label="Analytical DOS")
    plt.plot(bin_centers, hist_vals, label="Numerical proxy DOS (histogram)", linewidth=1.0)
    plt.xlabel("Energy E")
    plt.ylabel("Density of states (arb. units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4) Universality test: sample many (x,y) on the circle x^2+y^2=t^2 and check commutation at one energy
    rng = np.random.default_rng(12345)
    thetas = rng.uniform(0, 2*np.pi, size=200)
    results = []
    E_test = 0.5
    for th in thetas:
        x_test = model.t * np.cos(th)
        y_test = model.t * np.sin(th)
        inst = KochFractalModelImproved(t=model.t, x=x_test, y=y_test, Phi=model.Phi, epsilon=model.epsilon)
        errs1, errs2 = inst.verify_commutation_over_E(np.array([E_test]))
        results.append((x_test, y_test, float(errs1[0]), float(errs2[0])))

    results = np.array(results, dtype=float)
    # fraction that have small commutator at E_test
    frac_ok = np.sum((results[:,2] < 1e-8) & (results[:,3] < 1e-8)) / results.shape[0]
    print(f"Fraction of random (x,y) on x^2+y^2=t^2 with small commutator at E={E_test}: {frac_ok:.3f}")

    plt.figure(figsize=(6,6))
    plt.title("Universality test: (x,y) samples on circle (color by commutator error)")
    sc = plt.scatter(results[:,0], results[:,1], c=np.log10(results[:,2] + results[:,3] + 1e-20), s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(sc, label='log10(commutator error sum)')
    circle = plt.Circle((0,0), model.t, color='black', fill=False, linewidth=1.0)
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Optionally save the improved script for download
    script_path = "/mnt/data/koch_fractal_improved.py"
    if save_script:
        code_text =  "# Improved Koch Fractal Model - saved script (auto-generated)\\n" + "\\n"
        code_text += open(__file__, 'r').read() if '__file__' in globals() else '# script contents saved separately\\n'
        try:
            with open(script_path, 'w') as f:
                # Write the same code executed here for user's convenience
                import inspect
                src = inspect.getsource(KochFractalModelImproved) + '\\n\\n' + inspect.getsource(run_improved_analysis)
                f.write('# Auto-saved Koch Fractal Improved implementation\\n\\n' + src)
            print(f"Saved script to: {script_path}")
        except Exception as e:
            print(f"Could not save script: {e}")

    return {
        "model": model,
        "E_vals": E_vals,
        "allowed_energies": allowed_energies,
        "errs_21": errs_21,
        "errs_23": errs_23,
        "rho_analytic": rho_analytic,
        "script_path": script_path if os.path.exists(script_path) else None
    }

# Run the analysis and produce plots
results = run_improved_analysis(save_script=True)

# Provide a download link if the file was created
if results.get("script_path", None) is not None:
    print(f"[Download the improved script](/mnt/data/koch_fractal_improved.py)")
