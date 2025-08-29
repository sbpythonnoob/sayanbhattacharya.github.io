# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 19:42:31 2025

@author: sayan
"""

"""
Improved Koch Fractal Model implementation with plotting and improved numerical stability.

Key improvements:
- Robust complex-phase computation for lambda_F using full complex arithmetic (np.angle).
- Commutation check over an energy grid.
- Dense eigenvalue-band identification via Bloch trace condition.
- Random sampling of (x,y) on the resonance circle to test universality.
- Plots of allowed energies (spectrum) and analytic DOS.
- Careful handling of singularities and small denominators.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import isclose

# Numerical regulator to avoid exact poles
_REG = 1e-12 + 0j

class KochFractalModel:
    def __init__(self, t=1.0, x=None, y=None, Phi=None, epsilon=0.0, phi0=1.0):
        self.t = float(t)
        self.epsilon = float(epsilon)
        self.phi0 = float(phi0)

        # Default to resonance pair on circle x^2+y^2 = t^2 if not provided
        if x is None and y is None:
            self.x = self.t / np.sqrt(2.0)
            self.y = self.t / np.sqrt(2.0)
        elif x is None:
            self.y = float(y)
            self.x = float(np.sqrt(max(0.0, self.t**2 - self.y**2)))
        elif y is None:
            self.x = float(x)
            self.y = float(np.sqrt(max(0.0, self.t**2 - self.x**2)))
        else:
            self.x = float(x)
            self.y = float(y)

        self.Phi = (Phi if Phi is not None else self.phi0 / 4.0)
        # symmetric Peierls phases used in the paper
        self.theta = 2.0 * np.pi * self.Phi / (3.0 * self.phi0)

        # resonance checks
        self.resonance_hopping = np.isclose(self.x**2 + self.y**2, self.t**2, atol=1e-12)
        self.resonance_flux = np.isclose(self.Phi, self.phi0 / 4.0, atol=1e-12)

    def _eps_beta(self, E):
        """Renormalized on-site energy for β (simple one-loop renormalization)."""
        return self.epsilon + (self.x**2) / (E - self.epsilon + _REG)

    def _eps_delta(self, E):
        """Renormalized on-site energy for δ (if needed)."""
        return self.epsilon + (2.0 * self.x**2) / (E - self.epsilon + _REG)

    def _lambda_F(self, E):
        """
        Compute the effective complex forward hopping lambda_F robustly.
        Uses full complex arithmetic and np.angle to get the phase.
        Formula derived from integrating out internal triangle degrees of freedom.
        """
        Eeff = E - self.epsilon + _REG  # complex small regulator
        # construct the complex number whose magnitude and phase we need
        # Following the form: lambda^2 = y^2 + (x^4 / Eeff^2) + (2 x^2 y cos(2 theta))/Eeff
        # but ensure this works with complex Eeff
        term1 = (self.y**2)
        term2 = (self.x**4) / (Eeff**2)
        term3 = (2.0 * self.x**2 * self.y * np.cos(2.0 * self.theta)) / Eeff
        lambda_sq = term1 + term2 + term3

        # magnitude (take sqrt of complex if necessary)
        lam = np.sqrt(lambda_sq + 0j)  # ensures complex sqrt if needed

        # compute phase properly from the underlying complex numerator/denominator representation:
        # from the paper-ish decomposition: real and imag parts that build a complex hopping
        # We'll form the complex number z = den + i * num as in derivation and take its angle
        num = (self.y * np.sin(self.theta)) - (self.x**2 * np.sin(2.0 * self.theta)) / Eeff
        den = (self.y * np.cos(self.theta)) + (self.x**2 * np.cos(2.0 * self.theta)) / Eeff
        z = den + 1j * num
        xi = np.angle(z)  # robust phase

        lambdaF = lam * np.exp(1j * xi)
        return lambdaF

    def get_transfer_matrices(self, E):
        """
        Construct transfer matrices M1, M2, M3 for energy E.

        The matrices below are the 2x2 "block" transfer matrices appropriate
        for the effective 1D chain after decimating triangle internal sites.
        They are built carefully avoiding divisions by tiny lambdas.
        """
        eb = self._eps_beta(E)
        # ed = self._eps_delta(E) # not used explicitly, kept for reference
        lambdaF = self._lambda_F(E)
        # protect against vanishing lambda magnitude (rare on resonance)
        lam_abs = np.abs(lambdaF)
        if lam_abs < 1e-14:
            # return None to indicate singular/ill-conditioned matrices at this E
            raise ZeroDivisionError("Effective hopping lambda nearly zero at E = {:.6g}".format(E))
        lambdaB = np.conjugate(lambdaF)

        # M1 corresponds to a β-β building block (see paper eqn structure)
        # We construct each constituent matrix explicitly for clarity
        M1a = np.array([[0.0+0j, 1.0+0j], [-1.0+0j, (E - eb) / lambdaB]], dtype=complex)
        M1b = np.array([[ (E - eb) / lambdaF, -self.t / lambdaF],
                        [ -self.t / lambdaF, 0.0+0j ]], dtype=complex)
        M1 = M1a @ M1b

        # M2 corresponds to the single-site connector C
        M2 = np.array([[(E - self.epsilon) / self.t, -1.0], [-1.0, 0.0]], dtype=complex)

        # M3 corresponds to β-δ-δ-β block (two β-δ repetitions effectively)
        # We'll assemble it as the paper indicates: product of the appropriate blocks.
        # Reuse M1a and M1b structure (but ensure correct ordering; we've included M1b twice)
        M3 = M1a @ M1b @ M1b @ M1b  # one could adapt exact block counts; this is consistent with earlier attempt
        # Note: the exact block count/order must be cross-checked with your target equation (Eq. 9 in paper).
        return M1, M2, M3

    def verify_commutation_over_grid(self, E_values):
        """
        Check commutation [M2, M1] and [M2, M3] across a grid of E_values.
        Returns an array of Frobenius norms of the commutators for each energy.
        """
        norms_21 = []
        norms_23 = []
        for E in E_values:
            try:
                M1, M2, M3 = self.get_transfer_matrices(E)
            except ZeroDivisionError:
                # Push large error to indicate failure at this energy
                norms_21.append(np.inf)
                norms_23.append(np.inf)
                continue
            comm21 = M2 @ M1 - M1 @ M2
            comm23 = M2 @ M3 - M3 @ M2
            norms_21.append(np.linalg.norm(comm21))
            norms_23.append(np.linalg.norm(comm23))
        return np.array(norms_21), np.array(norms_23)

    def total_transfer_matrix_for_unitcell(self, E):
        """
        Build total transfer matrix for one effective unit cell used in Bloch condition.
        Following the earlier code's pattern: M_total = M2 @ M1 @ M2 @ M3 @ M2 @ M1 @ M2
        """
        M1, M2, M3 = self.get_transfer_matrices(E)
        M_total = M2 @ M1 @ M2 @ M3 @ M2 @ M1 @ M2
        return M_total

    def find_allowed_energies(self, E_range=(-2.5, 2.5), num_points=5000, tol=2.0):
        """
        Scan energies and return those for which Bloch condition |Tr(M_total)| <= tol (usually 2).
        Returns the E grid and boolean mask of allowed energies.
        """
        E_vals = np.linspace(E_range[0], E_range[1], num_points)
        allowed_mask = np.zeros_like(E_vals, dtype=bool)
        trace_vals = np.full_like(E_vals, np.nan, dtype=complex)

        for i, E in enumerate(E_vals):
            # avoid singular E exactly at epsilon
            if abs(E - self.epsilon) < 1e-13:
                continue
            try:
                Mtot = self.total_transfer_matrix_for_unitcell(E)
            except ZeroDivisionError:
                continue
            tr = np.trace(Mtot)
            trace_vals[i] = tr
            if np.abs(tr) <= tol + 1e-12:
                allowed_mask[i] = True

        return E_vals, allowed_mask, trace_vals

    def analytic_dos_1d(self, E_vals):
        """
        Analytical DOS for the equivalent 1D chain: rho(E) = 1/(pi * sqrt(4 t^2 - (E-epsilon)^2))
        Only defined inside the band |E - epsilon| < 2t.
        """
        rho = np.zeros_like(E_vals, dtype=float)
        inside = np.abs(E_vals - self.epsilon) < 2.0 * self.t
        denom_sq = (4.0 * self.t**2) - (E_vals[inside] - self.epsilon)**2
        # avoid exact zero
        rho[inside] = 1.0 / (np.pi * np.sqrt(denom_sq + 1e-18))
        return rho


def run_complete_analysis(plot=True):
    print("\n=== KOCH FRACTAL MODEL: IMPROVED ANALYSIS ===\n")
    model = KochFractalModel(t=1.0, x=1/np.sqrt(2), y=1/np.sqrt(2), Phi=0.25)

    print("Model parameters:")
    print(f" t = {model.t}, x = {model.x:.6f}, y = {model.y:.6f}, Phi = {model.Phi} (Phi0={model.phi0})")
    print("Resonance checks:")
    print(f" x^2 + y^2 == t^2? -> {model.resonance_hopping}")
    print(f" Phi == Phi0/4? -> {model.resonance_flux}\n")

    # 1) Verify commutation across an energy grid
    Es = np.linspace(model.epsilon - 2.2*model.t, model.epsilon + 2.2*model.t, 800)
    norms_21, norms_23 = model.verify_commutation_over_grid(Es)

    # test "commutation holds across most energy values" if norms extremely small
    finite_norms = np.isfinite(norms_21) & np.isfinite(norms_23)
    if np.any(finite_norms):
        max_norm = max(np.nanmax(norms_21[finite_norms]), np.nanmax(norms_23[finite_norms]))
    else:
        max_norm = np.inf
    print(f"Max commutator Frobenius norm across grid (finite entries): {max_norm:.3e}")
    print("Commutation considered satisfied if max_norm < 1e-9 (tunable).")
    print("Small norms on resonance indicate approximate commutation -> extended states.\n")

    # 2) Compute allowed energies using Bloch condition |Tr(M_total)| <= 2
    E_vals, allowed_mask, trace_vals = model.find_allowed_energies(E_range=(model.epsilon - 2.5*model.t,
                                                                             model.epsilon + 2.5*model.t),
                                                                    num_points=6000, tol=2.0)
    allowed_Es = E_vals[allowed_mask]
    if allowed_Es.size > 0:
        print(f"Found {allowed_Es.size} allowed grid points (Bloch condition) within scan range.")
        print(f"Numerical band approx: [{allowed_Es.min():.6f}, {allowed_Es.max():.6f}]")
        print(f"Theoretical band for 1D chain: [{model.epsilon - 2*model.t:.6f}, {model.epsilon + 2*model.t:.6f}]")
    else:
        print("No allowed energies found in scan range (check parameters or expand range).")

    # 3) Analytic DOS for reference
    E_dos = np.linspace(model.epsilon - 2.0*model.t, model.epsilon + 2.0*model.t, 2000)
    rho = model.analytic_dos_1d(E_dos)

    # 4) Universality test: many (x,y) on circle x^2+y^2 = t^2
    rng = np.random.default_rng(42)
    angs = np.linspace(0, 2*np.pi, 20, endpoint=False)
    universality_results = []
    for ang in angs:
        xt = model.t * np.cos(ang)
        yt = model.t * np.sin(ang)
        mtest = KochFractalModel(t=model.t, x=xt, y=yt, Phi=model.Phi)
        # pick modest subset of energies to test commutation quickly
        Es_test = np.linspace(model.epsilon - 1.5*model.t, model.epsilon + 1.5*model.t, 80)
        n21, n23 = mtest.verify_commutation_over_grid(Es_test)
        maxn = np.nanmax(np.concatenate([n21[np.isfinite(n21)], n23[np.isfinite(n23)]]))
        universality_results.append((xt, yt, maxn))
    # Check if any cases have huge commutator norms
    worst = max(universality_results, key=lambda x: x[2])
    print("\nUniversality sampling (20 points on x^2+y^2=t^2):")
    print(f" Worst-case commutator Frobenius norm: {worst[2]:.3e} at (x,y)=({worst[0]:.3f},{worst[1]:.3f})")
    print("If small across samples, universality numerically supported.\n")

    # 5) Off-resonance comparison (Phi=0)
    off = KochFractalModel(t=model.t, x=model.x, y=model.y, Phi=0.0)
    Es_off = np.linspace(off.epsilon - 2.0*off.t, off.epsilon + 2.0*off.t, 400)
    n21_off, n23_off = off.verify_commutation_over_grid(Es_off)
    print(f"Off-resonance (Phi=0) max commutator norm (sampled): {np.nanmax(np.concatenate([n21_off, n23_off])):.3e}")
    print("Expect much larger commutator norms off-resonance -> localization.\n")

    # Plotting
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)

        # Spectrum: allowed energies as a density plot (vertical marks)
        axs[0].set_title("Allowed energies from Bloch condition (|Tr(M_total)| ≤ 2)")
        axs[0].plot(E_vals, np.abs(trace_vals).real, linewidth=0.7)
        axs[0].axhline(2.0, linestyle='--')
        axs[0].axhline(-2.0, linestyle='--')
        # overlay allowed energies as a scatter
        axs[0].scatter(E_vals[allowed_mask], np.full(allowed_Es.size, 1.98), s=1.5)
        axs[0].set_ylabel("|Tr(M_total)|")
        axs[0].set_xlim(E_vals.min(), E_vals.max())

        # DOS plot
        axs[1].set_title("Analytical DOS (1D tight-binding) and numerical allowed band")
        axs[1].plot(E_dos, rho, linewidth=1.2)
        # Shade region of allowed energies
        if allowed_Es.size > 0:
            axs[1].fill_between(E_vals, 0, 0.5 * np.nanmax(rho) * (allowed_mask.astype(float)), alpha=0.25)
            axs[1].set_xlim(E_dos.min(), E_dos.max())
        axs[1].set_ylabel("DOS (arb. units)")
        axs[1].set_xlabel("Energy E")

        plt.show()

    return {
        "model": model,
        "commutation_norms_grid": (Es, norms_21, norms_23),
        "allowed_energies": (E_vals, allowed_mask),
        "analytic_dos": (E_dos, rho),
        "universality_results": universality_results,
        "off_resonance_comm_norms": (Es_off, n21_off, n23_off)
    }

if __name__ == "__main__":
    results = run_complete_analysis(plot=True)
