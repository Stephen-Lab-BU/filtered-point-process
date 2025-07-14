
import numpy as np
import warnings
from filtered_point_process.cif.Spectral_Gaussian import GaussianCIF
from filtered_point_process.cif.HomogeneousPoisson import HomogeneousPoissonCIF
from filtered_point_process.domains.frequency_domain import create_frequency_domain
from filtered_point_process.domains.time_domain import create_time_domain
from scipy.fftpack import ifft


class MultivariateCIF:
    """
    Multivariate Conditional-Intensity-Function container that supports both
    purely frequency-domain analysis and full time-domain simulation.

    If ``simulate=False`` (the default) we build:
      • individual CIF objects (all with simulate=False)  
      • per-process PSDs (`self.spectra`)  
      • full cross-spectral matrix (`self.cross_spectra`)  
      • a `FrequencyDomain` wrapper (`self.frequency_domain`)

    When ``simulate=True`` we additionally:
      • require a total time `T`  
      • generate `self.time_series` (shape PxNxNsims)  
      • create a `TimeDomain` wrapper (`self.time_domain`)
    """

    def __init__(
        self,
        num_processes,
        cif_types,
        cif_params,
        fs,
        NFFT=None,
        seed=None,
        simulate=False,
        T=None,
        Nsims=1,
        dependence="independent",
        weights=None,
    ):
        self.num_processes = num_processes
        self.cif_types = cif_types
        self.cif_params = cif_params
        self.fs = fs
        self.NFFT = NFFT
        self.seed = seed
        self.simulate = bool(simulate)
        self.T = T
        self.Nsims = Nsims
        if Nsims is None:
            self.Nsims = 1
        self.dependence = dependence
        self.random_state = np.random.RandomState(seed)
        self.num_bumps = None   # will be filled once CIFs are built
        self.weights   = None

        # Guard: time-domain quantities only if simulate=True
        if self.simulate:
            if self.T is None:
                raise ValueError("Total time T must be provided when simulate=True.")
            self.N = int(self.T * self.fs)
        else:
            self.N = None  # purely frequency-domain mode

        # ---------------- build individual CIFs ------------------------ #
        self.cifs = []
        for i in range(num_processes):
            cif_type = cif_types[i]
            params = cif_params[i]

            if cif_type == "Gaussian":
                cif = GaussianCIF(fs=fs, NFFT=NFFT, seed=seed, simulate=False, **params)

            elif cif_type == "HomogeneousPoisson":
                if dependence != "independent":
                    raise ValueError("Homogeneous Poisson CIFs can only be independent.")
                cif = HomogeneousPoissonCIF(
                    fs=fs, NFFT=NFFT, seed=seed, simulate=False, **params
                )
                # Mark that this multivariate container is Poisson
                self._cif_type_label = "HomogeneousPoisson"

            elif cif_type == "AR":
                raise NotImplementedError("Multivariate AR CIF is not implemented yet.")

            else:
                raise ValueError(f"Unknown or unsupported CIF type: {cif_type}")

            self.cifs.append(cif)

        # Parameter-consistency check for dependent Gaussian case
        if dependence == "dependent":
            self._check_cif_parameters_consistency()

        self.num_bumps = len(self.cifs[0].compute_bump_spectra())
        if weights is None:
            weights = np.ones(self.num_processes)
        self._stash_weights(weights)

        # Shared frequency axis
        self.frequencies = self.cifs[0].cif_frequencies

        # ---------------- frequency-domain objects --------------------- #
        self._compute_spectra()         
        self._compute_cross_spectra()   
        self.cif_PSD = self.spectra     # Rename (TO DO: don't do this)
        self.frequency_domain = create_frequency_domain(self.frequencies, self.cif_PSD)

        if self.simulate:
            self._simulate_time_domain_multi()

            self.time_domain = create_time_domain(
                self.time_axis, intensity_realization=self.time_series
            )

    # ─────────────────────────────────────────────────────────────
    def _stash_weights(self, w):
        """Store weights as an nd-array of shape (P,Q)."""
        w = np.asarray(w, dtype=complex)
        if w.ndim == 1:                       # legacy shape (P,)
            self.weights = w[:, None]         # broadcast to (P,1) for now
        elif w.ndim == 2:
            if w.shape[0] != self.num_processes:
                raise ValueError("weights.shape[0] must equal num_processes.")
            self.weights = w
        else:
            raise ValueError("weights must be 1-D or 2-D.")

    def _compute_spectra(self):
        """
        Build per-process auto-spectra + total field spectrum
        when dependence == 'dependent' **with bump-specific weights**.
        """
        if self.dependence == "independent":
            # unchanged
            self.spectra       = [cif.cif_PSD for cif in self.cifs]
            self.field_spectrum = np.sum(self.spectra, axis=0)
            return

        # -------------- DEPENDENT CASE --------------
        bump_PSDs = self.cifs[0].compute_bump_spectra()   # list length Q
        Q          = len(bump_PSDs)
        if self.weights is None:
            raise ValueError("Weights must be provided for dependent processes.")

        # ensure weights have shape (P,Q)
        if self.weights.ndim == 1:                  # (P,)  → broadcast
            self.weights = np.repeat(self.weights[:, None], Q, axis=1)
        elif self.weights.shape[1] != Q:
            raise ValueError(
                f"weights.shape[1] (={self.weights.shape[1]}) must equal num_bumps Q={Q}."
            )

        P = self.num_processes
        w = self.weights                           # shape (P,Q)

        self.spectra = []
        for i in range(P):
            Sii = np.zeros_like(self.frequencies, dtype=float)
            for q in range(Q):
                Sii += np.abs(w[i, q]) ** 2 * bump_PSDs[q]
            self.spectra.append(Sii)

        λ0_sum = sum(cif.lambda_0 for cif in self.cifs)
        κ = np.zeros_like(self.frequencies, dtype=float)
        for q in range(Q):
            κ += (np.abs(np.sum(w[:, q])) ** 2) * bump_PSDs[q]
        self.field_spectrum = κ + λ0_sum

    
    def get_field_spectrum(self):
        """Return the total field PSD S_field(f)."""
        return self.field_spectrum

    def get_component_spectra(self):
        """Return the weighted per-process auto-spectra (same as self.spectra)."""
        return self.spectra


    def _compute_cross_spectra(self):
        """
        Cross-spectral matrix with bump-specific weights.
        """
        P, F = self.num_processes, len(self.frequencies)
        X = np.zeros((P, P, F), dtype=complex)

        if self.dependence == "independent":
            for i in range(P):
                X[i, i, :] = self.spectra[i]
            self.cross_spectra = X
            return

        # ---------- dependent w/ bump weights ----------
        bump_PSDs = self.cifs[0].compute_bump_spectra()   # list length Q
        Q          = len(bump_PSDs)
        w = self.weights                                  # (P,Q)

        for i in range(P):
            for j in range(P):
                Sij = np.zeros(F, dtype=complex)
                for q in range(Q):
                    Sij += w[i, q] * np.conj(w[j, q]) * bump_PSDs[q]
                X[i, j, :] = Sij

        self.cross_spectra = X
  

    def _simulate_time_domain_multi(self):
        """
        Simulate intensity realisations for each subprocess (shape PxNxNsims).
        """
        if self.dependence == "independent":
            self.time_series = []
            for cif in self.cifs:
                cif.simulate   = True
                cif.T          = self.T
                cif.Nsims      = self.Nsims
                cif.N          = int(self.T * cif.fs)
                cif.time_axis  = np.linspace(0, self.T, cif.N, endpoint=False)

                cif_td = cif._simulate_time_domain()
                cif.time_domain = create_time_domain(
                    cif.time_axis, intensity_realization=cif_td
                )
                self.time_series.append(cif_td)

            self.time_series = np.array(self.time_series)          # (P,N,Nsims)
            self.time_axis   = self.cifs[0].time_axis

        elif self.dependence == "dependent":
            # ---- validate weights ------------------------------------------------
            bump_PSDs = self.cifs[0].compute_bump_spectra()
            Q         = len(bump_PSDs)
            w = np.asarray(self.weights, dtype=complex)
            if w.ndim == 1:
                w = np.repeat(w[:, None], Q, axis=1)
            if w.shape != (self.num_processes, Q):
                raise ValueError(
                    f"weights must have shape (P,Q)=({self.num_processes},{Q}); "
                    f"got {w.shape}."
                )
            self.weights = w   # store cleaned weights

            for cif in self.cifs:
                cif.simulate  = True
                cif.T         = self.T
                cif.Nsims     = self.Nsims
                cif.N         = int(self.T * cif.fs)
                cif.time_axis = np.linspace(0, self.T, cif.N, endpoint=False)

            # ---- check all CIFs identical -----------------------------------
            ref = self.cifs[0]
            for idx, cif in enumerate(self.cifs[1:], 1):
                if cif.__class__ != ref.__class__ or not np.allclose(cif.PSD, ref.PSD):
                    raise ValueError(
                        f"All CIFs must be identical for dependent mode (mismatch @ {idx})."
                    )

            # ---- generate one Gaussian driver per bump ----------------------
            M = len(ref.PSD)
            U_bumps = [ref._compute_U_freqdomain(psd, M) for psd in bump_PSDs]  # Q items

            # ---- build per-process frequency-domain signals -----------------
            V = np.zeros((self.num_processes, M, self.Nsims), dtype=complex)
            for q in range(Q):
                V += w[:, q, None, None] * U_bumps[q]   # broadcast over i

            # ---- IFFT to time-domain ----------------------------------------
            N = int(self.T * self.fs)
            self.time_series = np.zeros((self.num_processes, N, self.Nsims))
            self.time_axis   = np.linspace(0, self.T, N, endpoint=False)

            for i, cif in enumerate(self.cifs):
                Y_i = np.real_if_close(
                    np.sqrt(self.fs * M) *
                    ifft(np.fft.ifftshift(V[i], axes=0), axis=0)
                )
                λ_i = Y_i[:N, :] + cif.lambda_0
                λ_i[λ_i < 0] = 0.0
                self.time_series[i] = λ_i

                cif.time_domain = create_time_domain(
                    self.time_axis, intensity_realization=λ_i
                )


        else:
            raise ValueError(f"Unknown dependence type: {self.dependence}")

    def get_spectra(self):
        return self.spectra

    def get_total_spectrum(self):
        return self.field_spectrum

    def get_cross_spectra(self):
        if self.dependence != "dependent":
            raise ValueError("Cross-spectra are only available for dependent processes.")
        return self.cross_spectra

    def get_time_series(self):
        if not self.simulate:
            raise ValueError("Time series not available (simulate=False).")
        return self.time_series

    def get_time_axis(self):
        if not self.simulate:
            raise ValueError("Time axis not available (simulate=False).")
        return self.time_axis

    def get_frequencies(self):
        return self.frequencies

    @property
    def PSD(self):
        return self.spectra

    def _check_cif_parameters_consistency(self):
        """Ensure identical Gaussian spectral parameters when dependence='dependent'."""
        ref = {
            "peak_height": self.cifs[0].peak_height,
            "center_frequency": self.cifs[0].center_frequency,
            "peak_width": self.cifs[0].peak_width,
        }
        for i, cif in enumerate(self.cifs[1:], 1):
            cur = {
                "peak_height": cif.peak_height,
                "center_frequency": cif.center_frequency,
                "peak_width": cif.peak_width,
            }
            if cur != ref:
                raise ValueError(
                    f"Inconsistent Gaussian parameters for dependent processes.\n"
                    f"Process 0: {ref}\nProcess {i}: {cur}"
                )

class SumLinearLambda0Multivariate:
    """
    Compute per-process baselines for a dependent MultivariateCIF by:
      1) computing the sum-linear baseline λ_shared = 3√(∑_j h_j gamma_0_j)
         for the underlying multi-bump PSD (using GaussianCIF.sum_linear_lambda0),
      2) then scaling by |w_i|:
         λ₀ᵢ = |w_i| * λ_shared.
    """

    @staticmethod
    def compute(mv: "MultivariateCIF") -> list[float]:
        if mv.dependence != "dependent":
            raise ValueError("Only for dependent MultivariateCIF.")

        base      = mv.cifs[0]
        h_q       = np.array(base.peak_height, float)                    # shape (Q,)
        fs        = base.fs
        cf_q      = base.center_frequency
        width_q   = base.peak_width

        # gamma_0 for each bump
        from filtered_point_process.cif.Spectral_Gaussian import SumLinearLambda0
        gamma_q = np.array(
            [SumLinearLambda0.gamma0_for_bump(cf, w, fs) for cf, w in zip(cf_q, width_q)]
        )                                     # shape (Q,)

        # weights matrix (P,Q)
        w = np.asarray(mv.weights, dtype=float)
        if w.ndim == 1:                       # legacy (P,)
            w = w[:, None]

        # λ0_i = 3 * sqrt( Σ_q  h_q * γ_q * |w_{iq}|² )
        λ0_list = 3.0 * np.sqrt( (h_q * gamma_q) @ (np.abs(w)**2).T )
        return λ0_list.tolist()


def _sum_linear_lambda0(self) -> list[float]:
    return SumLinearLambda0Multivariate.compute(self)

# TO DO: refactor this 
MultivariateCIF.sum_linear_lambda0 = _sum_linear_lambda0
