#spectral_gaussian.py
import numpy as np
import warnings
from scipy.fftpack import ifft
from filtered_point_process.cif.BaseCIF import CIFBase
from filtered_point_process.domains.time_domain import create_time_domain, TimeDomain
from filtered_point_process.domains.frequency_domain import (
    create_frequency_domain,
    FrequencyDomain,
)


class GaussianCIF(CIFBase):
    """
    Initialize the GaussianCIF with multiple Gaussian peaks.

    Args:
        peak_height (list of float): Heights of the peaks in the PSD.
        center_frequency (list of float): Center frequencies of the Gaussian peaks.
        peak_width (list of float): Widths (standard deviations) of the Gaussian peaks.
        lambda_0 (float, optional): Baseline intensity. Defaults to 1.0.
        frequencies (np.ndarray, optional): Frequency vector. Defaults to None.
        NFFT (int, optional): Number of FFT points. Defaults to None.
        fs (float, optional): Sampling frequency in Hz. Defaults to None.
        seed (int, optional): Seed for random number generation. Defaults to None.
        simulate (bool, optional): Whether to perform simulation. Defaults to False.
        T (float, optional): Total time in seconds. Required if simulate is True. Defaults to None.
        Nsims (int, optional): Number of simulations to generate. Defaults to 1.

    Raises:
        ValueError: If the lengths of peak_height, center_frequency, and peak_width do not match.
    """

    def __init__(
        self,
        peak_height,
        center_frequency,
        peak_width,
        lambda_0=1.0,
        frequencies=None,
        NFFT=None,
        fs=None,
        seed=None,
        simulate=False,
        T=None,
        Nsims=1,
    ):
        # Ensure the lists of peaks are of the same length
        if not (len(peak_height) == len(center_frequency) == len(peak_width)):
            raise ValueError(
                "Lists peak_height, center_frequency, and peak_width must be of the same length."
            )

        self.peak_height = peak_height
        self.center_frequency = center_frequency
        self.peak_width = peak_width
        self.lambda_0 = lambda_0

        self.bump_PSDs: list[np.ndarray] | None = None

        super().__init__(frequencies, NFFT, fs, seed, simulate, T, Nsims)
    
    def compute_bump_spectra(self) -> list[np.ndarray]:
        """
        Return a list containing the PSD for *each* Gaussian bump
        (all frequencies), with every other bump turned off.

        The result is cached on first use and reused thereafter.
        """
        if self.bump_PSDs is not None:
            return self.bump_PSDs

        f = self.frequencies.squeeze()
        bump_spectra: list[np.ndarray] = []

        for height, center, width in zip(
            self.peak_height, self.center_frequency, self.peak_width
        ):
            bump = (
                height
                * np.exp(-((f - center) ** 2) / (2 * width**2))
                + height * np.exp(-((f + center) ** 2) / (2 * width**2))
            )
            # force 0‑Hz power to zero
            bump[np.argmin(np.abs(f))] = 0.0
            bump_spectra.append(bump)

        self.bump_PSDs = bump_spectra
        return bump_spectra

    def _compute_spectrum_legacy(self):
        """
        Compute the theoretical power spectral density (PSD) for the Gaussian CIF with multiple "Gaussian-like" peaks.

        This method calculates the PSD by summing Gaussian-shaped contributions for each specified peak. The PSD is enforced to be zero at zero frequency.

        Returns:
            np.ndarray: The computed power spectral density corresponding to the frequency vector.
        """
        frequencies = self.frequencies.squeeze()
        PSD = np.zeros_like(frequencies)

        # Sum Gaussian contributions for each peak
        for height, center, width in zip(
            self.peak_height, self.center_frequency, self.peak_width
        ):
            PSD += height * np.exp(
                -((frequencies - center) ** 2) / (2 * width**2)
            ) + height * np.exp(-((frequencies + center) ** 2) / (2 * width**2))

        # Set PSD at zero frequency to zero
        zero_freq_index = np.argmin(np.abs(frequencies))
        PSD[zero_freq_index] = 0

        return PSD

    def _compute_spectrum(self) -> np.ndarray:
        """
        Build the *total* PSD as the sum of all bump‑specific PSDs and
        (for bookkeeping) cache every individual bump spectrum.
        """
        bump_spectra = self.compute_bump_spectra()
        PSD = np.sum(bump_spectra, axis=0)

        # sanity check – the sum of individual bumps must equal total
        ## TO DO: MAKE INTO A TEST
        if len(bump_spectra) > 1:  # nothing to check if only one bump
            recon = np.sum(bump_spectra, axis=0)
            if not np.allclose(recon, PSD, rtol=1e-10, atol=1e-12):
                raise RuntimeError(
                    "Internal consistency check failed: "
                    "sum of individual bump PSDs ≠ total PSD."
                )

        return PSD

    def _simulate_time_domain(self):
        """
        Simulate the Gaussian process in the time domain.

        This method generates a time-domain realization of the Gaussian CIF by computing the inverse FFT of the frequency-domain representation. It adds the baseline intensity and enforces non-negativity by setting any negative values to zero.

        Returns:
            np.ndarray: Simulated CIF in the time domain with shape (N,).

        Raises:
            UserWarning: If any values in the intensity are negative after adding the baseline intensity.

        Citation: "Simulating Gaussian Random Processes with Specified Spectra" via Percival, 1992 (method 4)
        """
        M = len(self.PSD)
        U_freqdomain = self._compute_U_freqdomain(self.PSD, M)
        Y = np.real_if_close(
            np.sqrt(self.fs * M) * ifft(np.fft.ifftshift(U_freqdomain), axis=0)
        )
        cif_timedomain = Y[: self.N, :] + self.lambda_0
        cif_timedomain = np.real_if_close(cif_timedomain)
        negative_values = cif_timedomain < 0
        num_negative = np.sum(negative_values)
        if num_negative > 0:
            maximum_negative = np.min(cif_timedomain[negative_values])
            warnings.warn(
                f"{num_negative} values in the intensity were negative after adding lambda_0 "
                f"and have been set to zero. Consider increasing lambda_0. Given your simulation "
                f"parameters we recommend a value of at least {self.lambda_0 + np.abs(maximum_negative)} for lambda_0",
                UserWarning,
            )
        cif_timedomain[negative_values] = 0  # Enforce non-negative rate
        return cif_timedomain

    def _compute_U_freqdomain(self, S, M):
        """
        Compute the complex frequency-domain representation U for the Gaussian process.

        This method constructs the complex frequency-domain array U by generating random Gaussian variables for each frequency component, ensuring symmetry required for a real-valued time-domain signal.

        Args:
            S (np.ndarray): Power spectral density array.
            M (int): Number of frequency components.

        Returns:
            np.ndarray: Complex frequency-domain representation U with shape (M, Nsims).
        """
        U_freqdomain = np.zeros((M, self.Nsims), np.complex128)
        U_freqdomain[0, :] = np.sqrt(S[0]) * self.random_state.randn(self.Nsims)
        Mby2 = M // 2
        std = np.sqrt(S[1 : Mby2 + 1] / 2)
        U_freqdomain[1 : Mby2 + 1, :] = std[:, None] * self.random_state.randn(
            Mby2, self.Nsims
        ) + 1j * std[:, None] * self.random_state.randn(Mby2, self.Nsims)
        U_freqdomain[Mby2 + 1 :] = np.flipud(
            np.conj(U_freqdomain[1 : M - Mby2, :])
        ) 
        if M % 2 == 0:
            U_freqdomain[Mby2, :] = np.sqrt(S[Mby2]) * self.random_state.randn(self.Nsims)
        return U_freqdomain


class SumLinearLambda0:
    """
    Compute baseline:
        λ₀ = 3 * sqrt( sum_i [ h_i * gamma_0(cf_i, w_i) ] )
    where gamma_0(cf, w) is the zero-lag autocovariance of a unit-height bump.
    """

    @staticmethod
    def gamma0_for_bump(center_frequency: float, peak_width: float, fs: float) -> float:
        """
        Compute gamma_0 for a single unit-height Gaussian bump at `center_frequency` Hz
        with width `peak_width` Hz, using the same PSD code as GaussianCIF.
        """
        from filtered_point_process.cif.Spectral_Gaussian import GaussianCIF

        # instantiate a zero-mean, unit-height, single-bump CIF (no time simulate)
        temp = GaussianCIF(
            peak_height=[1.0],
            center_frequency=[center_frequency],
            peak_width=[peak_width],
            lambda_0=1.0,
            fs=fs,
            simulate=False,
        )
        psd = temp.PSD
        # inverse-FFT → real(0) times fs
        return np.fft.ifft(np.fft.ifftshift(psd)).real[0] * fs

    @classmethod
    def compute(
        cls,
        peak_height: list[float],
        center_frequency: list[float],
        peak_width: list[float],
        fs: float,
    ) -> float:
        """
        Given lists of heights, center freqs, and widths, returns
            λ₀ = 3 * sqrt( sum_i [ h_i * gamma_0(cf_i, w_i) ] )
        """
        if not (len(peak_height) == len(center_frequency) == len(peak_width)):
            raise ValueError("All three lists must have the same length.")

        gammas = [
            cls.gamma0_for_bump(cf, w, fs)
            for cf, w in zip(center_frequency, peak_width)
        ]
        total = np.sum(np.array(peak_height) * np.array(gammas))
        return 3.0 * np.sqrt(total)


def _sum_linear_lambda0(self) -> float:
    """
    Compute λ₀ via the sum-linear rule for this CIF’s own peaks.
    """
    return SumLinearLambda0.compute(
        self.peak_height, self.center_frequency, self.peak_width, self.fs
    )

# attach it as a public method:
## TO DO: Refactor this
GaussianCIF.sum_linear_lambda0 = _sum_linear_lambda0
