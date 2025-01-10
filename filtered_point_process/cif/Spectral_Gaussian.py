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

        super().__init__(frequencies, NFFT, fs, seed, simulate, T, Nsims)

    def _compute_spectrum(self):
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
        U_freqdomain[0, :] = np.sqrt(S[0]) * np.random.randn(self.Nsims)
        Mby2 = M // 2
        std = np.sqrt(S[1 : Mby2 + 1] / 2)
        U_freqdomain[1 : Mby2 + 1, :] = std[:, None] * np.random.randn(
            Mby2, self.Nsims
        ) + 1j * std[:, None] * np.random.randn(Mby2, self.Nsims)
        U_freqdomain[Mby2 + 1 :] = np.flipud(
            np.conj(U_freqdomain[1 : M - Mby2, :])
        )  # Correct length for flipping
        if M % 2 == 0:
            U_freqdomain[Mby2, :] = np.sqrt(S[Mby2]) * np.random.randn(self.Nsims)
        return U_freqdomain
