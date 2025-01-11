import numpy as np
import warnings
from filtered_point_process.cif.BaseCIF import CIFBase
from filtered_point_process.domains.time_domain import create_time_domain


class ARCIF(CIFBase):
    """CIF class for an Auto-Regressive (AR) process.
    Initialize the Auto-Regressive Conditional Intensity Function (ARCIF).

    This class models an auto-regressive (AR) process for conditional intensity
    functions, allowing for spectral computation and time-domain simulations.

    Citation: Priestley, 1982 ()

    To do: double check chapter number

    Parameters
    ----------
    ar_coeffs : list or np.ndarray
        Auto-regressive coefficients [phi_1, phi_2, ..., phi_p] defining the AR process.
    white_noise_variance : float
        Variance of the white noise driving the AR process.
    lambda_0 : float, optional (default=1.0)
        Baseline intensity added to the AR process.
    frequencies : np.ndarray, optional
        Frequency vector for spectral analysis. If not provided, it will be
        generated based on `NFFT` and `fs`.
    NFFT : int, optional
        Number of points for the Fast Fourier Transform (FFT). Determines the
        frequency resolution.
    fs : float, optional
        Sampling frequency in Hertz (Hz). Required if `frequencies` is not provided.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
    simulate : bool, optional (default=False)
        Flag indicating whether to perform time-domain simulations upon initialization.
    T : float, optional
        Total simulation time in seconds. Must be specified if `simulate` is set to True.
    Nsims : int, optional (default=1)
        Number of independent simulations to generate if `simulate` is True.

    Raises
    ------
    ValueError
        If `simulate` is True and `T` is not provided.
    """

    def __init__(
        self,
        ar_coeffs,
        white_noise_variance,
        lambda_0=1.0,
        frequencies=None,
        NFFT=None,
        fs=None,
        seed=None,
        simulate=False,
        T=None,
        Nsims=1,
    ):
        self.ar_coeffs = np.array(ar_coeffs)
        self.white_noise_variance = white_noise_variance
        self.lambda_0 = lambda_0

        super().__init__(frequencies, NFFT, fs, seed, simulate, T, Nsims)

    def _compute_spectrum(self):
        """
        Compute the theoretical power spectral density (PSD) of the AR process.

        The power spectrum is calculated based on the AR coefficients and the white noise variance.
        The PSD at zero frequency is explicitly set to zero to account for the baseline intensity.

        Returns
        -------
        PSD : np.ndarray
            The computed power spectral density of the AR process across the specified frequencies.

        """
        frequencies = self.frequencies.squeeze()
        omega = 2 * np.pi * frequencies / self.fs
        ar_poly = 1 - np.sum(
            [
                a * np.exp(-1j * omega * k)
                for k, a in enumerate(self.ar_coeffs, start=1)
            ],
            axis=0,
        )
        PSD = (self.white_noise_variance**2) / (np.abs(ar_poly) ** 2)

        # Set PSD at zero frequency to zero
        zero_freq_index = np.argmin(np.abs(frequencies))
        PSD[zero_freq_index] = 0

        return PSD

    def _simulate_time_domain(self):
        """
        Simulate the AR process in the time domain.

        Generates a realization for the AR process by iteratively applying the AR coefficients
        to the previously generated values and adding white noise (i.e., just a "standard" AR simulation). Ensures that the resulting
        intensity values are non-negative by setting any negative values to zero and issuing a warning.

        Returns
        -------
        time_series : np.ndarray
            Simulated intensity time series with shape (N, Nsims), where N is the number of time points
            and Nsims is the number of simulations.

        """
        p = len(self.ar_coeffs)
        time_series_array = np.zeros((self.Nsims, self.N))
        randn = self.random_state.randn

        for sim in range(self.Nsims):
            white_noise = randn(self.N) * np.sqrt(self.white_noise_variance)
            cif_timedomain = np.zeros(self.N)
            for t in range(p, self.N):
                cif_timedomain[t] = (
                    np.dot(self.ar_coeffs, cif_timedomain[t - p : t][::-1])
                    + white_noise[t]
                )
            cif_timedomain += self.lambda_0
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
            time_series_array[sim, :] = cif_timedomain
        return time_series_array.T  # Transpose to match dimensions
