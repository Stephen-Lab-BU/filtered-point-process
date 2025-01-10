import numpy as np
import warnings
from filtered_point_process.cif.BaseCIF import CIFBase
from filtered_point_process.domains.time_domain import create_time_domain


class HomogeneousPoissonCIF(CIFBase):
    """
    Initialize the Homogeneous Poisson Conditional Intensity Function (CIF).

    Parameters
    ----------
    lambda_0 : float
        Baseline intensity (rate) of the Poisson process. Must be a non-negative value.
    frequencies : np.ndarray, optional
        Frequency vector for spectral analysis. If not provided, it will be
        generated based on `NFFT` and `fs`.
    NFFT : int, optional
        Number of points for the Fast Fourier Transform (FFT). Determines the frequency
        resolution. If not provided, a default value is used based on `fs`.
    fs : float, optional
        Sampling frequency in Hertz (Hz). Required if `frequencies` is not provided.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility. If `None`, the
        random number generator is not seeded.
    simulate : bool, optional (default=False)
        Flag indicating whether to perform time-domain simulations upon initialization.
    T : float, optional
        Total simulation time in seconds. Must be specified if `simulate` is set to `True`.
    Nsims : int, optional (default=1)
        Number of independent simulations to generate if `simulate` is `True`.

    Raises
    ------
    ValueError
        If `simulate` is `True` and `T` is not provided.
        If `lambda_0` is negative.
    """

    def __init__(
        self,
        lambda_0,
        frequencies=None,
        NFFT=None,
        fs=None,
        seed=None,
        simulate=False,
        T=None,
        Nsims=1,
    ):
        self.lambda_0 = lambda_0
        super().__init__(frequencies, NFFT, fs, seed, simulate, T, Nsims)

    def _compute_spectrum(self):
        """
        Compute the theoretical power spectral density (PSD) for the Homogeneous Poisson process.

        The PSD of a homogeneous Poisson process is constant across all frequencies and is
        equal to the baseline intensity `lambda_0`. This method returns a PSD array where each
        element corresponds to the power at a specific frequency.


        Returns
        -------
        PSD : np.ndarray
            The computed power spectral density, which is a constant array with all elements
            equal to `lambda_0`.

        Warnings
        --------
        UserWarning
            Alerts the user that the spectrum of the CIF for a Homogeneous Poisson process is
            identical to the spectrum of the point process itself.
        """
        PSD = self.lambda_0 * np.ones(len(self.frequencies))
        warnings.warn(
            "The 'CIF' of a homogeneous process is its constant intensity."
            "As such, the spectrum of the 'CIF' of a Homo. Pois. process and"
            "The spectrum at the level of the point process is identical.",
            UserWarning,
        )
        return PSD

    def _simulate_time_domain(self):
        """
        Simulate the Homogeneous Poisson process in the time domain.

        This method generates simulated intensity time series for the Homogeneous Poisson process.
        Since the intensity is constant (`lambda_0`), the simulation simply returns an array filled
        with `lambda_0`. The shape of the returned array depends on the number of simulations (`Nsims`):

        - If `Nsims` is 1, returns a 1D array of shape (N,).
        - If `Nsims` is greater than 1, returns a 2D array of shape (N, Nsims).

        The method ensures that the intensity values are non-negative, adhering to the properties
        of a Poisson process.

        Returns
        -------
        intensity : np.ndarray
            Simulated intensity time series.
            - Shape (N,) if `Nsims` == 1.
            - Shape (N, Nsims) if `Nsims` > 1.

        """
        if self.Nsims == 1:
            intensity = self.lambda_0 * np.ones(self.N)
        else:
            intensity = self.lambda_0 * np.ones((self.N, self.Nsims))
        return intensity
