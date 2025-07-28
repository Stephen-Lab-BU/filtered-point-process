#basecif.py
import numpy as np
import warnings
from abc import ABC, abstractmethod
from filtered_point_process.domains.frequency_domain import create_frequency_domain
from filtered_point_process.domains.frequency_domain import FrequencyDomain
from filtered_point_process.domains.time_domain import create_time_domain, TimeDomain
from filtered_point_process.utils.helpers import nextpow2


class CIFBase(ABC):
    """
    Initialize the CIFBase class.

    Args:
        frequencies (np.ndarray): Frequency vector. Required for spectrum computation.
        NFFT (int): Number of FFT points.
        fs (float): Sampling frequency in Hz.
        seed (int): Seed for random number generation.
        simulate (bool): Whether to perform simulation. If True, T must be provided.
        T (float): Total time in seconds. Required if simulate is True.
        Nsims (int): Number of simulations to generate.
    """

    def __init__(
        self,
        frequencies=None,
        NFFT=None,
        fs=None,
        seed=None,
        simulate=False,
        T=None,
        Nsims=1,
    ):
        if Nsims > 1 or Nsims < 1:
            raise ValueError(
                "Only a single realization is supported right now. To get multiple realizations, use an uncontrolled seed and use a loop. This will be addressed in future versions. Please set 'Nsim': 1."
            )
        self.fs = fs
        self.NFFT = NFFT
        self.random_state = np.random.RandomState(seed)

        if simulate:
            if T is None:
                raise ValueError("Total time T must be provided for simulation.")
            self.n_samples = int(T * self.fs)
        else:
            self.n_samples = 100_000  # or set a default value if needed

        # Handle NFFT and fs according to the new requirements
        self._handle_nfft_fs()

        if frequencies is not None:
            self.frequencies = frequencies
        else:
            self.frequencies = self._generate_frequency_vector(self.NFFT, self.fs)

        self.PSD = self._compute_spectrum()
        self.frequency_domain = create_frequency_domain(self.frequencies, self.PSD)

        # Time domain attributes
        self.time_domain = None
        self.simulate = simulate

        if simulate:
            if T is None:
                raise ValueError("Total time T must be provided for simulation.")
            self.Nsims = Nsims
            if Nsims is None:
                self.Nsims = 1
            self.T = T

            self.N = int(T * self.fs)  # Number of time samples
            self.time_axis = np.linspace(0, T, self.N, endpoint=False)
            intensity = self._simulate_time_domain()
            self.time_domain = create_time_domain(
                self.time_axis, intensity_realization=intensity
            )

    def _handle_nfft_fs(self):
        """Set based on a reasonable contiuous heuristic or the user's specification of NFFT and fs."""
        # Ensure fs is provided
        if self.fs is None:
            raise ValueError("Sampling frequency fs must be provided.")

        # Set default NFFT if not provided
        #if self.NFFT is None:
            #self.NFFT = nextpow2(self.n_samples) + 100_000

        target_df = 0.01  
        desired_len = int(np.ceil(self.fs / target_df))
        self.NFFT = nextpow2(desired_len)
        #print(f"{self.NFFT} is NFFT")

        # Check if fs exceeds the threshold
        #if self.fs > 15_000:
        #    self.NFFT =  nextpow2(self.n_samples) + 100_000
        #    warnings.warn(
        #        "Sampling frequency fs is greater than 15,000 Hz. "
        #        "Functions and plots will be very computationally expensive.",
        #        UserWarning,
        #    )

    @abstractmethod
    def _compute_spectrum(self):
        """Compute the theoretical power spectral density (PSD)."""
        pass

    @staticmethod
    def _generate_frequency_vector(NFFT, fs):
        """Generate a symmetric frequency vector for FFT."""
        frequencies = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1 / fs))
        return frequencies

    @abstractmethod
    def _simulate_time_domain(self):
        """Simulate the time-domain process."""
        pass

    @property
    def cif_time_axis(self):
        """Get the time axis of the CIF."""
        if self.time_domain:
            return self.time_domain.get_time_axis()
        else:
            raise ValueError(
                "Time domain data is not available. Please set 'simulate=True' and provide 'T'."
            )

    @property
    def cif_realization(self):
        """Get the CIF realization."""
        if self.time_domain:
            return self.time_domain.get_intensity_realization()
        else:
            raise ValueError(
                "Time domain data is not available. Please set 'simulate=True' and provide 'T'."
            )

    @property
    def cif_frequencies(self):
        """Get the frequency axis of the CIF."""
        return self.frequency_domain.get_frequencies()

    @property
    def cif_PSD(self):
        """Get the PSD of the CIF."""
        PSD = self.frequency_domain.get_PSD()
        return PSD
