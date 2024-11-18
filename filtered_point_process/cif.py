import numpy as np
from scipy.fftpack import ifft
import json
import os
import warnings
from .helpers import (
    TimeDomain,
    FrequencyDomain,
    create_frequency_domain,
    create_time_domain,
)
from .ParamSetter import ParamSetter, GlobalSeed
from .utils import nextpow2


class ConditionalIntensityFunction(ParamSetter, GlobalSeed):
    
    def __init__(self, params=None, seed=None, config_file=None):
        super().set_params(config_file, params)
        super()._set_seed(seed)
        del seed

        # Time_axis
        N = int(self.params["T"] * self.params["fs"])  # Number of samples
        self._time_axis = np.linspace(
            0, self.params["T"], N, endpoint=False
        )  # Time vector

        # Check if frequency_vector is defined and has elements
        if "frequency_vector" in self.params and self.params["frequency_vector"] is not None:
            # Ensure the frequency vector is squeezed if needed
            frequency_vector = self.params["frequency_vector"].squeeze()
            
            # Calculate NFFT
            self.params["NFFT"] = self._nfft_via_frequency_vector(frequency_vector)

            # Generate the full frequency vector including both positive and negative frequencies
            self.params["frequencies"] = self._generate_full_frequency_vector(self.params["NFFT"], frequency_vector)
            
        else: 
            # Frequencies
            self.params["NFFT"] = self.__compute_NFFT()
            self.params["frequencies"] = self.__generate_frequency_vector(method="total")

        # Theoretical Power Spectrum
        self.PSD = self._spectrum()

        self.frequency_domain = create_frequency_domain(
            self.params["frequencies"], self.PSD
        )

    ######################### CORE FUNCTIONALITY #########################

    def _nfft_via_frequency_vector(self, frequency_vector):
        nfft = 2 * len(frequency_vector) - 1
        return nfft

    def _generate_full_frequency_vector(self, nfft, positive_frequency_vector):
        """
        Generate a full frequency vector including both positive and negative frequencies symmetrically.
        """

        # Ensure the positive frequency vector is valid
        if not isinstance(positive_frequency_vector, np.ndarray) or len(positive_frequency_vector) == 0:
            raise ValueError("positive_frequency_vector must be a non-empty numpy array.")

        # Exclude zero from the positive frequency vector for mirroring
        non_zero_positive_frequencies = positive_frequency_vector[positive_frequency_vector != 0]

        # Generate negative frequencies by mirroring the non-zero positive frequencies
        negative_frequencies = np.flip(non_zero_positive_frequencies)

        # Combine negative frequencies and the original positive frequencies
        full_frequency_vector = np.concatenate([negative_frequencies, positive_frequency_vector])

        return full_frequency_vector

    
    def _spectrum(self):
        """Compute the theoretical power spectral density."""

        if self.params["method"] == "gaussian":
            return self._compute_cox_cif_S()

        elif self.params["method"] == "Homogeneous_Poisson":
            return self._compute_homogeneous_cif_S()

        elif self.params["method"] == "AR(p)":
            return self._compute_ar_cif_S()

        else:
            raise ValueError("Invalid method")

    def _simulate(self):
        """Generate a simulation of a time series process."""
        method_simulation = {
            "gaussian": self._simulate_gaussian,
            "Homogeneous_Poisson": self._simulate_homog_pois,
            "AR(p)": self._simulate_ar,
        }

        simulation_func = method_simulation.get(self.params["method"])
        if not simulation_func:
            raise ValueError(f"Invalid method: {self.params['method']}")

        intensity = simulation_func()

        self.time_domain = create_time_domain(
            self._time_axis, intensity_realization=intensity
        )

    def __compute_NFFT(self):
        """Compute NFFT based on number of points in the time vector."""

        N = self.params["T"] * self.params["fs"]
        if nextpow2(N) < 100:
            return 100000 #+ nextpow2(N) 
        else:
            #return 100 * nextpow2(N)
            return 100000 #+ nextpow2(N)

    def __generate_frequency_vector(self, method="total"):
        """
        Generate a frequency vector for the theoretical power spectrum.

        Method can be either total (all negative and positive frequencies) or positive (which only returns positive frequencies up to Nyquist (i.e., fs // 2))

        Parameters:
        nfft (int): Number of FFT points.
        fs (float): Sampling frequency.

        Returns:
        np.ndarray: Frequency vector.

        Raises:
        ValueError: If the method is not 'total' or 'positive'.
        """
        if method == "total":
            # Generate the frequency vector
            frequencies = np.fft.fftfreq(self.params["NFFT"], d=1 / self.params["fs"])
            return abs(frequencies) # was abs()
        elif method == "positive":
            frequencies = np.fft.fftfreq(self.params["NFFT"], d=1 / self.params["fs"])
            return frequencies[:self.params["NFFT"] // 2]
        else:
            # Raise an error if the method is not recognized
            raise ValueError("Invalid method. Choose 'total' or 'positive'.")

    ######################### HOMOGENEOUS POISSON PROCESS #########################

    def _simulate_homog_pois(self):
        """Simulate homogenous Poisson process."""

        homogenous_pois_cif_S = self._compute_homogeneous_cif_S()
        return np.real(self._rate_function(homogenous_pois_cif_S, self.params["fs"]))

    def _compute_homogeneous_cif_S(self):
        """Compute the theoretical power spectrum for the homogeneous Poisson process (HPP).
        There is no frequency dependence for a HPP and since the power
        spectrum of a conditional intensity function (or intensity function
        in this case) is capturing the variance in the rate, there is no power
        with the DC offset removed. Otherwise, all power is at 0. Here, we assumed
        a removed DC offset, like all other CIFs in the package."""

        return 0 * np.ones(self.params["frequencies"].shape)

    def _rate_function(self, S, fs):
        """Generate realizations of a homogeneous Poisson process in the time domain,
        with each realization being a horizontal line at lambda."""

        # Create an array with dimensions (number of simulations, length of t)
        rate_function_realizations = np.tile(
            self.params["lambda_0"] * np.ones_like(self._time_axis),
            (self.params["Nsims"], 1),
        )

        return rate_function_realizations

    ######################### COX PROCESS #########################

    ######################### GAUSSIAN PROCESS CIF #########################

    def _simulate_gaussian(self):
        """Simulate Gaussian process. This is the combined function for generating
        the theoretical power spectrum of the CIF (i.e., via _compute_cox_cif_S) and
        the time realization of the Gaussian (i.e., via _simulate_gaussian_process_approx_fd)
        """

        NFFT = self.__compute_NFFT()
        cox_cif_S = self._compute_cox_cif_S()
        intensity_realization = self._simulate_gaussian_process_approx_fd(
            cox_cif_S, len(self._time_axis), self.params["fs"]
        )
        return intensity_realization[0]

    def _compute_cox_cif_S(self):

        frequencies = self.params["frequencies"].squeeze()
        NFFT = self.params["NFFT"]
        M = len(frequencies)
        Mby2 = M // 2

        # Initialize the power spectrum with zeros
        cox_cif_S = np.zeros_like(frequencies)

        # Extract the base parameters
        center_frequencies = [self.params["center_frequency"]]
        peak_widths = [self.params["peak_width"]]
        peak_heights = [self.params["peak_height"]]

        # Check for additional sets of parameters
        index = 2
        while f"center_frequency_{index}" in self.params:
            center_frequencies.append(self.params[f"center_frequency_{index}"])
            peak_widths.append(self.params[f"peak_width_{index}"])
            peak_heights.append(self.params[f"peak_height_{index}"])
            index += 1

        # Sum the contributions from all Gaussian peaks
        for center_frequency, peak_width, peak_height in zip(center_frequencies, peak_widths, peak_heights):
            cox_cif_S += peak_height * np.exp(
                -(
                    (frequencies - center_frequency) ** 2
                )
                / (2 * peak_width ** 2)
            )

        cox_cif_S[0] = 0

        if M % 2 == 0:
            cox_cif_S[Mby2:] = np.flip(cox_cif_S[1:Mby2 + 1])
        else:
            cox_cif_S[Mby2 + 1:] = np.flip(cox_cif_S[1:Mby2 + 1])


        return cox_cif_S


    
    def _simulate_gaussian_process_approx_fd(self, S, N, fs):
        """Generate time domain realizations of a Gaussian process from the frequency
        defined Gaussian process in _compute_U_freqdomain."""
        M = S.shape[0]
        U_freqdomain = self._compute_U_freqdomain(S, M)
        Y = np.real_if_close(np.sqrt(fs * M) * ifft(U_freqdomain, axis=0))
        cif_timedomain = Y[:N, :] + self.params["lambda_0"]
        cif_timedomain = np.real_if_close(cif_timedomain)
        cif_timedomain[cif_timedomain < 0] = 0  # Enforce non-negative rate
        return cif_timedomain, U_freqdomain

 
    
    def _compute_U_freqdomain(self, S, M):
        """
        _compute_U_freqdomain: Compute U in the frequency domain, which is a complex representation
        of a Gaussian process for simulation purposes.
        """
        U_freqdomain = np.zeros((M, self.params["Nsims"]), np.complex128)
        U_freqdomain[0, :] = np.sqrt(S[0]) * np.random.randn(self.params["Nsims"])
        Mby2 = M // 2
        std = np.sqrt(S[1:Mby2 + 1] / 2)  # Adjusted to include the Mby2 index
        U_freqdomain[1:Mby2 + 1, :] = std[:, None] * np.random.randn(Mby2, self.params["Nsims"]) + 1j * std[:, None] * np.random.randn(Mby2, self.params["Nsims"])
        U_freqdomain[Mby2 + 1:] = np.flipud(np.conj(U_freqdomain[1:M - Mby2, :]))  # Correct length for flipping
        if M % 2 == 0:
            U_freqdomain[Mby2, :] = np.sqrt(S[Mby2]) * np.random.randn(self.params["Nsims"])
        return U_freqdomain


    ######################### AR(p) CIF #########################


    def _compute_ar_cif_S(self):
        """
            Calculate the theoretical power spectrum for an AR(p) process.
        """
        
        M = len(self.params["frequencies"])
        Mby2 = M // 2
        p = len(self.params["ar_coeffs"])
        omega = (2 * np.pi * self.params["frequencies"] * (1 / self.params["fs"]))
        ar_coeffs = np.array(self.params["ar_coeffs"])
        omega = 2 * np.pi * np.arange(M) / M
        denom = 1 - np.sum(
            ar_coeffs[None, :] *
            np.exp(-1j * omega[:, None] * np.arange(1, p + 1)[None, :]),
            axis=1
        )
        spectrum = ((self.params["white_noise_variance"] ** 2) * (1 / self.params["fs"])) / ((np.abs(denom) ** 2))
        
        # Symmetrize the spectrum for even and odd lengths
        if M % 2 == 0:
            spectrum[Mby2:] = np.flipud(spectrum[1:Mby2 + 1])
        else:
            spectrum[Mby2 + 1:] = np.flipud(spectrum[1:Mby2 + 1])

        return spectrum

    def _compute_ar_time(self, n_points, n_sims):
        """Generate AR(p) time series."""
        p = len(self.params["ar_coeffs"])
        ar_coeffs = self.params["ar_coeffs"]
        time_series_array = np.zeros((n_sims, n_points))
        
        for sim in range(n_sims):
            white_noise = np.random.normal(0, np.sqrt(self.params["white_noise_variance"]), n_points)
            time_series = np.zeros(n_points)
            for t in range(p, n_points):
                # Generate AR(p) process time series
                time_series[t] = np.sum([ar_coeffs[j] * time_series[t - j - 1] for j in range(p)]) + white_noise[t]
            
            time_series_array[sim, :] = time_series
            time_series_array += self.params["lambda_0"]
            time_series_array[time_series_array < 0] = 0
        
        return time_series_array

    def _simulate_ar(self):
        """Simulate the AR(p) process and return the time series."""
        t = np.arange(0, self.params["T"], (1 / self.params["fs"]))
        
        #ar_cif_S = self._compute_ar_cif_S()
        #ar_cif_S[0] = 0  # Enforce zero DC component
        
        n_sims = self.params["Nsims"]
        return self._compute_ar_time(n_points=len(t), n_sims=n_sims)
