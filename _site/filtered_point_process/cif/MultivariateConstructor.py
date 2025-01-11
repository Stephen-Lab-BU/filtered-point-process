import numpy as np
import warnings
from filtered_point_process.cif.Spectral_Gaussian import GaussianCIF
from filtered_point_process.cif.HomogeneousPoisson import HomogeneousPoissonCIF
from filtered_point_process.domains.frequency_domain import create_frequency_domain
from filtered_point_process.domains.time_domain import create_time_domain
from scipy.fftpack import ifft


class MultivariateCIF:
    """
    Initialize the MultivariateCIF instance.

    Parameters
    ----------
    num_processes : int
        Number of subprocesses.
    cif_types : list of str
        List specifying the CIF type for each subprocess. Supported types include
        "Gaussian", "HomogeneousPoisson", and "AR".
    cif_params : list of dict
        List of parameter dictionaries corresponding to each CIF type. Each dictionary
        should contain the necessary parameters for initializing the respective CIF.
    fs : float
        Sampling frequency in Hertz (Hz).
    NFFT : int, optional
        Number of points to determine the frequency
        resolution. If not provided, a default value is used based on `fs`.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility. If None, the
        random number generator is not seeded.
    simulate : bool, optional (default=False)
        Flag indicating whether to perform time-domain simulations upon initialization.
    T : float, optional
        Total simulation time in seconds. Must be specified if `simulate` is set to True.
    Nsims : int, optional (default=1)
        Number of independent simulations to generate if `simulate` is True.
    dependence : str, optional (default="independent")
        Type of dependence between subprocesses. Must be either "independent" or "dependent".
        - "independent": Subprocesses operate independently.
        - "dependent": Subprocesses are linearly dependent based on provided weights.
    weights : list or np.ndarray, optional
        Weights for linear transformation in dependent Gaussian processes. Required if
        `dependence` is set to "dependent". Each weight corresponds to a subprocess and
        defines its contribution to the shared underlying Gaussian process.

    Raises
    ------
    ValueError
        - If an unsupported CIF type is provided in `cif_types`.
        - If `dependence` is set to "dependent" but `weights` are not provided or their
            length does not match `num_processes`.
        - If `simulate` is True but `T` is not provided.
    NotImplementedError
        - If a CIF type that is not yet implemented (e.g., "AR") is specified.
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
        self.simulate = simulate
        self.T = T
        self.N = int(self.T * self.fs)
        self.Nsims = Nsims
        self.dependence = dependence
        self.weights = weights
        self.random_state = np.random.RandomState(seed)

        # Initialize individual CIFs
        self.cifs = []
        for i in range(num_processes):
            cif_type = cif_types[i]
            params = cif_params[i]
            if cif_type == "Gaussian":
                cif = GaussianCIF(fs=fs, NFFT=NFFT, seed=seed, simulate=False, **params)
            elif cif_type == "HomogeneousPoisson":
                if dependence != "independent":
                    raise ValueError(
                        "Homogeneous Poisson CIFs can only be independent."
                    )
                cif = HomogeneousPoissonCIF(
                    fs=fs, NFFT=NFFT, seed=seed, simulate=False, **params
                )
                self._cif_type_label = "HomogeneousPoisson"
            elif cif_type == "AR":
                raise NotImplementedError("Multivariate AR CIF is not implemented yet.")
            else:
                raise ValueError(f"Unknown or unsupported CIF type: {cif_type}")
            self.cifs.append(cif)

        # Check for parameter consistency if dependent
        if dependence == "dependent":
            self._check_cif_parameters_consistency()

        # Use the frequencies from the first CIF
        self.frequencies = self.cifs[0].cif_frequencies

        # Compute the spectra for each subprocess
        self._compute_spectra()

        # Set the cif_PSD attribute
        self.cif_PSD = self.spectra

        # Create the frequency domain object
        self.frequency_domain = create_frequency_domain(self.frequencies, self.cif_PSD)

        # If dependent and Gaussian, compute cross-spectra
        if dependence == "dependent":
            for cif_type in cif_types:
                if cif_type != "Gaussian":
                    raise ValueError(
                        "Dependent processes are only supported for Gaussian CIFs."
                    )
            self._compute_cross_spectra_gaussian()

        # Simulate if required
        if simulate:
            if T is None:
                raise ValueError("Total time T must be provided for simulation.")
            self._simulate_time_domain_multi()

            # Set the cif_realization attribute
            self.cif_realization = self.time_series

            # Create the time domain object
            self.time_domain = create_time_domain(
                self.time_axis, intensity_realization=self.cif_realization
            )

    def _compute_spectra(self):
        """
        Compute the power spectral densities (PSDs) for each subprocess.

        This method iterates through each CIF instance in `self.cifs` and retrieves
        their respective PSDs. The computed PSDs are stored in the `self.spectra` list.

        Raises
        ------
        AttributeError
            If any CIF instance does not have a `cif_PSD` attribute.
        """
        self.spectra = []
        for cif in self.cifs:
            self.spectra.append(cif.cif_PSD)

    def _compute_cross_spectra_gaussian(self):
        """
        Compute the cross-spectral density matrices for dependent Gaussian processes.

        This method calculates the cross-spectra between all pairs of Gaussian CIFs
        based on the provided weights. The resulting cross-spectra are stored in
        `self.cross_spectra` as a 3D NumPy array with shape (num_processes, num_processes, num_frequencies).

        Raises
        ------
        ValueError
            - If `weights` are not provided when `dependence` is set to "dependent".
            - If the length of `weights` does not match `num_processes`.
        """

        if self.weights is None:
            raise ValueError("Weights must be provided for dependent Gaussian CIFs.")

        weights = np.array(self.weights, dtype=complex)
        if weights.shape[0] != self.num_processes:
            raise ValueError("Weights array must have length equal to num_processes.")

        # Assuming all processes share the same frequency axis and PSD
        frequencies = self.cifs[0].cif_frequencies
        shared_PSD = self.cifs[0].cif_PSD

        # Compute the cross-spectral density matrix
        num_freqs = len(frequencies)
        self.cross_spectra = np.zeros(
            (self.num_processes, self.num_processes, num_freqs), dtype=complex
        )

        for i in range(self.num_processes):
            for j in range(self.num_processes):
                self.cross_spectra[i, j, :] = shared_PSD * (
                    weights[i] * np.conj(weights[j])
                )

    def _simulate_time_domain_multi(self):
        """
        Simulate the time-domain processes for all subprocesses.

        Depending on the `dependence` attribute, this method handles simulations for
        independent or dependent subprocesses.

        - For "independent" dependence:
            Each CIF is simulated independently, and the resulting time series are
            stored in `self.time_series` with shape (num_processes, N, Nsims).

        - For "dependent" dependence:
            Generates a shared underlying Gaussian process and applies the specified
            weights to obtain dependent subprocesses. The time series are stored in
            `self.time_series` with shape (num_processes, N, Nsims).

        After simulation, a time domain object is created containing the simulated
        intensity realizations.

        Raises
        ------
        ValueError
            - If `dependence` is set to an unsupported type.
            - If `dependence` is "dependent" but `weights` are not provided or their length
              does not match `num_processes`.
        """
        if self.dependence == "independent":
            self.time_series = []
            for cif in self.cifs:
                # Update simulate parameters
                cif.simulate = True
                cif.T = self.T
                cif.Nsims = self.Nsims
                cif.N = int(self.T * cif.fs)
                cif.time_axis = np.linspace(0, self.T, cif.N, endpoint=False)

                # Simulate the time-domain intensity
                cif_timedomain = cif._simulate_time_domain()

                # Store the intensity and create time domain object
                cif.time_domain = create_time_domain(
                    cif.time_axis, intensity_realization=cif_timedomain
                )
                self.time_series.append(cif_timedomain)

            # Stack time series to form an array of shape (num_processes, N, Nsims)
            self.time_series = np.array(self.time_series)
            self.time_axis = self.cifs[0].time_axis

        elif self.dependence == "dependent":
            # -- Shared Gaussian process, then apply weights ---------------------
            if self.weights is None:
                raise ValueError(
                    "Weights must be provided for dependent Gaussian CIFs."
                )
            weights = np.array(self.weights, dtype=complex)
            if weights.shape[0] != self.num_processes:
                raise ValueError(
                    "Weights array must have length equal to num_processes."
                )

            # Make sure each CIF has the correct simulation parameters
            for cif in self.cifs:
                cif.simulate = True
                cif.T = self.T
                cif.Nsims = self.Nsims
                cif.N = int(self.T * cif.fs)
                cif.time_axis = np.linspace(0, self.T, cif.N, endpoint=False)

            # Generate one underlying Gaussian freq. process (from the 1st CIF)
            M = len(self.cifs[0].PSD)
            U_freqdomain = self.cifs[0]._compute_U_freqdomain(self.cifs[0].PSD, M)

            # Initialize time_series array
            N = int(self.T * self.fs)
            self.time_series = np.zeros((self.num_processes, N, self.Nsims))

            # For each CIF, multiply the underlying process by its weight
            for i, cif in enumerate(self.cifs):
                W_i = weights[i]
                V_i_freqdomain = W_i * U_freqdomain  # shape: (M, Nsims)

                # IFFT along axis=0 to get time domain
                Y_i = np.real_if_close(
                    np.sqrt(self.fs * M)
                    * ifft(np.fft.ifftshift(V_i_freqdomain, axes=0), axis=0)
                )
                # Extract first N samples & add baseline lambda_0
                cif_i = Y_i[:N, :] + cif.lambda_0
                cif_i = np.real_if_close(cif_i)
                # Enforce non-negative rate
                cif_i[cif_i < 0] = 0

                # Store intensity into self.time_series
                self.time_series[i, :, :] = cif_i

                # Update the CIF's own time_domain
                cif.time_domain = create_time_domain(
                    cif.time_axis, intensity_realization=cif_i
                )

            self.time_axis = self.cifs[0].time_axis

        else:
            raise ValueError(f"Unknown dependence type: {self.dependence}")

    def get_spectra(self):
        """Get the spectra for each subprocess at all frequencies."""
        return self.spectra

    def get_total_spectrum(self):
        """Get the total spectrum (sum of all subprocess spectra)."""
        return np.sum(self.spectra, axis=0)

    def get_cross_spectra(self):
        """Get the cross-spectra for dependent processes at all frequencies."""
        if self.dependence != "dependent":
            raise ValueError(
                "Cross-spectra are only available for dependent processes."
            )
        return self.cross_spectra

    def get_time_series(self):
        """Get the simulated time series for each subprocess."""
        if not self.simulate:
            raise ValueError(
                "Time series not available. Set simulate=True to generate simulations."
            )
        return self.time_series

    def get_time_axis(self):
        """Get the time axis for simulations."""
        if not self.simulate:
            raise ValueError(
                "Time axis not available. Set simulate=True to generate simulations."
            )
        return self.time_axis

    def get_frequencies(self):
        """Get the frequency axis."""
        return self.frequencies

    @property
    def PSD(self):
        """Get the PSD of the CIF."""
        return self.spectra

    def _check_cif_parameters_consistency(self):
        """Check that all CIFs have identical spectral parameters when dependent."""
        # For Gaussian CIFs, compare peak_height, center_frequency, and peak_width
        first_cif_params = {
            "peak_height": self.cifs[0].peak_height,
            "center_frequency": self.cifs[0].center_frequency,
            "peak_width": self.cifs[0].peak_width,
        }
        for i, cif in enumerate(self.cifs[1:], start=1):
            current_cif_params = {
                "peak_height": cif.peak_height,
                "center_frequency": cif.center_frequency,
                "peak_width": cif.peak_width,
            }
            if current_cif_params != first_cif_params:
                raise ValueError(
                    f"Inconsistent CIF parameters detected for dependent processes.\n"
                    f"Process 0 parameters: {first_cif_params}\n"
                    f"Process {i} parameters: {current_cif_params}\n\n"
                    f"All processes must have identical spectral parameters (peak_height, center_frequency, peak_width) "
                    f"when 'dependence' is set to 'dependent'.\n"
                    f"Please adjust your model parameters to ensure consistency."
                )
