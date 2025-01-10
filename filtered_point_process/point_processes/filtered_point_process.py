import numpy as np
from scipy.interpolate import interp1d
from filtered_point_process.point_processes.filters import Filter

class FilteredPointProcess:
    """Class to interact with the filtered point process.

    This class provides functionalities to apply a sequence of filters to a point process model.
    It supports both univariate and multivariate models, handling spectra and continuously-simulated
    spike times convolved continuously prior to being sampled.

    Attributes:
        model (Model): The point process model.
        fs (float): Sampling rate or reference frequency extracted from the model.
        time_axis (np.ndarray): Continuous time axis for plotting and evaluation.
        frequencies (np.ndarray): Frequency axis corresponding to the model.
        N (int): Number of time points in the time axis.
        filter_params (dict): Parameters for each filter.
        filters (Filter): Instance managing the filter configurations.
        filter_instances (dict): Individual filter instances managed by the Filter class.
        num_processes (int): Number of processes in the model.
        decompositions (list): Frequency-domain decompositions for each process.
        final_time_series_per_process (np.ndarray): Continuous convolved output for each process.
        final_spectrum_per_process (np.ndarray): Frequency-domain results for each process.
        final_time_series (np.ndarray): Sum of time series across all processes.
        final_spectrum (np.ndarray): Sum of spectra across all processes.
    """

    def __init__(self, filters=None, model=None, filter_params=None):
        """
        Initializes the FilteredPointProcess with a point process model and optional filters.

        Args:
            filters (dict, optional):
                A dictionary mapping filter names to filter types.
            model (Model, optional):
                An instance of the Model class containing necessary attributes:
                    - cif.fs (float): Sampling rate or reference frequency.
                    - time_axis (np.ndarray): Continuous time axis for plotting/evaluation.
                    - frequencies (np.ndarray): Frequency axis.
                    - pp (PointProcess): The point-process object.
                    - spikes (list or np.ndarray): Spike times, not discretized.
            filter_params (dict, optional):
                A dictionary of parameters for each filter.
        """
        self.model = model
        self.fs = self.model.cif.fs
        self.time_axis = self.model.time_axis
        self.frequencies = self.model.frequencies
        self.N = len(self.time_axis)
        self.filter_params = filter_params if filter_params is not None else {}

        # Initialize filters
        self.filters = Filter(
            filters=filters, model=self.model.pp, filter_params=self.filter_params
        )
        self.filter_instances = self.filters.filter_instances

        # Determine if univariate or multivariate
        self.num_processes = 1
        if hasattr(self.model.cif, "cifs"):
            self.num_processes = len(self.model.cif.cifs)

        # Create frequency-domain decompositions for each process
        self.decompositions = [
            self._compute_initial_decomposition(i) for i in range(self.num_processes)
        ]

        self.final_time_series_per_process = None
        self.final_spectrum_per_process = None
        self.final_time_series = None
        self.final_spectrum = None

    def _compute_initial_decomposition(self, process_idx):
        """
        Compute the initial frequency-domain decomposition for a specified process.

        This method initializes the decomposition with 'lambda_only' and 'cif' in the frequency domain.
        """
        if self.num_processes == 1:
            lambda_0 = self.model.cif.lambda_0
            cif_spectrum = self.model.cif.PSD
        else:
            lambda_0 = self.model.cif.cifs[process_idx].lambda_0
            cif_spectrum = self.model.cif.PSD[process_idx]

        return {
            "lambda_only": lambda_0 * np.ones_like(self.frequencies),
            "cif": cif_spectrum.copy(),
        }

    def continuous_convolution(self, spike_times, kernel, kernel_time, out_time):
        """
        Convolve spike times with a kernel in a continuous manner.
        For each spike time, we shift 'kernel' and add via interpolation.
        """
        k_interp = interp1d(
            kernel_time, kernel, kind="cubic", bounds_error=False, fill_value=0.0
        )
        out = np.zeros_like(out_time)
        for s in spike_times:
            out += k_interp(out_time - s)
        return out


    def _combine_two_filters_continuous(self, filter1_name, filter2_name):
        """
        Combine exactly two filters by convolving them (shift-and-sum)
        on the *same dt* as filter1, returning a single combined kernel
        and a matching time axis. This helps avoid leakage from time-axis
        mismatches.

        The math is approximating:
            (k1 * k2)(t) = ∫ k1(τ) k2(t - τ) dτ
        in a discrete Riemann-sum form, so we multiply by 'dt' at the end.
        """
        import numpy as np
        from scipy.interpolate import interp1d

        # Get the two filters
        f1 = self.filter_instances[filter1_name]
        f2 = self.filter_instances[filter2_name]
        t1, k1 = f1.kernel_time_axis, f1.kernel
        t2, k2 = f2.kernel_time_axis, f2.kernel

        # Use filter1's sampling interval as the "base" dt
        dt = t1[1] - t1[0]
        # Maximum time of the combined kernel is roughly t1[-1] + t2[-1]
        max_time = t1[-1] + t2[-1]

        # Build a unified time axis with the same dt
        combined_time = np.arange(0, max_time + dt/2, dt)

        # We'll interpolate filter2 onto that same axis when shifting
        k2_interp = interp1d(t2, k2, kind='linear', bounds_error=False, fill_value=0.0)

        # Shift-and-sum approximation of the convolution integral
        out = np.zeros_like(combined_time)
        for i, tau_val in enumerate(t1):
            amp = k1[i]
            # Evaluate k2 at (combined_time - tau_val)
            out += amp * k2_interp(combined_time - tau_val)

        # Multiply by dt to approximate the ∫ in continuous time
        out *= dt

        return out, combined_time


    def apply_filter_sequences(self, filter_sequences):
        """
        Applies a specified sequence of filters to each process in both frequency
        and time domains. For each sequence:
            - If there's 1 filter, use it directly.
            - If there are 2 filters, combine them into one via continuous convolution.
            - Otherwise, raise an error (no assumptions beyond 1 or 2 filters).

        Then convolve the final single kernel with the spike times in a continuous manner.
        """
        if self.num_processes == 1 and not isinstance(filter_sequences[0], list):
            filter_sequences = [filter_sequences]

        new_time_series = []
        new_spectra = []

        for i, seq in enumerate(filter_sequences):
            if len(seq) == 1:
                # Single filter
                f_name = seq[0]
                f_inst = self.filter_instances[f_name]
                combined_kernel = f_inst.kernel
                combined_kernel_time = f_inst.kernel_time_axis
                total_filter_spectrum = f_inst.kernel_spectrum

            elif len(seq) == 2:
                # Two filters -> combine once
                f1, f2 = seq[0], seq[1]
                combined_kernel, combined_kernel_time = self._combine_two_filters_continuous(f1, f2)
                # Multiply their spectra
                total_filter_spectrum = (
                    self.filter_instances[f1].kernel_spectrum
                    * self.filter_instances[f2].kernel_spectrum
                )
            else:
                raise ValueError(
                    "apply_filter_sequences only handles sequences of length 1 or 2. "
                    f"Got {len(seq)} filters in {seq}."
                )

            # Convolve final kernel with spike times
            if self.num_processes > 1:
                spike_times = self.model.spikes[0][i]
            else:
                spike_times = self.model.spikes

            filtered_train = self.continuous_convolution(
                spike_times, combined_kernel, combined_kernel_time, self.time_axis
            )

            # Frequency-domain result
            lambda_only_filtered = (
                self.decompositions[i]["lambda_only"] * total_filter_spectrum
            )
            cif_filtered = self.decompositions[i]["cif"] * total_filter_spectrum

            new_time_series.append(filtered_train)
            new_spectra.append(lambda_only_filtered + cif_filtered)

        self.final_time_series_per_process = np.array(new_time_series)
        self.final_spectrum_per_process = np.array(new_spectra)

        self.final_time_series = np.sum(self.final_time_series_per_process, axis=0)
        self.final_spectrum = np.sum(self.final_spectrum_per_process, axis=0)

    def get_final_spectrum(self, decomposition=True):
        """
        Retrieve the final spectrum, time series, and optionally the decomposition.

        Args:
            decomposition (bool): Whether to include the detailed process_decompositions.

        Returns:
            dict: {
                "final_total_spectrum": np.ndarray,
                "final_time_series": np.ndarray,
                "process_decompositions": list (if decomposition=True)
            }
        """
        result = {
            "final_total_spectrum": self.final_spectrum,
            "final_time_series": self.final_time_series,
        }
        if decomposition:
            result["process_decompositions"] = self.decompositions
        return result

    def get_combined_kernel(self, filter_name1, filter_name2):
        """
        Explicitly retrieve and visualize the combined kernel of two filters.

        TESTING ONLY JAN 10

        Args:
            filter_name1 (str): Name of the first filter (e.g., "GABA").
            filter_name2 (str): Name of the second filter (e.g., "1/f").

        Returns:
            tuple:
                combined_kernel (np.ndarray): Combined kernel values.
                combined_time_axis (np.ndarray): Time axis for the combined kernel.
        """
        combined_kernel, combined_time_axis = self._combine_two_filters_continuous(filter_name1, filter_name2)
        return combined_kernel, combined_time_axis

