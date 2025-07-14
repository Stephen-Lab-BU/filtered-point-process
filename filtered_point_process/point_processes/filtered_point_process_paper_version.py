# filtered_point_process.py
import numpy as np
from scipy.interpolate import interp1d
from filtered_point_process.point_processes.filters import Filter
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import os

VALID_FILTER_TYPES = ["AMPA", "GABA", "Fast_AP", "Slow_AP", "1/f", "Lorenzian"]


class FilteredPointProcess:
    """
    Wrapper for analysing a point-process model with (optional) filters in both
    the time- and frequency-domains.  If the upstream Model was built with
    ``simulation_params["simulate"] == False`` we **skip** all continuous
    convolutions and only carry out frequency-domain operations.
    """

    def __init__(self, filters=None, model=None, filter_params=None):
        self.model = model
        self.fs = self.model.cif.fs

        self.has_time_domain = bool(self.model.simulation_params.get("simulate", False))

        if self.has_time_domain:
            self.time_axis = self.model.time_axis          # (N,)
            self.N = len(self.time_axis)
        else:
            self.time_axis, self.N = None, None

        self.frequencies = self.model.frequencies
        self.filter_params = filter_params or {}


        self.filters = Filter(
            filters=filters, model=self.model.pp, filter_params=self.filter_params
        )
        self.filter_instances = self.filters.filter_instances

        # Univariate vs multivariate
        self.num_processes = (
            len(self.model.cif.cifs) if hasattr(self.model.cif, "cifs") else 1
        )

        # Decomposition for each process
        self.decompositions = [
            self._compute_initial_decomposition(i) for i in range(self.num_processes)
        ]

        # Allocate space for now
        self.final_time_series_per_process = None
        self.final_spectrum_per_process = None
        self.final_time_series = None
        self.final_spectrum = None

\
    def _compute_initial_decomposition(self, idx):
        """Start with λ₀ and CIF-PSD for process *idx* (frequency-domain only)."""
        if self.num_processes == 1:
            lambda_0 = self.model.cif.lambda_0
            cif_psd = self.model.cif.PSD
        else:
            lambda_0 = self.model.cif.cifs[idx].lambda_0
            cif_psd = self.model.cif.PSD[idx]

        return {
            "lambda_only": lambda_0 * np.ones_like(self.frequencies),
            "cif": cif_psd.copy(),
        }

    def continuous_convolution(self, spike_times, kernel, kernel_time, out_time):
        """Continuous convolution of spike train with kernel (slow but exact)."""
        k_interp = interp1d(kernel_time, kernel, "cubic", bounds_error=False, fill_value=0.0)
        out = np.zeros_like(out_time)
        for s in spike_times:
            out += k_interp(out_time - s)
        return out

    def _build_multivariate_sequences(self, filter_sequences):
        """
        If we have multiple processes, parse the user-provided filter sequences and re-map them
        so that each process gets a single non-1_over_f filter plus (optionally) a single 1_over_f filter. Here,
        I'm using 

        Example:
            If user passes one list like: [["filter_1", "filter_2", "filter_3"]]
            and internally filter_1->GABA, filter_2->AMPA, filter_3->"1/f",
            we detect:
                non-spectral = [filter_1, filter_2]
                spectral = [filter_3]
            If num_processes=2, we form:
                process1: [filter_1, filter_3]
                process2: [filter_2, filter_3]

            If only one non-1_over_f filter is provided (e.g., "filter_1"->AMPA) for 2 processes,
            both processes get that filter, plus the same 1_over_f filter if present.

        Returns:
            final_sequences (list of lists):
                A list of sequences, one for each process.
                Each sequence is either [non_spectral_filter_i] or [non_spectral_filter_i, spectral_filter].
        """
        # Flatten whatever the user gave into a single list of filter names:
        flattened = []
        for seq in filter_sequences:
            flattened.extend(seq)

        # Separate into spectral and non-spectral
        spectral_filters = []
        non_spectral_filters = []
        for f_name in flattened:
            #f_type = self.filter_instances[f_name].filter_type
            f_type = self.filters.filter_types[f_name]
            if f_type in ["1/f", "Lorenzian"]:
                spectral_filters.append(f_name)
            else:
                non_spectral_filters.append(f_name)

        # Sanity checks (assuming at most 1 1_over_f filter is typical, though can handle more if you like)
        if len(spectral_filters) > 1:
            raise ValueError(
                f"Multiple 1_over_f filters found ({spectral_filters}). "
                "Please provide at most one 1_over_f filter, or adjust logic."
            )

        # We can replicate a single non-1_over_f filter for all processes,
        # or else we must have exactly as many non-1_over_f filters as processes
        if len(non_spectral_filters) == self.num_processes:
            pass  # Perfect match
        elif len(non_spectral_filters) == 1 and self.num_processes > 1:
            # Use the same filter for each process
            non_spectral_filters = non_spectral_filters * self.num_processes
        else:
            raise ValueError(
                f"Number of non-1_over_f filters ({len(non_spectral_filters)}) must be 1 or match "
                f"the number of processes ({self.num_processes})."
            )

        # Build the final list of sequences
        final_sequences = []
        spectral_filter = spectral_filters[0] if len(spectral_filters) == 1 else None
        for i in range(self.num_processes):
            if spectral_filter:
                final_sequences.append([non_spectral_filters[i], spectral_filter])
            else:
                final_sequences.append([non_spectral_filters[i]])
        return final_sequences

    def apply_filter_sequences(self, filter_sequences, output_dir=None):
        """
        Apply arbitrary-length filter chains to each process,
        do time-domain conv if simulate=True, then build
        filtered cross-spectra + total field spectrum.
        """
        # 1) if multivariate, remap user input into one list-per-process
        if self.num_processes > 1:
            filter_sequences = self._build_multivariate_sequences(filter_sequences)

        # record (name,type) for each chain
        self.applied_sequences = [
            [(f, self.filters.filter_types[f]) for f in seq]
            for seq in filter_sequences
        ]

        new_time_series = []
        new_auto = []
        complex_kernels = []
        dt = (self.time_axis[1] - self.time_axis[0]) if self.has_time_domain else 1/self.fs

        # 2) loop processes
        for i, seq in enumerate(filter_sequences):
            # grab the spike train if we need it
            if self.has_time_domain:
                spike_times = (
                    self.model.spikes[0][i] if self.num_processes>1 else self.model.spikes
                )

            # collect all Filter instances for this chain
            insts = [self.filter_instances[name] for name in seq]

            total_filter_pspec = np.ones_like(self.frequencies)
            for inst in insts:
                total_filter_pspec *= inst.kernel_spectrum
            K_i = np.ones_like(self.frequencies, dtype=complex)
            for inst in insts:
                K_i *= inst.kernel_density_not_squared
            complex_kernels.append(K_i)

            # build the time-domain kernel by chaining FFT-convolutions (this causes leakage)
            if self.has_time_domain:
                kern = insts[0].kernel
                t_axis = insts[0].kernel_time_axis
                for inst in insts[1:]:
                    kern = fftconvolve(kern, inst.kernel, mode="full") * dt
                    t_axis = np.linspace(
                        t_axis[0] + inst.kernel_time_axis[0],
                        t_axis[-1] + inst.kernel_time_axis[-1],
                        len(kern),
                    )
                filtered_train = self.continuous_convolution(
                    spike_times, kern, t_axis, self.time_axis
                )
            else:
                filtered_train = None

            new_time_series.append(filtered_train)

            # auto‐spectra = (λ0 + CIF) ⋅ |H|²
            λ0 = self.decompositions[i]["lambda_only"]
            S_cif = self.decompositions[i]["cif"]
            new_auto.append((λ0 + S_cif) * total_filter_pspec)

        self.final_time_series_per_process = (
            np.array(new_time_series) if self.has_time_domain else None
        )
        if self.has_time_domain:
            self.final_time_series = np.sum(self.final_time_series_per_process, axis=0)
        else:
            self.final_time_series = None

        self.final_spectrum_per_process = np.array(new_auto)

        # 4) build filtered cross-spectra + total field spectrum
        if (
            self.num_processes > 1
            and getattr(self.model.cif, "dependence", "independent") == "dependent"
        ):
            X_raw = self.model.cif.get_cross_spectra()     # (P,P,F)
            P, F = self.num_processes, len(self.frequencies)
            K    = np.array(complex_kernels)               # (P,F) complex

            X_filt = np.zeros((P, P, F), dtype=complex)
            for ii in range(P):
                for jj in range(P):
                    X_filt[ii, jj] = K[ii] * np.conj(K[jj]) * X_raw[ii, jj]
            self.filtered_cross_spectra = X_filt
            for ii in range(P):
                for jj in range(P):
                    X_filt[ii, jj] = K[ii] * np.conj(K[jj]) * X_raw[ii, jj]

            self.filtered_cross_spectra = X_filt
            
            
            # 1) sum over i,j → includes diag & off-diag
            all_filt  = np.sum(X_filt, axis=(0,1))            # (F,)

            # 2) extract just the diagonal (|H|²·S_cif part)
            diag_only = np.trace(X_filt, axis1=0, axis2=1)    # (F,)

            # 3) cross‐terms = total minus diagonal
            cross_terms = all_filt - diag_only                # (F,)

            # 4) auto terms already in final_spectrum_per_process
            auto_filtered = np.sum(self.final_spectrum_per_process, axis=0)  # (F,)

            # 5) final = auto (|H|²(λ₀+S_cif)) + cross (K_iK_j* S_ij)
            self.final_spectrum = (auto_filtered + cross_terms).real        # (F,)

        else:
            # univariate fallback
            self.filtered_cross_spectra = None
            self.final_spectrum = np.sum(self.final_spectrum_per_process, axis=0).real


    def _plot_two_approaches(
        self,
        approach1,
        approach2,
        label1="Approach1",
        label2="Approach2",
        output_dir=None
    ):
        """
        Plot the two approaches for debugging/comparison and also plot their difference/error.
        """
        error = approach1 - approach2

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(self.time_axis, approach1, label=label1, linewidth=2)
        ax1.plot(self.time_axis, approach2, label=label2, linestyle='--', alpha=0.7)
        ax1.set_title("Comparison of Two Convolution Approaches")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Filtered Signal")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(self.time_axis, error, label="Difference (Approach1 - Approach2)",
                 linewidth=1.5)
        ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax2.set_title("Error Between the Two Approaches")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Error")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        fig.tight_layout()

        if output_dir is not None:
            filename = "two_approaches_comparison.png"
            filepath = os.path.join(output_dir, filename)
            print(f"Saving comparison plot to: {filepath}")
            fig.savefig(filepath)

        plt.close(fig)

    def _compare_two_approaches_spectrum(
        self,
        approach1_signal,
        approach2_signal,
        label1="Approach1",
        label2="Approach2",
        output_dir=None
    ):
        """
        Compute and plot the multitaper PSDs of two approaches, then plot their difference.
        """
        from spectral_connectivity import Multitaper, Connectivity
        
        def compute_multitaper_psd(signal, fs):
            mt = Multitaper(signal, sampling_frequency=fs, n_tapers=5, start_time=0.0)
            conn = Connectivity.from_multitaper(mt)
            freqs = conn.frequencies
            psd = conn.power().squeeze()
            return freqs, psd

        freqs1, psd1 = compute_multitaper_psd(approach1_signal, self.fs)
        freqs2, psd2 = compute_multitaper_psd(approach2_signal, self.fs)
        psd_diff = psd1 - psd2

        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.loglog(freqs1, psd1, label=label1, linewidth=2)
        ax1.loglog(freqs2, psd2, label=label2, linestyle='--', alpha=0.7)
        ax1.set_title("Multitaper PSD Comparison of Two Convolution Approaches")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("PSD")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.semilogx(freqs1, psd_diff, linewidth=1.5, label="PSD Difference")
        ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax2.set_title("Difference (Approach1 PSD - Approach2 PSD)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("PSD Diff")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        fig.tight_layout()

        if output_dir is not None:
            filename = "two_approaches_psd_comparison.png"
            filepath = os.path.join(output_dir, filename)
            print(f"Saving PSD comparison plot to: {filepath}")
            fig.savefig(filepath)

        plt.close(fig)

    def get_final_spectrum(self, decomposition=True):
        """Return final PSD (always) plus time series (if any)."""
        out = {
            "final_total_spectrum": self.final_spectrum,
            "final_time_series": self.final_time_series,
        }
        if decomposition:
            out["process_decompositions"] = self.decompositions
        return out

    def get_combined_kernel(self, filter_name1, filter_name2):
        """Return combined kernel/time axis for two filters (requires time axis)."""
        if not self.has_time_domain:
            raise RuntimeError("No time axis available (model was not simulated).")

        f1, f2 = self.filter_instances[filter_name1], self.filter_instances[filter_name2]
        dt = self.time_axis[1] - self.time_axis[0]
        kernel = fftconvolve(f1.kernel, f2.kernel, mode="full") * dt
        t_axis = np.linspace(
            f1.kernel_time_axis[0] + f2.kernel_time_axis[0],
            f1.kernel_time_axis[-1] + f2.kernel_time_axis[-1],
            len(kernel),
        )
        return kernel, t_axis

    def get_applied_sequences_with_types(self):
        """Return applied_sequences with (name, type) tuples for readability."""
        if not hasattr(self, "applied_sequences"):
            return None
        typed = []
        for seq in self.applied_sequences:
            typed.append([(f, self.filters.filter_types[f]) for f in seq])
        return typed
