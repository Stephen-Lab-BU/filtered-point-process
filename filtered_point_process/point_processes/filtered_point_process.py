# filtered_point_process.py
import numpy as np
from scipy.interpolate import interp1d
from filtered_point_process.point_processes.filters import Filter
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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

        # Final outputs (filled later)
        self.final_time_series_per_process = None
        self.final_spectrum_per_process = None
        self.final_time_series = None
        self.final_spectrum = None

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

    def continuous_convolution_legacy(self, spike_times, kernel, kernel_time, out_time):
        """Continuous convolution of spike train with kernel (slow but exact)."""
        k_interp = interp1d(kernel_time, kernel, "cubic", bounds_error=False, fill_value=0.0)
        out = np.zeros_like(out_time)
        for s in spike_times:
            out += k_interp(out_time - s)
        return out

    def continuous_convolution_expensive(self, spike_times, kernel, kernel_time, out_time):
        k_interp = interp1d(kernel_time, kernel, "cubic",
                            bounds_error=False, fill_value=0.0)
        # build a (T × S) matrix of time‐differences
        #   T = len(out_time),  S = len(spike_times)
        diffs = out_time[:, None] - spike_times[None, :]
        # interpolate all at once → shape (T, S), then sum over spikes
        return np.sum(k_interp(diffs), axis=1)


    def continuous_convolution_chunked_legacy(self, spike_times, kernel, kernel_time, out_time, max_mem_bytes=200_000_000):
        """
        Continuous convolution via chunked broadcasting to cap peak memory.
        """
        T   = len(out_time)
        S   = len(spike_times)
        est = T * S * 8  # bytes needed if you did it in one go
        print(f"Full broadcast would need {est/1e9:.1f} GB; chunk cap is 200 MB")
        k_interp = interp1d(kernel_time, kernel, "cubic", bounds_error=False, fill_value=0.0)
        T = len(out_time)
        S = len(spike_times)
        # each double is 8 bytes
        mem_needed = T * S * 8
        if mem_needed <= max_mem_bytes:
            # small enough to do in one go
            diffs = out_time[:, None] - spike_times[None, :]
            return k_interp(diffs).sum(axis=1)
        # otherwise, do it in chunks
        out = np.zeros(T, dtype=float)
        # compute chunk size so chunk_size * S * 8 <= max_mem_bytes
        chunk_size = max(int(max_mem_bytes / (S * 8)), 1)
        for start in range(0, T, chunk_size):
            stop = min(T, start + chunk_size)
            diffs = out_time[start:stop][:, None] - spike_times[None, :]
            out[start:stop] = k_interp(diffs).sum(axis=1)
        return out

    def continuous_convolution_chunked_test(self, spike_times, kernel, kernel_time, out_time, max_mem_bytes=1_000_000_000):
        """
        Continuous convolution via chunked broadcasting to cap peak memory, parallelized over chunks.
        """
        T   = len(out_time)
        S   = len(spike_times)
        est = T * S * 8
        print(f"Full broadcast would need {est/1e9:.1f} GB; chunk cap is {max_mem_bytes/1e6:.0f} MB")
        k_interp = interp1d(kernel_time, kernel, "cubic", bounds_error=False, fill_value=0.0)

        mem_needed = est
        if mem_needed <= max_mem_bytes:
            diffs = out_time[:, None] - spike_times[None, :]
            return k_interp(diffs).sum(axis=1)

        # compute chunk boundaries
        chunk_size = max(int(max_mem_bytes / (S * 8)), 1)
        ranges = [(start, min(start + chunk_size, T)) for start in range(0, T, chunk_size)]

        out = np.zeros(T, dtype=float)

        def compute_chunk(r):
            s, e = r
            diffs = out_time[s:e][:, None] - spike_times[None, :]
            return s, k_interp(diffs).sum(axis=1)

        # parallel map
        with ThreadPoolExecutor() as exe:
            futures = [exe.submit(compute_chunk, r) for r in ranges]
            for fut in as_completed(futures):
                s, vals = fut.result()
                out[s : s + vals.shape[0]] = vals

        return out

    def continuous_convolution_chunked(self, spike_times, kernel, kernel_time, out_time, max_mem_bytes=1_000_000_000):
        T, S = len(out_time), len(spike_times)
        est = T * S * 8
        kernel_time = np.ascontiguousarray(np.real(kernel_time), dtype=float)
        kernel      = np.ascontiguousarray(np.real(kernel), dtype=float)
        kx, ky = kernel_time, kernel
        if np.iscomplexobj(ky):
            ky = np.real(ky)
        # if it fits, do it all at once
        if est <= max_mem_bytes:
            diffs = out_time[:,None] - spike_times[None,:]               # shape (T,S)
            flat  = diffs.ravel()
            flat = flat.real.astype(float, copy=False)
            #vals  = np.interp(flat, kx, ky, left=0.0, right=0.0)       # 1D C loop
            vals = np.interp(flat, np.real(kx), np.real(ky), left=0.0, right=0.0)
            return vals.reshape(T, S).sum(axis=1)
        # otherwise chunk exactly like before, but still with np.interp
        out = np.zeros(T, float)
        chunk_size = max(int(max_mem_bytes / (S*8)), 1)
        for start in range(0, T, chunk_size):
            stop  = min(T, start+chunk_size)
            diffs = out_time[start:stop][:,None] - spike_times[None,:]
            flat  = diffs.ravel()
            flat = flat.real.astype(float, copy=False)
            #vals  = np.interp(flat, kx, ky, left=0.0, right=0.0)
            vals = np.interp(flat, np.real(kx), np.real(ky), left=0.0, right=0.0)
            out[start:stop] = vals.reshape(stop-start, S).sum(axis=1)
        return out

    
    def _build_multivariate_sequences(self, filter_sequences):
        """
        If we have multiple processes, parse the user-provided filter sequences and re-map them
        so that each process gets a single non-spectral filter plus (optionally) a single spectral filter.

        Example:
            If user passes one list like: [["filter_1", "filter_2", "filter_3"]]
            and internally filter_1->GABA, filter_2->AMPA, filter_3->"1/f",
            we detect:
                non-spectral = [filter_1, filter_2]
                spectral = [filter_3]
            If num_processes=2, we form:
                process1: [filter_1, filter_3]
                process2: [filter_2, filter_3]

            If only one non-spectral filter is provided (e.g., "filter_1"->AMPA) for 2 processes,
            both processes get that filter, plus the same spectral filter if present.

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

        # Sanity checks (assuming at most 1 spectral filter is typical, though can handle more if you like)
        if len(spectral_filters) > 1:
            raise ValueError(
                f"Multiple spectral filters found ({spectral_filters}). "
                "Please provide at most one spectral filter, or adjust logic."
            )

        # We can replicate a single non-spectral filter for all processes,
        # or else we must have exactly as many non-spectral filters as processes
        if len(non_spectral_filters) == self.num_processes:
            pass  # Perfect match
        elif len(non_spectral_filters) == 1 and self.num_processes > 1:
            # Use the same filter for each process
            non_spectral_filters = non_spectral_filters * self.num_processes
        else:
            raise ValueError(
                f"Number of non-spectral filters ({len(non_spectral_filters)}) must be 1 or match "
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
        import warnings
        import time

        t0 = time.perf_counter()

        # 1) multivariate remapping
        if self.num_processes > 1:
            filter_sequences = self._build_multivariate_sequences(filter_sequences)

        # record which filters were applied
        self.applied_sequences = [
            [(f, self.filters.filter_types[f]) for f in seq]
            for seq in filter_sequences
        ]

        new_time_series = []
        new_auto        = []
        complex_kernels = []
        dt = (self.time_axis[1] - self.time_axis[0]) if self.has_time_domain else 1/self.fs

        # ─── override each filter’s time‐vector to its effective support ─────────
        if self.has_time_domain:
            eps = self.filter_params.get("support_eps", 1e-3)
            warnings.warn(
                "Effective support of time-domain filter kernels was automatically updated "
                "based on a decay threshold of ε = {:.1e}. This overrides the default or user-specified "
                "`filter_time_vector` values for AMPA, GABA, and 1/f filters. If precise temporal support is "
                "critical (e.g., for short-duration kernels or fixed-lag analyses), manually specify "
                "`filter_time_vector` or adjust `support_eps` in `filter_params`.".format(eps),
                UserWarning
            )
            for fname, inst in self.filter_instances.items():
                ftype = self.filters.filter_types[fname]
                if ftype in ("AMPA", "GABA"):
                    tau = inst.filter_params["tau_decay"]
                elif ftype == "1/f":
                    A   = inst.filter_params.get("A", 1/0.1)
                    tau = 1.0 / A
                else:
                    continue
                T_eff = -tau * np.log(eps)
                inst.filter_params["filter_time_vector"] = np.arange(0, T_eff + dt, dt)
                inst.compute_filter()
        t1 = time.perf_counter()
        print(f"[timing] recompute filters took {t1-t0:.2f}s")
        # ─────────────────────────────────────────────────────────────────────────────

        # 2) loop processes
        for i, seq in enumerate(filter_sequences):
            # grab the spike train
            if self.has_time_domain:
                spike_times = (
                    self.model.spikes[0][i] if self.num_processes > 1 else self.model.spikes
                )

            # collect all Filter instances for this chain
            insts = [self.filter_instances[name] for name in seq]

            # build the total frequency-domain power filter: ∏ |H|²
            total_filter_pspec = np.ones_like(self.frequencies)
            for inst in insts:
                total_filter_pspec *= inst.kernel_spectrum

            # build the complex kernel density: ∏ H(f)
            K_i = np.ones_like(self.frequencies, dtype=complex)
            for inst in insts:
                K_i *= inst.kernel_density_not_squared
            complex_kernels.append(K_i)

            # time-domain convolution
            if self.has_time_domain:
                first_type  = self.filters.filter_types[seq[0]]
                second_type = self.filters.filter_types[seq[1]] if len(insts) > 1 else None

                if len(insts)==2 and second_type=="1/f" and first_type in ("AMPA","GABA"):
                    # 1) pull parameters
                    tau_r  = insts[0].filter_params["tau_rise"]
                    tau_d  = insts[0].filter_params["tau_decay"]
                    A      = insts[1].filter_params.get("A", 1/0.1)

                    # 2) effective supports
                    eps    = 1e-12
                    dt     = self.time_axis[1] - self.time_axis[0]
                    tau_slowest = max(tau_d, tau_r, 1.0/A)
                    Tcomb = -tau_slowest * np.log(eps)
                    #T1     = -tau_d   * np.log(eps)
                    #T2     = -(1.0/A) * np.log(eps)
                    #Tcomb  = T1 + T2

                    # 3) analytic kernel on [0…Tcomb]
                    t_axis = np.arange(0, Tcomb + dt, dt)
                    h_comb  = (
                        (np.exp(-t_axis/tau_d) - np.exp(-A*t_axis)) / (A - 1/tau_d)
                    - (np.exp(-t_axis/tau_r) - np.exp(-A*t_axis)) / (A - 1/tau_r)
                    )

                    # ——— DEBUG PLOT 1: kernel & FFT vs analytic product ———
                    import matplotlib.pyplot as plt, os
                    f_axis   = np.fft.rfftfreq(len(h_comb), dt)
                    H_fft    = np.fft.rfft(h_comb) * dt
                    H1       = 1/(1/tau_d   + 1j*2*np.pi*f_axis) - 1/(1/tau_r + 1j*2*np.pi*f_axis)
                    H2       = 1/(A         + 1j*2*np.pi*f_axis)
                    H_prod   = H1 * H2

                    fig, ax = plt.subplots(1,2,figsize=(10,4))
                    # FFT vs product
                    ax[0].loglog(f_axis, np.abs(H_fft)**2,    label="FFT(h_comb)")
                    ax[0].loglog(f_axis, np.abs(H_prod)**2, '--', label="Analytic |H1H2|²")
                    ax[0].set_title("Combined Kernel Spectrum")
                    ax[0].set_xlabel("Frequency (Hz)")
                    ax[0].set_ylabel("|H|²")
                    ax[0].legend(fontsize=8)

                    # kernel
                    ax[1].plot(t_axis, h_comb, lw=1.5)
                    ax[1].set_title("Analytic h_comb(t)")
                    ax[1].set_xlabel("Time (s)")
                    ax[1].set_ylabel("Amplitude")
                    
                    output_dir = "/Users/patrick_bloniasz/filtered-point-process/examples/demo_cutoff_lambda/"
                    plt.tight_layout()
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        fig.savefig(os.path.join(output_dir, "analytic_kernel_and_fft.png"), dpi=300)
                    plt.close(fig)
                    # ——————————————————————————————————————————————

                    # 4) pad and convolve so no spike‐tail is lost
                    pad    = Tcomb
                    t_full = np.arange(
                        self.time_axis[0] - pad,
                        self.time_axis[-1] + pad + dt,
                        dt
                    )
                    #filtered_full = self.continuous_convolution(spike_times, h_comb, t_axis, t_full)
                    t2 = time.perf_counter()
                    filtered_full = self.continuous_convolution_chunked(
                        spike_times, h_comb, t_axis, t_full,
                        max_mem_bytes=500_000_000  # for example 200 MB cap
                    )
                    t3 = time.perf_counter()
                    print(f"[timing] continuous convolution took {t3-t2:.2f}s")

                    start          = int(np.round(pad / dt))
                    filtered_train = filtered_full[start : start + self.N]
                    t4 = time.perf_counter()
                    print(f"[timing] apply_filter_sequences total: {t4-t0:.2f}s")

                else:
                    # numeric cascade + continuous convolution
                    kern   = insts[0].kernel
                    t_axis = insts[0].kernel_time_axis
                    for inst in insts[1:]:
                        kern = fftconvolve(kern, inst.kernel, mode="full") * dt
                        t_axis = np.linspace(
                            t_axis[0] + inst.kernel_time_axis[0],
                            t_axis[-1] + inst.kernel_time_axis[-1],
                            len(kern),
                        )
                    filtered_train = self.continuous_convolution_chunked(
                        spike_times, kern, t_axis, self.time_axis
                    )
            else:
                filtered_train = None

            new_time_series.append(filtered_train)

            # auto‐spectra = (λ0 + CIF) ⋅ |H|²
            λ0    = self.decompositions[i]["lambda_only"]
            S_cif = self.decompositions[i]["cif"]
            new_auto.append((λ0 + S_cif) * total_filter_pspec)

        # 3) stash per-process results
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
            X_raw = self.model.cif.get_cross_spectra()
            P, F  = self.num_processes, len(self.frequencies)
            K     = np.array(complex_kernels)

            X_filt = np.zeros((P, P, F), dtype=complex)
            for ii in range(P):
                for jj in range(P):
                    X_filt[ii, jj] = K[ii] * np.conj(K[jj]) * X_raw[ii, jj]
            # sum diagonal & off-diagonals
            all_filt    = np.sum(X_filt, axis=(0, 1))
            diag_only   = np.trace(X_filt, axis1=0, axis2=1)
            cross_terms = all_filt - diag_only
            auto_filt   = np.sum(self.final_spectrum_per_process, axis=0)
            self.final_spectrum = (auto_filt + cross_terms).real
        else:
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
