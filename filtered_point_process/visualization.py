import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Class to visualize the filtered point process."""

    def __init__(self, fpp_instance):
        self.fpp = fpp_instance

    def plot_all_stages(self):
        # Gather data for plotting
        filters = self.fpp.get_filters()
        convolutions = self.fpp.perform_convolutions()
        spectra = self.fpp.get_spectra()

        # Total number of rows required: CIF + Point Process + Filters + Convolutions
        num_rows_time = 2 + len(filters) + len(convolutions)
        num_rows_freq = 2 + len(filters) + len(spectra)

        fig_time, axs_time = plt.subplots(
            num_rows_time, 1, figsize=(15, num_rows_time * 8)
        )
        fig_freq, axs_freq = plt.subplots(
            num_rows_freq, 1, figsize=(15, num_rows_freq * 8)
        )

        row_time = 0
        row_freq = 0

        # CIF (Time Domain)
        if hasattr(self.fpp.pp, "cif_time_axis") and hasattr(
            self.fpp.pp, "cif_realization"
        ):
            axs_time[row_time].plot(
                self.fpp.pp.cif_time_axis, self.fpp.pp.cif_realization
            )
            axs_time[row_time].set_title("CIF (Time Domain)")
            axs_time[row_time].set_xlabel("Time (s)")
            axs_time[row_time].set_ylabel("Intensity")
            row_time += 1

        # CIF (Frequency Domain)
        if hasattr(self.fpp.pp, "cif_frequencies") and hasattr(self.fpp.pp, "cif_PSD"):
            axs_freq[row_freq].plot(self.fpp.pp.cif_frequencies, self.fpp.pp.cif_PSD)
            axs_freq[row_freq].set_title("CIF (Power Spectrum)")
            axs_freq[row_freq].set_xlabel("Frequency (Hz)")
            axs_freq[row_freq].set_ylabel("Power")
            if (
                "center_frequency" in self.fpp.pp.params
                and "peak_width" in self.fpp.pp.params
            ):
                x_min = (
                    self.fpp.pp.params["center_frequency"]
                    - 5 * self.fpp.pp.params["peak_width"]
                )
                x_max = (
                    self.fpp.pp.params["center_frequency"]
                    + 5 * self.fpp.pp.params["peak_width"]
                )
                axs_freq[row_freq].set_xlim([x_min, x_max])
            row_freq += 1

        # Point Process (Time Domain)
        if hasattr(self.fpp.pp, "pp_events"):
            axs_time[row_time].eventplot(self.fpp.pp.pp_events, colors="black")
            axs_time[row_time].set_title("Point Process (Time Domain)")
            axs_time[row_time].set_xlabel("Time (s)")
            axs_time[row_time].set_ylabel("Events")
            axs_time[row_time].spines["top"].set_visible(False)
            axs_time[row_time].spines["right"].set_visible(False)
            axs_time[row_time].spines["left"].set_visible(False)
            axs_time[row_time].spines["bottom"].set_visible(False)
            axs_time[row_time].set_xticks([])
            axs_time[row_time].set_yticks([])
            row_time += 1

        # Point Process (Frequency Domain)
        if hasattr(self.fpp.pp, "pp_frequencies") and hasattr(self.fpp.pp, "pp_PSD"):
            axs_freq[row_freq].plot(self.fpp.pp.pp_frequencies, self.fpp.pp.pp_PSD)
            axs_freq[row_freq].set_title("Point Process (Power Spectrum)")
            axs_freq[row_freq].set_xlabel("Frequency (Hz)")
            axs_freq[row_freq].set_ylabel("Power")
            if (
                "center_frequency" in self.fpp.pp.params
                and "peak_width" in self.fpp.pp.params
            ):
                x_min = (
                    self.fpp.pp.params["center_frequency"]
                    - 5 * self.fpp.pp.params["peak_width"]
                )
                x_max = (
                    self.fpp.pp.params["center_frequency"]
                    + 5 * self.fpp.pp.pp.params["peak_width"]
                )
                axs_freq[row_freq].set_xlim([x_min, x_max])
            row_freq += 1

        # Filters (Time Domain and Frequency Domain)
        for filter_name, filter_output in filters.items():
            label = self.fpp.filters[filter_name]  # Get the actual label of the filter
            if "time_axis" in filter_output and "kernel" in filter_output:
                axs_time[row_time].plot(
                    filter_output["time_axis"], filter_output["kernel"]
                )
                axs_time[row_time].set_title(f"{label} Kernel (Time Domain)")
                axs_time[row_time].set_xlabel("Time (s)")
                axs_time[row_time].set_ylabel("Kernel")
                row_time += 1

            if "frequencies" in filter_output and "power_spectrum" in filter_output:
                axs_freq[row_freq].loglog(
                    filter_output["frequencies"], filter_output["power_spectrum"]
                )
                axs_freq[row_freq].set_title(f"{label} Kernel (Power Spectrum)")
                axs_freq[row_freq].set_xlabel("Frequency (Hz)")
                axs_freq[row_freq].set_ylabel("Power")
                row_freq += 1

        # Convolutions (Time Domain)
        for label, sim_PSPs in convolutions.items():
            min_len = min(len(self.fpp.pp.pp_time_axis), len(sim_PSPs))
            axs_time[row_time].plot(
                self.fpp.pp.pp_time_axis[:min_len], sim_PSPs[:min_len], color="purple"
            )
            axs_time[row_time].set_title(f"Convolution with {label} (Time Domain)")
            axs_time[row_time].set_xlabel("Time (s)")
            axs_time[row_time].set_ylabel("Amplitude")
            row_time += 1

        # Spectra for Convolutions (Frequency Domain)
        for label, spectrum in spectra.items():
            axs_freq[row_freq].loglog(
                self.fpp.pp.pp_frequencies, spectrum, color="purple"
            )
            axs_freq[row_freq].set_title(f"Convolution with {label} (Power Spectrum)")
            axs_freq[row_freq].set_xlabel("Frequency (Hz)")
            axs_freq[row_freq].set_ylabel("Power")
            row_freq += 1

        fig_time.tight_layout()
        fig_freq.tight_layout()
        plt.show()
