import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Class to visualize the filtered point process."""

    def __init__(self, fpp_instance):
        self.fpp = fpp_instance

    def plot_all_stages(self):
        fig, axs = plt.subplots(6, 2, figsize=(15, 18))

        # Model 1 CIF
        if self.fpp.pp_1:
            if hasattr(self.fpp.pp_1, "cif_time_axis") and hasattr(
                self.fpp.pp_1, "cif_realization"
            ):
                axs[0, 0].plot(
                    self.fpp.pp_1.cif_time_axis, self.fpp.pp_1.cif_realization
                )
                axs[0, 0].set_title("Model 1 - CIF (Time Domain)")
                axs[0, 0].set_xlabel("Time (s)")
                axs[0, 0].set_ylabel("Intensity")

            if hasattr(self.fpp.pp_1, "cif_frequencies") and hasattr(
                self.fpp.pp_1, "cif_PSD"
            ):
                axs[0, 1].plot(self.fpp.pp_1.cif_frequencies, self.fpp.pp_1.cif_PSD)
                axs[0, 1].set_title("Model 1 - CIF (Power Spectrum)")
                axs[0, 1].set_xlabel("Frequency (Hz)")
                axs[0, 1].set_ylabel("Power")

                # Set x-axis limits based on center frequency and peak width
                if (
                    "center_frequency" in self.fpp.pp_1.params
                    and "peak_width" in self.fpp.pp_1.params
                ):
                    x_min = (
                        self.fpp.pp_1.params["center_frequency"]
                        - 5 * self.fpp.pp_1.params["peak_width"]
                    )
                    x_max = (
                        self.fpp.pp_1.params["center_frequency"]
                        + 5 * self.fpp.pp_1.params["peak_width"]
                    )
                    axs[0, 1].set_xlim([x_min, x_max])

        # Model 1 Point Process
        if self.fpp.pp_1:
            if hasattr(self.fpp.pp_1, "pp_events"):
                axs[1, 0].eventplot(self.fpp.pp_1.pp_events, colors="black")
                axs[1, 0].set_title("Model 1 - Point Process (Time Domain)")
                axs[1, 0].set_xlabel("Time (s)")
                axs[1, 0].set_ylabel("Events")
                axs[1, 0].spines["top"].set_visible(False)
                axs[1, 0].spines["right"].set_visible(False)
                axs[1, 0].spines["left"].set_visible(False)
                axs[1, 0].spines["bottom"].set_visible(False)
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])

            if hasattr(self.fpp.pp_1, "pp_frequencies") and hasattr(
                self.fpp.pp_1, "pp_PSD"
            ):
                axs[1, 1].plot(self.fpp.pp_1.pp_frequencies, self.fpp.pp_1.pp_PSD)
                axs[1, 1].set_title("Model 1 - Point Process (Power Spectrum)")
                axs[1, 1].set_xlabel("Frequency (Hz)")
                axs[1, 1].set_ylabel("Power")

                # Set x-axis limits based on center frequency and peak width
                if (
                    "center_frequency" in self.fpp.pp_1.params
                    and "peak_width" in self.fpp.pp_1.params
                ):
                    x_min = (
                        self.fpp.pp_1.params["center_frequency"]
                        - 5 * self.fpp.pp_1.params["peak_width"]
                    )
                    x_max = (
                        self.fpp.pp_1.params["center_frequency"]
                        + 5 * self.fpp.pp_1.params["peak_width"]
                    )
                    axs[1, 1].set_xlim([x_min, x_max])

        # Model 1 Filter 1
        if (
            hasattr(self.fpp, "filter_1_instance_model_1")
            and self.fpp.filter_1_instance_model_1
        ):
            filter_1_outputs = self.fpp.get_individual_outputs().get(
                "model_1_filter_1", {}
            )
            if "time_axis" in filter_1_outputs and "kernel" in filter_1_outputs:
                axs[2, 0].plot(
                    filter_1_outputs["time_axis"],
                    filter_1_outputs["kernel"],
                    color="blue",
                )
                axs[2, 0].set_title("Model 1 - Filter 1 Kernel (Time Domain)")
                axs[2, 0].set_xlabel("Time (s)")
                axs[2, 0].set_ylabel("Kernel")

            if (
                "frequencies" in filter_1_outputs
                and "power_spectrum" in filter_1_outputs
            ):
                axs[2, 1].loglog(
                    filter_1_outputs["frequencies"],
                    filter_1_outputs["power_spectrum"],
                    color="blue",
                )
                axs[2, 1].set_title("Model 1 - Filter 1 Kernel (Power Spectrum)")
                axs[2, 1].set_xlabel("Frequency (Hz)")
                axs[2, 1].set_ylabel("Power")

        # Model 1 Filter 2 (Optional)
        if (
            hasattr(self.fpp, "filter_2_instance_model_1")
            and self.fpp.filter_2_instance_model_1
        ):
            filter_2_outputs = self.fpp.get_individual_outputs().get(
                "model_1_filter_2", {}
            )
            if "time_axis" in filter_2_outputs and "kernel" in filter_2_outputs:
                axs[3, 0].plot(
                    filter_2_outputs["time_axis"],
                    filter_2_outputs["kernel"],
                    color="green",
                )
                axs[3, 0].set_title("Model 1 - Filter 2 Kernel (Time Domain)")
                axs[3, 0].set_xlabel("Time (s)")
                axs[3, 0].set_ylabel("Kernel")

            if (
                "frequencies" in filter_2_outputs
                and "power_spectrum" in filter_2_outputs
            ):
                axs[3, 1].loglog(
                    filter_2_outputs["frequencies"],
                    filter_2_outputs["power_spectrum"],
                    color="green",
                )
                axs[3, 1].set_title("Model 1 - Filter 2 Kernel (Power Spectrum)")
                axs[3, 1].set_xlabel("Frequency (Hz)")
                axs[3, 1].set_ylabel("Power")

        # Model 1 Convolution with Filter 1
        if hasattr(self.fpp, "perform_convolutions"):
            convolution_results_1 = self.fpp.perform_convolutions().get("model_1", {})
            if (
                "sim_PSPs" in convolution_results_1
                and len(convolution_results_1["sim_PSPs"]) > 0
            ):
                min_len = min(
                    len(self.fpp.pp_1.pp_time_axis),
                    len(convolution_results_1["sim_PSPs"]),
                )
                axs[4, 0].plot(
                    self.fpp.pp_1.pp_time_axis[:min_len],
                    convolution_results_1["sim_PSPs"][:min_len],
                    color="purple",
                )
                axs[4, 0].set_title("Model 1 - Convolution with Filter 1 (Time Domain)")
                axs[4, 0].set_xlabel("Time (s)")
                axs[4, 0].set_ylabel("Amplitude")

            if "sim_PSPs" in convolution_results_1:
                h_spectrum_1 = self.fpp._get_h_spectra().get(
                    "model_1_filter_1", [[], []]
                )[1]
                if len(h_spectrum_1) > 0:
                    axs[4, 1].loglog(
                        self.fpp.pp_1.pp_frequencies, h_spectrum_1, color="purple"
                    )
                    axs[4, 1].set_title(
                        "Model 1 - Convolution with Filter 1 (Power Spectrum)"
                    )
                    axs[4, 1].set_xlabel("Frequency (Hz)")
                    axs[4, 1].set_ylabel("Power")

        # Model 1 Convolution with Filter 2 (Optional)
        if (
            hasattr(self.fpp, "filter_2_instance_model_1")
            and self.fpp.filter_2_instance_model_1
            and "sim_LFP" in convolution_results_1
        ):
            min_len = min(
                len(self.fpp.pp_1.pp_time_axis), len(convolution_results_1["sim_LFP"])
            )
            axs[5, 0].plot(
                self.fpp.pp_1.pp_time_axis[:min_len],
                convolution_results_1["sim_LFP"][:min_len],
                color="orange",
            )
            axs[5, 0].set_title("Model 1 - Convolution with Filter 2 (Time Domain)")
            axs[5, 0].set_xlabel("Time (s)")
            axs[5, 0].set_ylabel("Amplitude")

            h_spectrum_2 = self.fpp._get_h_spectra().get("model_1_filter_2", [[], []])[
                1
            ]
            if len(h_spectrum_2) > 0:
                axs[5, 1].loglog(
                    self.fpp.pp_1.pp_frequencies, h_spectrum_2, color="orange"
                )
                axs[5, 1].set_title(
                    "Model 1 - Convolution with Filter 2 (Power Spectrum)"
                )
                axs[5, 1].set_xlabel("Frequency (Hz)")
                axs[5, 1].set_ylabel("Power")

        plt.tight_layout()
        plt.show()
