from .filters import Filter
import numpy as np


class FilteredPointProcess(Filter):
    """Class to interact with the filtered point process."""

    def __init__(self, filters=None, model=None):
        super().__init__(filters=filters, model=model)
        self.filter_names = list(filters.keys())
        self.filter_labels = list(filters.values())

    def get_filters(self):
        outputs = {}
        for filter_name, filter_instance in zip(
            self.filter_names, self.filter_instances.values()
        ):
            outputs[filter_name] = {
                "time_axis": filter_instance.kernel_time_axis,
                "kernel": filter_instance.kernel,
                "power_spectrum": filter_instance.kernel_spectrum,
                "frequencies": filter_instance.frequencies,
            }
        return outputs

    def perform_convolutions(self):
        """Perform convolutions with the kernels and spikes stored in the models."""
        results = {}
        spike_times = self.model.pp_events
        time_axis = self.model.pp_time_axis
        fs = self.model.params["fs"]

        spike_train = self._create_spike_train(spike_times, time_axis, fs)

        for i, (filter_name, filter_instance) in enumerate(
            zip(self.filter_labels, self.filter_instances.values())
        ):
            if i == 0:
                sim_PSPs = self._convolve_spikes_with_kernels(
                    spike_train, filter_instance.kernel, fs
                )
                results[f"pp ⨂ {filter_name}"] = sim_PSPs
            else:
                previous_filter_label = " ⨂ ".join(self.filter_labels[:i])
                combined_label = f"pp ⨂ {previous_filter_label} ⨂ {filter_name}"
                sim_PSPs = self._convolve_spikes_with_kernels(
                    results[f"pp ⨂ {previous_filter_label}"], filter_instance.kernel, fs
                )
                results[combined_label] = sim_PSPs
        return results

    def get_spectra(self):
        """Calculate the theoretical power spectrum of the filter."""
        spectra = {}

        for i, filter_name in enumerate(self.filter_names):
            filter_instance = self.filter_instances[filter_name]
            if i == 0:
                spectrum = self._calculate_spectrum(filter_instance) 
                spectra[f"pp * {self.filter_labels[i]}"] = spectrum
            else:
                previous_spectrum = spectra[f"pp * {' * '.join(self.filter_labels[:i])}"]
                current_spectrum = filter_instance.kernel_spectrum 
                combined_spectrum = previous_spectrum * current_spectrum
                combined_spectrum_name = f"pp * {' * '.join(self.filter_labels[:i+1])}"
                spectra[combined_spectrum_name] = combined_spectrum
        return spectra

    def _create_spike_train(self, spike_times, time_axis, fs):
        """Create a spike train from the spike times and the time axis."""
        spike_train = np.zeros(len(time_axis))
        spike_indices = np.searchsorted(time_axis, spike_times)
        spike_train[spike_indices] = 1
        return spike_train

    def _convolve_spikes_with_kernels(self, spike_train, kernel, fs):
        """Helper function to perform convolutions with given kernels and spike train."""
        num_samples = len(spike_train)
        sim_PSPs = np.convolve(spike_train, kernel, mode="full")[:num_samples] / fs
        return sim_PSPs

    def _calculate_spectrum(self, filter_instance):
        """Helper function to calculate power spectrum for a given filter instance."""
        S = filter_instance.pp.pp_PSD
        pp_power_spectrum = np.squeeze(np.array(S))
        primary_spectrum = filter_instance.kernel_spectrum * (
            pp_power_spectrum / filter_instance.pp.params["fs"] ** 2
        )
        return primary_spectrum
