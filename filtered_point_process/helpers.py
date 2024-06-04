class TimeDomain:
    def __init__(self, time_axis, intensity_realization=None, events=None):
        self.time_axis = time_axis
        self.intensity_realization = intensity_realization
        self.events = events

    def get_time_axis(self):
        return self.time_axis

    def get_intensity_realization(self):
        return self.intensity_realization

    def get_events(self):
        return self.events


class FrequencyDomain:
    def __init__(self, frequencies, PSD):
        self.frequencies = frequencies
        self.PSD = PSD

    def get_frequencies(self):
        return self.frequencies

    def get_PSD(self):
        return self.PSD


def create_time_domain(time_axis, intensity_realization=None, events=None):
    return TimeDomain(time_axis, intensity_realization, events)


def create_frequency_domain(frequencies, PSD):
    return FrequencyDomain(frequencies, PSD)
