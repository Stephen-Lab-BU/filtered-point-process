class FrequencyDomain:
    """
    Initialize a FrequencyDomain instance.

    Args:
        frequencies (np.ndarray): Array of frequency values in Hz.
        PSD (np.ndarray): Power spectral density values corresponding to the frequencies.
    """

    def __init__(self, frequencies, PSD):
        self.frequencies = frequencies
        self.PSD = PSD

    def get_frequencies(self):
        """
        Retrieve the frequency values.

        Returns:
            np.ndarray: Array of frequency values in Hz."""
        return self.frequencies

    def get_PSD(self):
        """
        Retrieve the power spectral density (PSD) values.

        Returns:
            np.ndarray: Array of PSD values corresponding to the frequencies.
        """
        return self.PSD


def create_frequency_domain(frequencies, PSD):
    """
    Factory function to create a FrequencyDomain object.

    This function instantiates a FrequencyDomain class with the provided frequencies and PSD.

    Args:
        frequencies (np.ndarray): Array of frequency values in Hz.
        PSD (np.ndarray): Power spectral density values corresponding to the frequencies.

    Returns:
        FrequencyDomain: An instance of the FrequencyDomain class containing the provided data.
    """
    return FrequencyDomain(frequencies, PSD)
