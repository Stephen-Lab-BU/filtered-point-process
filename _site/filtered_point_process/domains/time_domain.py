class TimeDomain:
    """
    Initialize a TimeDomain instance.

    Args:
        time_axis (np.ndarray): Array representing the time points in seconds.
        intensity_realization (np.ndarray, optional): Array of intensity values at each time point. Defaults to None.
        events (list of float, optional): List of event times in seconds. Defaults to None.
    """

    def __init__(self, time_axis, intensity_realization=None, events=None):
        self.time_axis = time_axis
        self.intensity_realization = intensity_realization
        self.events = events

    def get_time_axis(self):
        """
        Retrieve the time axis.

        Returns:
            np.ndarray: Array of time points in seconds.
        """
        return self.time_axis

    def get_intensity_realization(self):
        """
        Retrieve the intensity realization.

        Returns:
            np.ndarray or None: Array of intensity values at each time point, or None if not set.
        """
        return self.intensity_realization

    def get_events(self):
        """
        Retrieve the list of event times.

        Returns:
            list of float or None: List of event times in seconds, or None if not set.
        """
        return self.events


def create_time_domain(time_axis, intensity_realization=None, events=None):
    """ "
    Factory function to create a TimeDomain object.

    This function instantiates a TimeDomain class with the provided time axis, intensity realization, and events.

    Args:
        time_axis (np.ndarray): Array representing the time points in seconds.
        intensity_realization (np.ndarray, optional): Array of intensity values at each time point. Defaults to None.
        events (list of float, optional): List of event times in seconds. Defaults to None.

    Returns:
        TimeDomain: An instance of the TimeDomain class containing the provided data.
    """
    return TimeDomain(time_axis, intensity_realization, events)
