import numpy as np
from numpy.fft import fft, ifft

class LowPassFilter:
    def __init__(self, alpha):
        """
        Initialize the low-pass filter.
        :param alpha: Smoothing factor within the range (0,1). 
                      The closer alpha is to 1, the less smoothing (more weight to recent data).
        """
        self.alpha = alpha
        self.last_filtered_value = None

    def apply_filter(self, new_value):
        """
        Apply the low-pass filter to a new value to get the smoothed output.
        :param new_value: The new incoming data point as a 2D vector [x, y].
        :return: The smoothed data point as a 2D vector [x, y].
        """
        if self.last_filtered_value is None:
            # Initialize with the first value if not already done
            self.last_filtered_value = new_value
        else:
            # Apply the exponential moving average formula
            self.last_filtered_value = np.array([
                self.alpha * new_value[i] + (1 - self.alpha) * self.last_filtered_value[i] for i in range(3)
            ])
        return self.last_filtered_value