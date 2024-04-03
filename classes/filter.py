import numpy as np

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_filtered_value = None

    def apply_filter(self, new_value):
        #print("new value:", new_value)
        if self.last_filtered_value is None:
            # Initialize with the first value if not already done
            self.last_filtered_value = new_value
        else:
            # Apply the exponential moving average formula
            
            print("new value:", new_value)
            print("last filtered value:", self.last_filtered_value)
            for i in range(len(new_value)):
                for j in range(len(new_value[i])):
                    self.last_filtered_value[i][j] = self.alpha * new_value[i][j] + (1 - self.alpha) * self.last_filtered_value[i][j]
        print("filtered value:", self.last_filtered_value)
        return self.last_filtered_value