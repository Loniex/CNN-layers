import numpy as np

def normalization(inputs, mean, variance, epsilon):
    inputs = np.array(inputs)
    mean = np.array(mean)
    variance = np.array(variance)
    epsilon = np.array(epsilon)

    std_dev = np.sqrt(variance)

    result = (inputs - mean) / (std_dev + epsilon)

    return result
