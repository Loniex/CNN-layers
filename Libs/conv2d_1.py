import numpy as np
import relu as f

def conv2dLayer1(inputs, weights, bias):
    inputs = np.array(inputs)
    weights = np.array(weights)
    bias = np.array(bias)

    batch, height, width, channels = inputs.shape
    outputChannels = bias.shape[0]
    filterHeight = weights.shape[0]
    filterWidth = weights.shape[1]
    resHeight = height - filterHeight + 1
    resWidth = width - filterWidth + 1
    result = np.zeros((batch, resHeight, resWidth, outputChannels))

    for b in range(batch):
       for ch in range(outputChannels):
          for rH in range(resHeight):
             for rW in range(resWidth):
                filterSum = 0
                for fH in range(filterHeight):
                   for fW in range(filterWidth):
                      for fCH in range(channels):
                         filterSum += inputs[b, rH + fH, rW + fW, fCH] * weights[fH, fW, fCH, ch]
                         result[b, rH, rW, ch] = filterSum + bias[ch]
                         result[b, rH, rW, ch] = f.ReLU(result[b, rH, rW, ch])



    return result

def conv2dLayer(inputs, weights, bias):
    inputs = np.array(inputs)
    weights = np.array(weights)
    bias = np.array(bias)

    batch, height, width, filterChannels = inputs.shape
    filterHeight = weights.shape[0]
    filterWidth = weights.shape[1]
    resultHeight = height - filterHeight + 1
    resultWidth  = width - filterWidth + 1
    channels = bias.shape[0]
    result = np.zeros((batch, resultHeight, resultWidth, channels))

    for b in range(batch):
       for rH in range(resultHeight):
          for rW in range(resultWidth):
             for ch in range(channels):
                filterSum = 0
                for fH in range(filterHeight):
                   for fW in range(filterWidth):
                      for inCH in range(filterChannels):
                         filterSum = filterSum + inputs[b, rH + fH, rW + fW, inCH] * weights[fH, fW, inCH, ch]
                result[b, rH, rW, ch] = filterSum + bias[ch]
                result[b, rH, rW, ch] = f.ReLU(result[b, rH, rW, ch])

    return result




