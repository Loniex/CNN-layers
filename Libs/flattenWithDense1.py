import numpy as np
import relu as f

def flatten(inputs):
    inputs = np.array(inputs)
    outputSize = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
    result  = np.zeros((inputs.shape[0], outputSize)) 

    result = inputs.reshape((inputs.shape[0], outputSize))
    return result

#Input shape: 1 , 12544
#Weight shape: 12544 , 128
#Bias shape: 128
#(1, 128)
def dense(inputs, weights, bias):
    inputs  = np.array(inputs)
    weights = np.array(weights)
    bias = np.array(bias)

    outputSize = bias.shape[0]  
    inputSize =  inputs.shape[1]  
    outputAmount = inputs.shape[0] 
    
    result = np.zeros((outputAmount, outputSize))
    
    for i in range(outputAmount):
     for x in range(outputSize):
      for y in range(inputSize):
       result[i,x] += inputs[i,y] * weights[y,x]
      result[i,x] += bias[x]
      result[i,x] = f.ReLU(result[i,x])
      
    return result
