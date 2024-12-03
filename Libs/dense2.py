#Dense2 for the neural network 
import numpy as np 

def dense_layer2(inputs, weights, bias):
   #Ensure that whatever we are getting is translated to numpy 
   inputs = np.array(inputs)
   weights = np.array(weights)
   bias = np.array(bias)

   inputs_range_dimensions = inputs.shape[0] #check how many dimensions result has 
   inputs_range = inputs.shape[1] #128 values
   output_range = bias.shape[0] #8 values 
   #get the shape of each input for the for loop
   result = np.zeros((inputs_range_dimensions,output_range))

   for i in range(inputs_range_dimensions):
    for n in range(output_range): 
     for w in range(inputs_range):
       #Inputs- 1,128, weights 128,8, bias(8)
       result[i,n] = result[i,n] + inputs[i,w] * weights[w,n] 
     result[i,n] = result[i,n] + bias[n]
   return result
