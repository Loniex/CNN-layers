import numpy as np

def maxPooling2D(inputs):
   inputs = np.array(inputs)
   amount, height, width, channels = inputs.shape

   result = np.zeros((amount, height // 2, width // 2, channels))
   print(amount, height, width, channels, "Result array shape: ", result.shape)
    
   for a in range(amount):
      for c in range(channels):
         for h in range(height // 2):
            for w in range(wigth // 2): 
               block2d = np.zeros((2,2))
               for hb in range(2):
                  for wb in range(2):
                     block[hb,wb] = inputs[a, h * 2 + hb, w * 2 + wb, c]
               result[a, h, w, c] = np.max(block)
   return result
