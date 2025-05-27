import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding

        # Weights structure initialization
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        # Kaiming Initialization
        self.weights = self.weights * np.sqrt(2. / (in_channels * kernel_size * kernel_size))

        self.bias = np.zeros(out_channels)

    def forward(self, x):
        batch_size, h_in, w_in, c_in = x.shape
        assert c_in == self.in_channels, f"Expected {self.in_channels} channels, got {c_in}"

        # 1. Adding padding to the input dimensions
        # 2. Subtracting kernel size 
        # 3. Dividing by stride
        # 4. Adding 1 for the output dimensions
        h_out = int((h_in + 2 * self.padding - self.kernel_size) / self.stride) + 1
        w_out = int((w_in + 2 * self.padding - self.kernel_size) / self.stride) + 1

        # Adding padding to the input images
        x_padded = np.pad(x, ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)), mode='constant')

        # Output structure initialization
        out = np.zeros((batch_size, h_out, w_out, self.out_channels))

        # for each image in the batch
        for image_index in range(batch_size):
            # for each row of the output
            for i in range(h_out):
                # for each column of the output
                for j in range(w_out):
                    # for each output channel
                    for oc in range(self.out_channels):

                        h_start = i * self.stride
                        w_start = j * self.stride

                        patch = x_padded[image_index,                           # Selecting the image 
                                        h_start:(h_start + self.kernel_size),   # Selecting the range of rows (height patch) in the image
                                        w_start:(w_start + self.kernel_size),   # Selecting the range of columns (width patch) in the image
                                        :self.in_channels]                      # Selecting all channels 
                    
                        out[image_index, i, j, oc] = np.sum(patch * self.weights[oc]) + self.bias[oc]

        return out

    def backward(self, dout):
        batch_size, h_in, w_in, in_channels = self.x.shape
        _, h_out, w_out, out_channels = dout.shape
        
        # Initialize gradients with zeros
        dx = np.zeros_like(self.x)
        dW = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)
        
        # Pad input and gradient w.r.t input for padding
        x_padded = np.pad(self.x,
                          ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)),
                          mode='constant')
        dx_padded = np.pad(dx,
                           ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)),
                           mode='constant')
        
        # Compute gradient w.r.t bias: sum over batch and spatial dimensions
        db = np.sum(dout, axis=(0,1,2))
        
        # Compute gradients for weights and input
        for image_index in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    for oc in range(out_channels):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        
                        # Extract the patch from padded input
                        patch = x_padded[image_index,
                                        h_start:(h_start + self.kernel_size), 
                                        w_start:(w_start + self.kernel_size), 
                                        :]
                        
                        # Update gradient w.r.t weights: multiply patch by dout scalar at this position
                        dW[oc] += patch * dout[image_index, i, j, oc]
                        
                        # Update gradient w.r.t input (padded): multiply weights by dout scalar
                        dx_padded[image_index, 
                                h_start:(h_start + self.kernel_size), 
                                w_start:(w_start + self.kernel_size), 
                                :] += self.weights[oc] * dout[image_index, i, j, oc]
        
        # Remove padding from dx_padded to get dx with original input shape
        if self.padding > 0:
            dx = dx_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dx = dx_padded
        
        # Save gradients for optimizer update
        self.dW = dW
        self.db = db
        
        return dx

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db