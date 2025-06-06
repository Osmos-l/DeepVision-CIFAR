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
        self.input = x  # Save input for backward pass
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

                        patch_permuted = np.transpose(patch, (2,0,1))

                        out[image_index, i, j, oc] = np.sum(patch_permuted * self.weights[oc]) + self.bias[oc]

        return out

    def backward(self, dout):
        batch_size, h_out, w_out, out_channels = dout.shape
        _, h_in, w_in, in_channels = self.input.shape

        # Initialize gradients with zeros
        dW = np.zeros_like(self.weights)  # shape (out_channels, in_channels, k, k)
        db = np.zeros(out_channels)
        dx_padded = np.zeros((batch_size, h_in + 2 * self.padding, w_in + 2 * self.padding, in_channels))

        # Pad the input
        x_padded = np.pad(self.input,
                        ((0, 0),
                        (self.padding, self.padding),
                        (self.padding, self.padding),
                        (0, 0)),
                        mode='constant')

        # Compute gradients for weights and input
        for image_index in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    for oc in range(out_channels):
                        h_start = i * self.stride
                        w_start = j * self.stride

                        # Extract the patch: shape (kernel_size, kernel_size, in_channels)
                        patch = x_padded[image_index,
                                        h_start:(h_start + self.kernel_size),
                                        w_start:(w_start + self.kernel_size),
                                        :]

                        # patch has shape (k, k, in_channels), weights have shape (out_channels, in_channels, k, k)
                        # We need to match axes: so transpose patch to (in_channels, k, k)
                        patch_transposed = patch.transpose(2, 0, 1)  # now (in_channels, k, k)

                        # Update gradient w.r.t weights: element-wise multiply patch with scalar dout and sum
                        dW[oc] += patch_transposed * dout[image_index, i, j, oc]

                        # Update gradient w.r.t input (padded)
                        # weights[oc] shape: (in_channels, k, k), multiply by scalar dout
                        # We transpose back weights[oc] to (k, k, in_channels) for dx_padded addition
                        dx_padded[image_index,
                                h_start:(h_start + self.kernel_size),
                                w_start:(w_start + self.kernel_size),
                                :] += (self.weights[oc].transpose(1, 2, 0)) * dout[image_index, i, j, oc]

        # Remove padding from dx_padded to get dx shape (batch_size, h_in, w_in, in_channels)
        if self.padding > 0:
            dx = dx_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dx = dx_padded

        # Gradient for biases: sum over batch and spatial dimensions
        db = np.sum(dout, axis=(0, 1, 2))

        # Store gradients
        self.dW = dW
        self.db = db

        return dx


    def update(self, learning_rate):
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db

    def save_weights(self, f):
        np.save(f, self.weights)
        np.save(f, self.bias)

    def load_weights(self, f):
        self.weights = np.load(f)
        self.bias = np.load(f)