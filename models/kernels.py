import numpy as np
import cv2
from scipy.ndimage import convolve, uniform_filter
import torch
import torchvision.models as models

class GaborDescriptor:
    def __init__(self, use_alexnet=False, padding='nopad', normalize=True, lambdas=[1.0, 2.0, 3.0], thetas=np.arange(0, np.pi, np.pi / 6), sigmas=[3, 5]):
        self.use_alexnet = use_alexnet
        self.lambdas = lambdas
        self.thetas = thetas
        self.sigmas = sigmas
        self.normalize = normalize
        self.padding = padding
        self.kernels = self.build_kernels()
    
    def build_kernels(self):
        if self.use_alexnet:
            return self.build_alexnet_kernels()
        else:
            return self.build_gabor_kernels()
    
    def build_gabor_kernels(self):
        kernels = []
        for theta in self.thetas:
            for sigma in self.sigmas:
                for lambd in self.lambdas:
                    ksize = int(6 * sigma + 1)  # Kernel size, typically 6*sigma + 1
                    gamma = 1.0                # Spatial aspect ratio
                    psi = 0                    # Phase offset
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
                    kernels.append(kernel)
        if self.normalize:
            kernels = [kernel / np.linalg.norm(kernel) for kernel in kernels]
        return kernels

    def build_alexnet_kernels(self):
        # Load the pre-trained AlexNet model
        alexnet = models.alexnet(pretrained=True)
        
        # Extract the first convolutional layer's weights
        conv1_weights = alexnet.features[0].weight.data.numpy()
        conv1_weights_avg = np.mean(conv1_weights, axis=1, keepdims=True)
        if self.normalize:
            conv1_weights_avg -= conv1_weights_avg.min()
            conv1_weights_avg /= conv1_weights_avg.max()
        
        # Extract the first filter from the first channel
        # and prepare it as a single kernel
        kernels = [conv1_weights_avg[i, 0, :, :] for i in range(conv1_weights_avg.shape[0])]
        
        return kernels

    def apply_pooling(self, responses, pool_size=2):
        pooled_responses = []
        for response in responses:
            pooled_response = uniform_filter(response, size=pool_size)
            pooled_responses.append(pooled_response)
        # Combine the pooled responses to get the final feature map
        final_feature_map = np.max(pooled_responses, axis=0)
        return final_feature_map

    def __call__(self, image):
        responses = self.apply_filters(image)
        pooled_response = self.apply_pooling(responses)
        return pooled_response

    def apply_filters(self, image):
        responses = []
        for kernel in self.kernels:
            if self.padding == 'nopad':
                filtered = convolve(image, kernel, mode='constant', cval=0.0)
            else:
                filtered = convolve(image, kernel, mode='reflect')
            responses.append(filtered)
        return np.array(responses)