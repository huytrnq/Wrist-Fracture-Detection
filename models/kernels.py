import numpy as np
import cv2
from scipy.ndimage import convolve, uniform_filter
import torch
from torchvision import transforms
import torchvision.models as models
from skimage.feature import hog
from skimage.feature import local_binary_pattern

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
    
    
class HogDescriptor:
    def __init__(self):
        self.hog = hog
    
    def __call__(self, image):
        feature, hog_image = hog(image, orientations=16, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
        return feature
    

class LBP:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius
        
    def compute(self, image):
        # Assuming the image is already in grayscale
        lbp = local_binary_pattern(image, self.num_points, self.radius, method="uniform")
        # Return the histogram of the LBP
        # (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.num_points + 3), range=(0, self.num_points + 2))
        # hist = hist.astype("float")
        # hist /= (hist.sum() + 1e-6)
        return lbp

class LBPDescriptor:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius
        self.lbp = LBP(self.num_points, self.radius)
        
    def __call__(self, image):
        return self.lbp.compute(image)
    
    
class CannyDescriptor:
    def __init__(self, threshold1=150, threshold2=200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    
    def __call__(self, image):
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        edges = cv2.Canny(gray_image, self.threshold1, self.threshold2)
        feature, hog_image = hog(edges, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
        return feature.flatten()
    

class AlexNetDescriptor:
    def __init__(self, selected_features=[0,1,2,4,5,7,15,19,21,26,31,36,37,38,40,44,51]):
        self.alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1').features[0]
        self.alexnet.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.selected_features = selected_features
        
    def __call__(self, image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            features = self.alexnet(image)
        features = features.squeeze(0).detach().numpy()
        features = features[self.selected_features, :, :]
        features = features.mean(axis=0)
        return features