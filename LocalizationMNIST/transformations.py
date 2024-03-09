import random
import torch
from PIL import ImageFilter
from torchvision.transforms import functional as F
from skimage.util import random_noise
import numpy as np


class GaussianBlur(object):
    def __init__(self, kernel, sigma=(0.1,0.2)):
        self.kernel = kernel
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))
    

class GaussianNoise(object):
    def __init__(self, mean=0., var=0.05, clip=True):
        self.mean = mean
        self.var = var
        self.clip = clip
    
    def __call__(self, img):
        # Convert PIL image to NumPy array
        img_array = np.array(img)
        # Apply Gaussian noise
        noisy_img = random_noise(img_array, mode='gaussian', mean=self.mean, var=self.var, clip=self.clip)
        # Convert back to PIL image
        noisy_img_pil = F.to_pil_image(torch.tensor(noisy_img).float())
        return noisy_img_pil

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, var={1}, clip={2})'.format(self.mean, self.var, self.clip)
    