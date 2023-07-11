import numpy as np
import torch
from PIL import Image
from random import random
from numpy import *
import torchio

from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
    RandomBlur,
    RandomSpike,
    RandomGhosting,
    
)


torch.manual_seed(0)
np.random.seed(0)

class super_transformation(object):
    def __init__(self):
#         super(super_transformation, self).__init__()
        cur_val = float(0.999999999)
        self.transform = Compose([
#             OneOf({
#                 RandomAffine(): 0.5,
#                 RandomMotion():0.5,
# #                 RandomNoise():0.25,
# #                 RandomBlur():0.25,
#                 },p=0.2),
            
            RandomFlip(axes=(0), flip_probability=0.5),
#             RandomMotion(p=0.5),
            RandomAffine(scales = cur_val,
                         degrees = 2,
                         translation =2,
                         p=0.5),
        ])
    
    def __call__(self, image):
        image = self.transform(image)
#         print('type:',image.dtype)
        image = image.astype(float)
        return image
    
    
    
# class super_transformation(object):
#     def __init__(self):
# #         super(super_transformation, self).__init__()
#         self.transform = Compose([
# #             OneOf({
# #                 RandomAffine(): 0.5,
# #                 RandomMotion():0.5,
# # #                 RandomNoise():0.25,
# # #                 RandomBlur():0.25,
# #                 },p=0.2),
            
#             RandomFlip(axes=(0), flip_probability=0.5),
#             RandomMotion(p=0.5),
#         ])
    
#     def __call__(self, image):
#         image = self.transform(image)
# #         print('type:',image.dtype)
#         image = image.astype(float)
#         return image
    
        
    
    
    
    
    
    

