import torchvision
from torchvision.transforms import transforms as T
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.cm as mpl_color_map
import copy
from utils.misc_functions import *


class Display():
    def __init__(self, model, module_name, layer_index, norm=None, img_size=None):
        '''
        model: torch
        module_name: the module corresponding to the layer which you want to take
        layer_index: the index in the module
        '''
        # check is dataparallel mode or not
        self.img_size = img_size
        self.norm = norm
        if isinstance(model, nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        
        if self.norm is None:
            self.norm = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if self.img_size is None:
            self.img_size = (224, 224)
        self.transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(self.norm[0], self.norm[1])
        ])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.module_name = module_name
        self.layer_index = layer_index
        self.model.to(self.device)

    def forward(self, image):
        self.model.eval()
        with torch.no_grad():
            out = image.clone()
            # get the output of the final layer
            final_out = self.model(out)

            for name, module in self.model._modules.items():
                # get goal module
                if name == self.module_name:
                    for index, layer in enumerate(module):
                        # make sure the data have been transfered into [batch, dimension] form
                        if isinstance(layer, nn.Linear):
                            if layer.in_features != out.shape[-1]:
                                out = torch.flatten(out, 1)
                        out = layer(out)
                        if index == self.layer_index:
                            break
                    break
                # pass all previous layers
                else:
                    out = module(out)
        return out, final_out

    def generate_image(self, image, target_class=None):
        """
        image: PIL
        target_class: int (optional)
        """
        with torch.no_grad():
            # get final output and feature map
            image = self.transform(image)
            image = image.unsqueeze(0)
            image = image.to(self.device)
            feature_map, output = self.forward(image)
            feature_map = feature_map.squeeze()
            
            if target_class is None:
                target_class = np.argmax(output.cpu().data.numpy())

            # Create empty numpy array for cam
            cam = np.ones(feature_map.shape[1:], dtype=np.float32)
            # Multiply each weight with its conv output and then, sum
            for i in range(len(feature_map)):
                # Unsqueeze to 4D
                saliency_map = torch.unsqueeze(torch.unsqueeze(feature_map[i, :, :],0),0)
                # Upsampling to input size
                saliency_map = F.interpolate(saliency_map, size=self.img_size, mode='bilinear', align_corners=False)
                if saliency_map.max() == saliency_map.min():
                    continue
                # Scale between 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                # Get the target score
                w = F.softmax(self.forward(image*norm_saliency_map)[1],dim=1)[0][target_class]
                cam += w.cpu().data.numpy() * feature_map[i, :, :].cpu().data.numpy()
            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((image.shape[2],
                           image.shape[3]), Image.ANTIALIAS))/255
        return cam
    
    def save(self, image, file='network_name'):
        cam = self.generate_image(image)
        save_class_activation_images(image, cam, file)
