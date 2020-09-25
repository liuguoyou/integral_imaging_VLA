""" Input stage implementation"""

import os
import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm

import utils


class InputStage(object):
    def __init__(self, f, g, is_masking, is_prediction):
        self.path = './intermediate/' + 'F' + str(f) + 'G' + str(g) + 'M_' + is_masking + 'P_' + is_prediction + '/'

        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def central_depth(self, f, g):
        """Calculate central depth of the integral imaging system

        Args:
            f : Focal length of elemental lens
            g : Gap between lens and display
        Returns:
            d : Central depth
        """
        d = (f * g) / (g - f)
        return d

    def pixel_size_object_img(self, d, g, P_D):
        """Calculate a pixel size of the object image

        Args:
            d   : central depth
            g   : Gap between lens and display
            P_D : Pixel pitch of LCD
        Returns:
            P_I : Pixel size of the object image
        """
        P_I = (d / g) * P_D
        return P_I

    def convert_depth(self, color, depth, mask, f, g, P_D, P_L, is_masking=False):
        """Convert the depth map

        Args:
            color    : Color RGB image
            depth    : Depth map corresponding the RGB image
            mask     : Mask image for extracting ROI
            f        : Focal length of elemental lens
            g        : Gap between lens and display
            P_D      : Pixel pitch of LCD
            P_L      : Size of elemental lens
        Returns:
            d        : Central depth
            P_I      : Pixel size of the object image
            delta_d  : Depth range of integral imaging
            L        : Converted depth map
            masked_c : Masked RGB object image based on mask image
            masked_d : Masked Depth image based on mask image 
        """
        height, width, _ = color.shape

        d = self.central_depth(f, g)
        P_I = self.pixel_size_object_img(d, g, P_D)

        delta_d = ((2 * d) / P_L) * P_I
        converted_depth_min = d - delta_d / 2
        converted_depth_max = d + delta_d / 2

        L = (d * (depth[depth > 0].max() + depth[depth > 0].min())) / (depth * 2)
        L[np.isinf(L)] = 0
        
        L[L < converted_depth_min] = converted_depth_min
        L[L > converted_depth_max] = converted_depth_max

        if is_masking:
            color, L = utils.extract_roi(color, L, mask)
        
        utils.save_image(color, self.path + 'color.png')
        utils.save_image(utils.visualize_depth(L), self.path + 'converted_depth.png')

        return d, P_I, delta_d, color, L
        
        # if is_masking:
        #     L = np.zeros((color.shape[0], color.shape[1]))
        #     for i in tqdm(range(height), ascii=True, desc='convert_depth'):
        #         for j in range(width):
        #             if depth[i, j] != 0:
        #                 L[i, j] = (d * (depth[depth > 0].max() + depth[depth > 0].min())) / (depth[i, j] * 2)

        #     L[L < converted_depth_min] = converted_depth_min
        #     L[L > converted_depth_max] = converted_depth_max

        #     # Extract region of interest using mask image
        #     masked_c, masked_L = utils.extract_roi(color, L, mask)
            
        #     utils.save_image(masked_c, self.path + 'masked_color.png')
        #     utils.save_image(utils.visualize_depth(masked_L), self.path + 'masked_converted_depth.png')
        #     return d, P_I, delta_d, masked_c, masked_L
        # else:
        #     L = (d * (depth.max() + depth.min())) / (depth * 2)
        #     L[L < converted_depth_min] = converted_depth_min
        #     L[L > converted_depth_max] = converted_depth_max

        #     utils.save_image(utils.visualize_depth(L), self.path + 'converted_depth.png')
        #     return d, P_I, delta_d, color, L