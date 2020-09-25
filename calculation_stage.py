""" Calculation stage implementation"""

import os
import numpy as np

from PIL import Image
from tqdm import tqdm

import utils


class CalculationStage(object):
    def __init__(self, f, g, is_masking, is_prediction):
        self.path = './elemental_image/' + 'F' + str(f) + 'G' + str(g) + 'M_' + is_masking + 'P_' + is_prediction + '/'
        
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def generate_object_coords(self, color, L):
        """Generate coordinates of object pixel

        Args:
            color        : Color image
            L            : Converted depth image
        Returns:
            pixel_coords : Coordinates of object pixels (x, y, depth)
        """
        height, width, _ = color.shape
        coords = utils.generate_coords(height, width, L, is_depth=True)
        return coords

    def generate_virtual_lens(self, num_of_lens, P_L):
        """Generate virtual lens array

        Args:
            num_of_lens : Number of lenses of lens array
            P_L         : Size of elemental lens
        Returns:
            lenses_idx  : Indices of virtual lenses
            lenses_loc  : Location of each lenses
        """
        lenses_idx = []
        for i_L in range(num_of_lens):
            for j_L in range(num_of_lens):
                lenses_idx.append((i_L, j_L))

        lenses_loc = utils.generate_coords(num_of_lens, num_of_lens)
        return lenses_idx, lenses_loc

    def generate_elemental_plane(self, num_of_lens, P_L):
        """Generate elemental image plane

        Args:
            num_of_lens : Number of lenses of lens array
            P_L         : Size of elemental lens
        Returns:
            elem_plane  : Elemental image plane
            elem_coords : Coordinates of elemental image
        """
        elem_plane_h = P_L * num_of_lens
        elem_plane_w = P_L * num_of_lens

        elem_plane = np.zeros((elem_plane_h, elem_plane_w, 3))
        elem_coords = utils.generate_coords(elem_plane_h, elem_plane_h)
        return elem_plane, elem_coords

    def points_transfrom(self, i, j, i_L, j_L, P_L, P_I, g, L):
        """Transform points of object pixels to elemental image pixels

        Args:
            i   : Location of pixel
            j   : Location of pixel
            i_L : Elemental lens index
            j_L : Elemental lens index
            P_L : Size of elemental lens
            P_I : Pixel size of the object image
            g   : Gap between lens and display
            L   : Converted depth information
        Returns:
            u   : Transformed coordinate corresponding 'x'
            v   : Transformed coordinate corresponding 'y'
        """
        u = P_L * i_L - ((i * P_I) - (P_L * i_L)) * (g / L)
        v = P_L * j_L - ((j * P_I) - (P_L * j_L)) * (g / L)
        return u, v

    def generate_elemental_imgs_CPU(self, color, L, P_L, P_I, g, num_of_lens):
        """Generate elemental images by paper's method

        Args:
        Returns:
        """
        height, width, _ = color.shape

        pixel_coords = self.generate_object_coords(color, L)
        lenses_idx, lenses_loc= self.generate_virtual_lens(num_of_lens, P_L)
        elem_plane, elem_coords = self.generate_elemental_plane(num_of_lens, P_L)

        elem_plane_h, elem_plane_w, _ = elem_plane.shape
        half_elem = elem_plane_h // 2
        half_h = height // 2
        half_w = width // 2

        pixel_idx = []
        for i in range(height):
            for j in range(width):
                pixel_idx.append((i, j))

        for i_L, j_L in tqdm(lenses_idx, ascii=True):
            shift_x = j_L * P_L
            shift_y = i_L * P_L
            
            lens_coords = elem_coords[:, shift_y:shift_y + P_L, shift_x:shift_x + P_L]
            lens_x_min, lens_x_max = lens_coords[0].min(), lens_coords[0].max()
            lens_y_min, lens_y_max = lens_coords[1].min(), lens_coords[1].max()
        
            u, v = self.points_transfrom(pixel_coords[0],
                                         pixel_coords[1],
                                         lenses_loc[0, i_L, j_L],
                                         lenses_loc[1, i_L, j_L],
                                         P_L,
                                         P_I,
                                         g,
                                         pixel_coords[2])
            
            for i, j in pixel_idx:
                if lens_x_min <= int(u[i, j]) <= lens_x_max and lens_y_min <= int(v[i, j]) <= lens_y_max:
                    elem_plane[int(v[i, j]) + half_elem, int(u[i, j]) + half_elem, :] = color[int(pixel_coords[1, i, j]) + half_h, 
                                                                                              int(pixel_coords[0, i, j]) + half_w, 
                                                                                              :]
        
        utils.save_image(elem_plane, self.path + 'elemental_plane.png')
        return elem_plane