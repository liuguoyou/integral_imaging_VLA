import os
import numpy as np

from PIL import Image

import utils


class SubAperture(object):
    def __init__(self, f, g, is_masking, is_prediction):
        self.path = './sub_aperture/' + 'F' + str(f) + 'G' + str(g) + 'M_' + is_masking + 'P_' + is_prediction + '/'
        
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def generate_sub_apertures(self, elem_plane, P_L, num_of_lenses):
        sub_apertures = np.zeros((P_L * num_of_lenses, P_L * num_of_lenses, 3))
        sub_aperture = np.zeros((num_of_lenses, num_of_lenses, 3))

        for elem_i in range(P_L):
            for elem_j in range(P_L):
                for i in range(num_of_lenses):
                    for j in range(num_of_lenses):
                        sub_aperture[i, j, :] = elem_plane[elem_i + (i * P_L), elem_j + (j * P_L), :]

                # utils.save_image(sub_aperture, self.check + 'sub_aperture{},{}.png'.format(elem_i, elem_j))
                sub_apertures[elem_i * num_of_lenses:elem_i * num_of_lenses + num_of_lenses, elem_j * num_of_lenses:elem_j * num_of_lenses + num_of_lenses] = sub_aperture

        utils.save_image(sub_apertures, self.path + 'sub_aperture.png')
        return sub_apertures

