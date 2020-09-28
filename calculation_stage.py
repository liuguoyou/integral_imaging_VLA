""" Calculation stage implementation"""

import os
import numpy as np

from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import utils


class CalculationStage(object):
    def __init__(self, name, f, g, is_masking, is_prediction):
        self.path = './elemental_image/' + name + '_F' + str(f) + 'G' + str(g)

        if is_masking:
            self.path = self.path + 'M'
        if is_prediction:
            self.path = self.path + 'P'

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

    def generate_elemental_imgs(self, color, L, P_L, P_I, g, num_of_lens):
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
        
        utils.save_image(elem_plane, self.path + '/elemental_plane.png')
        return elem_plane
    
    def generate_elemental_imgs_GPU(self, color, L, P_L, P_I, g, num_of_lens):
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

        elem_plane_R = elem_plane[:, :, 0].astype(np.float32)
        elem_plane_G = elem_plane[:, :, 1].astype(np.float32)
        elem_plane_B = elem_plane[:, :, 2].astype(np.float32)
        elem_plane_R_gpu = cuda.mem_alloc(elem_plane_R.nbytes)
        elem_plane_G_gpu = cuda.mem_alloc(elem_plane_G.nbytes)
        elem_plane_B_gpu = cuda.mem_alloc(elem_plane_B.nbytes)
        cuda.memcpy_htod(elem_plane_R_gpu, elem_plane_R)
        cuda.memcpy_htod(elem_plane_G_gpu, elem_plane_G)
        cuda.memcpy_htod(elem_plane_B_gpu, elem_plane_B)

        pixel_x = pixel_coords[0].astype(np.float32)
        pixel_y = pixel_coords[1].astype(np.float32)
        pixel_L = pixel_coords[2].astype(np.float32)
        pixel_x_gpu = cuda.mem_alloc(pixel_x.nbytes)
        pixel_y_gpu = cuda.mem_alloc(pixel_y.nbytes)
        pixel_L_gpu = cuda.mem_alloc(pixel_L.nbytes)
        cuda.memcpy_htod(pixel_x_gpu, pixel_x)
        cuda.memcpy_htod(pixel_y_gpu, pixel_y)
        cuda.memcpy_htod(pixel_L_gpu, pixel_L)

        lens_loc_x = lenses_loc[0].astype(np.float32)
        lens_loc_y = lenses_loc[1].astype(np.float32)
        lens_loc_x_gpu = cuda.mem_alloc(lens_loc_x.nbytes)
        lens_loc_y_gpu = cuda.mem_alloc(lens_loc_y.nbytes)
        cuda.memcpy_htod(lens_loc_x_gpu, lens_loc_x)
        cuda.memcpy_htod(lens_loc_y_gpu, lens_loc_y)

        R = color[:, :, 0].astype(np.float32)
        G = color[:, :, 1].astype(np.float32)
        B = color[:, :, 2].astype(np.float32)
        R_gpu = cuda.mem_alloc(R.nbytes)
        G_gpu = cuda.mem_alloc(G.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        cuda.memcpy_htod(R_gpu, R)
        cuda.memcpy_htod(G_gpu, G)
        cuda.memcpy_htod(B_gpu, B)

        elem_coords_x = elem_coords[0].astype(np.float32)
        elem_coords_y = elem_coords[1].astype(np.float32)
        elem_coords_x_gpu = cuda.mem_alloc(elem_coords_x.nbytes)
        elem_coords_y_gpu = cuda.mem_alloc(elem_coords_y.nbytes)
        cuda.memcpy_htod(elem_coords_x_gpu, elem_coords_x)
        cuda.memcpy_htod(elem_coords_y_gpu, elem_coords_y)

        mod = SourceModule("""
            __global__ void generate_EIA(float * R, float * G, float * B,
                                         float * elem_R, float * elem_G, float * elem_B,
                                         float * pixel_x, float * pixel_y, float * pixel_L,
                                         float * elem_coords_x, float * elem_coords_y,
                                         float * lens_loc_x, float * lens_loc_y,
                                         float P_L, float P_I, float g,
                                         int height, int width, int num_of_lens) {

                int p_x = threadIdx.x + blockDim.x * blockIdx.x;
                int p_y = threadIdx.y + blockDim.y * blockIdx.y;

                int i, j;
                float u, v;
                float p_i, p_j;
                int shift_x, shift_y;

                int half_h = (int)(height / 2);
                int half_w = (int)(width / 2);

                int len_of_elem = num_of_lens * (int)P_L;
                int half_elem = (int)(len_of_elem / 2);

                float lens_min_x, lens_min_y;
                
                if (p_x < width && p_y < height) {
                    for (i = 0; i < num_of_lens; i++) {
                        for (j = 0; j < num_of_lens; j++) {
                            shift_x = i * (int)P_L;
                            shift_y = j * (int)P_L;
                            
                            u = P_L * lens_loc_x[i + j * num_of_lens] - ((pixel_x[p_x + p_y * width] * P_I) - (P_L * lens_loc_x[i + j * num_of_lens])) * (g / pixel_L[p_x + p_y * width]);
                            v = P_L * lens_loc_y[i + j * num_of_lens] - ((pixel_y[p_x + p_y * width] * P_I) - (P_L * lens_loc_y[i + j * num_of_lens])) * (g / pixel_L[p_x + p_y * width]);

                            lens_min_x = elem_coords_x[shift_x + shift_y * len_of_elem];
                            lens_min_y = elem_coords_y[shift_x + shift_y * len_of_elem];
                            
                            if ((lens_min_x <= u && u <= lens_min_x + P_L) && (lens_min_y <= v && v <= lens_min_y + P_L)) {
                                u += half_elem;
                                v += half_elem;
                                u = __float2int_ru(u);
                                v = __float2int_ru(v);

                                p_i = pixel_x[p_x + p_y * width] + half_w;
                                p_j = pixel_y[p_x + p_y * width] + half_h;
                                p_i = __float2int_ru(p_i);
                                p_j = __float2int_ru(p_j);

                                if ((0 <= (int)u && (int)u < len_of_elem) && (0 <= (int)v && (int)v < len_of_elem)) {
                                    elem_R[(int)u + (int)v * len_of_elem] = R[(int)p_i + (int)p_j * width];
                                    elem_G[(int)u + (int)v * len_of_elem] = G[(int)p_i + (int)p_j * width];
                                    elem_B[(int)u + (int)v * len_of_elem] = B[(int)p_i + (int)p_j * width];
                                }
                            }
                        }
                    }
                }   
            }
        """)
        
        func = mod.get_function("generate_EIA")
        func(R_gpu, G_gpu, B_gpu,
             elem_plane_R_gpu, elem_plane_G_gpu, elem_plane_B_gpu,
             pixel_x_gpu, pixel_y_gpu, pixel_L_gpu,
             elem_coords_x_gpu, elem_coords_y_gpu,
             lens_loc_x_gpu, lens_loc_y_gpu,
             np.float32(P_L), np.float32(P_I), np.float32(g),
             np.int32(height), np.int32(width), np.int32(num_of_lens),
             block=(20, 20, 1),
             grid=(32, 24))

        elem_R = np.empty_like(elem_plane_R)
        elem_G = np.empty_like(elem_plane_G)
        elem_B = np.empty_like(elem_plane_B)
        cuda.memcpy_dtoh(elem_R, elem_plane_R_gpu)
        cuda.memcpy_dtoh(elem_G, elem_plane_G_gpu)
        cuda.memcpy_dtoh(elem_B, elem_plane_B_gpu)
        
        EIA = np.stack([elem_R, elem_G, elem_B], axis=2)
        utils.save_image(EIA, self.path + '/elemental_plane.png')
        return EIA