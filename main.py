import os
import math
import argparse
import numpy as np

from PIL import Image

from predict import DepthEstimation
from input_stage import InputStage
from calculation_stage import CalculationStage
from sub_aperture import SubAperture

import utils 


parser = argparse.ArgumentParser(description='Generation Light Field using Depth Estimation.')
parser.add_argument('--color_path', type=str, default='./inputs/color.png', help='Input image.')
parser.add_argument('--depth_path', type=str, default='./inputs/depth.png', help='Depth image.')
parser.add_argument('--mask_path', type=str, default=None, help='Mask image.')

parser.add_argument('--model_path', type=str, default='./networks/model.h5', help='Model file for predicting a depth.')

parser.add_argument('--is_masking', action='store_true', help='Check masking or no masking.')
parser.add_argument('--is_prediction', action='store_true', help='Depth estimation from a RGB image.')
parser.add_argument('--is_gpu', action='store_true', help='Select calculation system.')

parser.add_argument('--num_of_lenses', type=int, default=12769, help='Number of elemental lenses.')
parser.add_argument('--P_D', type=float, default=0.1245, help='Pixel pitch of LCD.')
parser.add_argument('--P_L', type=float, default=1.992, help='Size of elemental lens.')
parser.add_argument('--f', type=float, default=10, help='Focal length of elemental lens.')
parser.add_argument('--g', type=float, default=11, help='Gap between lens and display.')
args = parser.parse_args()


def cvt_mm2pixel(inputs, pitch_of_pixel):
    cvt_inputs = {}
    cvt_inputs['depth'] = utils.cvt_mm2pixel(inputs['depth'], pitch_of_pixel)
    cvt_inputs['P_D'] = utils.cvt_mm2pixel(inputs['P_D'], pitch_of_pixel)
    cvt_inputs['P_L'] = utils.cvt_mm2pixel(inputs['P_L'], pitch_of_pixel)
    cvt_inputs['f'] = utils.cvt_mm2pixel(inputs['f'], pitch_of_pixel)
    cvt_inputs['g'] = utils.cvt_mm2pixel(inputs['g'], pitch_of_pixel)
    return cvt_inputs


def get_input_params():
    """Parameters
    
    Object image
        - color : Color image of the 3D object
        - depth : Depth image of the 3D object
        - mask  : Mask image for extracting ROI
    
    Parameter input
        Information of Lens-array
            - P_L           : Size of elemental lens
            - num_of_lenses : Number of elemental lens
            - f             : Focal length of elemental lens

        Information of Display
            - P_D           : Pixel pitch of LCD
            - g             : Gap between lens and display
    """
    inputs = {}
    color = utils.load_image(args.color_path)

    if args.is_prediction:
        print('Estimate Depth...')
        depth_estimation = DepthEstimation()
        depth = depth_estimation.prediction(args.color_path, args.model_path)
        utils.save_image(utils.visualize_depth(depth), './intermediate/predicted_depth.png')
        print('Depth Estimation Done.')
    else:
        depth = utils.load_image(args.depth_path)
    
    if args.is_masking:
        mask = utils.load_image(args.mask_path)
    else:
        mask = None

    inputs['color'] = color
    inputs['depth'] = depth
    inputs['mask'] = mask
    inputs['num_of_lenses'] = args.num_of_lenses
    inputs['P_D'] = args.P_D
    inputs['P_L'] = args.P_L
    inputs['f'] = args.f
    inputs['g'] = args.g
    return inputs


def main():
    '''
        Input Stage
    '''
    # Set up the inputs.
    # Micro lens array : 113 x 113 (12769 lenses)
    print('\nInput Stage...')

    inputs = get_input_params()

    # Convert mm to pixel
    cvt_inputs = cvt_mm2pixel(inputs, pitch_of_pixel=inputs['P_D'])

    # Convert depth data
    inputstage = InputStage(int(args.f),
                            int(args.g),
                            str(args.is_masking),
                            str(args.is_prediction))
    d, P_I, delta_d, color, L = inputstage.convert_depth(inputs['color'],
                                                         cvt_inputs['depth'],
                                                         inputs['mask'],
                                                         cvt_inputs['f'],
                                                         cvt_inputs['g'],
                                                         cvt_inputs['P_D'],
                                                         cvt_inputs['P_L'],
                                                         is_masking=args.is_masking)

    print('Input Stage Done.')

    # Print parameters
    utils.print_params(inputs, cvt_inputs, d, P_I, delta_d, color, L)

    '''
        Calculation Stage
    '''
    # Generate elemental images
    print('\nCalculation Stage...')

    calculationstage = CalculationStage(int(args.f),
                                        int(args.g),
                                        str(args.is_masking),
                                        str(args.is_prediction))
    elem_plane = calculationstage.generate_elemental_imgs_CPU(color, 
                                                              L,
                                                              int(cvt_inputs['P_L']),
                                                              P_I,
                                                              cvt_inputs['g'],
                                                              int(np.sqrt(inputs['num_of_lenses'])))

    print('Calculation Stage Done.')

    '''
        Generate Sub Aperture
    '''
    print('Generate sub aperture images...')

    aperture = SubAperture(int(args.f),
                           int(args.g),
                           str(args.is_masking),
                           str(args.is_prediction))
    sub_apertures = aperture.generate_sub_apertures(elem_plane,
                                                    int(cvt_inputs['P_L']),
                                                    int(np.sqrt(inputs['num_of_lenses'])))
    
    print('Done.')


if __name__ == "__main__":
    main()