from PIL import Image

import numpy as np


def load_image(path):
    """Load image

    Args:
        path : Image path
    Returns:
        img  : Numpy array (image)
    """
    img = Image.open(path)
    img = np.asarray(img)
    return img


def save_image(img, path):
    """Save image

    Args:
        img  : Numpy array (image)
        path : Path for saving image
    """
    img = Image.fromarray(img.astype(np.uint8))
    img.save(path)


def visualize_depth(depth):
    """Visualize depth datas

    Args:
        depth        : depth map
    Returns:
        visual_depth : Visualized depth map (0 ~ 255)
    """
    # Normalize depth map (using depth.max())
    depth = depth / depth.max()
    visual_depth = np.uint8(depth * 255.0)
    return visual_depth


def generate_coords(height, width, depth=None, is_depth=False):
    """

    Args:
    Returns
    """
    x = np.linspace(-int(width / 2), int(width / 2) - 1, width)
    y = np.linspace(-int(height / 2), int(height / 2) - 1, height)
    x, y = np.meshgrid(x, y)
    
    if is_depth:
        coords = np.stack([x, y, depth], axis=0)
    else:
        coords = np.stack([x, y], axis=0)
    return coords


def cvt_mm2pixel(mm, pixel_pitch):
    """Convert mm to pixel

    Args:
        mm          : mm unit of size
        pixel_pitch : Pixel pitch of LCD
    Returns:
        pixels      : Converted unit (mm to pixel)
    """
    pixels = mm / pixel_pitch
    return pixels


def extract_roi(color, depth, mask):
    """Extract region of interest

    Args:
        color    : Color RGB image
        depth    : Depth map corresponding the RGB image
        mask     : Mask image for extracting ROI
    Returns:
        masked_c : masked RGB image
        masked_d : masked Depth image
    """
    masked_r = color[:, :, 0] * mask
    masked_g = color[:, :, 1] * mask
    masked_b = color[:, :, 2] * mask
    masked_color = np.stack([masked_r, masked_g, masked_b], axis=2)
    masked_depth = depth * mask
    return masked_color, masked_depth


def print_params(inputs, cvt_inputs, d, P_I, delta_d, color, L):
    print('\n[Parameters Input]')
    print('     Number of elemental lens : {}'.format(inputs['num_of_lenses']))
    print('     (P_D) Pixel pitch of LCD : {} mm'.format(inputs['P_D']))
    print('     (P_L) Size of elemental lens : {} mm'.format(inputs['P_L']))
    print('     (f) Focal length of elemental lens : {} mm'.format(inputs['f']))
    print('     (g) Gap between lens and display : {} mm'.format(inputs['g']))
    print('     Minimum Depth : {:0.4f} mm'.format(inputs['depth'][inputs['depth'] > 0].min()))
    print('     Maximum Depth : {:0.4f} mm'.format(inputs['depth'][inputs['depth'] > 0].max()))

    print('\n[Parameters converted mm to pixel]')
    print('     (P_D) Pixel pitch of LCD : {} pixels'.format(cvt_inputs['P_D']))
    print('     (P_L) Size of elemental lens : {:0.4f} pixels'.format(cvt_inputs['P_L']))
    print('     (f) Focal length of elemental lens : {:0.4f} pixels'.format(cvt_inputs['f']))
    print('     (g) Gap between lens and display : {:0.4f} pixels'.format(cvt_inputs['g']))
    print('     Minimum Depth : {:0.4f} pixels'.format(cvt_inputs['depth'][cvt_inputs['depth'] > 0].min()))
    print('     Maximum Depth : {:0.4f} pixels'.format(cvt_inputs['depth'][cvt_inputs['depth'] > 0].max()))

    print('\n[Parameters for converting depth]')
    print('     (d) central depth : {:0.4f} pixels'.format(d))
    print('     (P_I) pixel size of the object image : {:0.4f} pixels'.format(P_I))
    print('     (∆d) delta_d : {:0.4f} pixels'.format(delta_d))

    print('\n[Converted min & max depth]')
    print('     Minimum Depth : {:0.4f} pixels'.format(L[L > 0].min()))
    print('     Maximum Depth : {:0.4f} pixels'.format(L[L > 0].max()))