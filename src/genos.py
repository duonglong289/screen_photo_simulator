import glob
import cv2
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
import argparse
import random

from image_tools import *
from moire import linear_wave, dither
from basic_shapes import circles, radialShape
from module import RecaptureModule

import imgaug.augmenters as iaa
from opensimplex import OpenSimplex


def create_simplex_mask(shape=[100, 100], feature_size=None):
    wid, hei = shape[:2]
    if feature_size is None:
        feature_size = random.randint(2, int(0.5*min(wid, hei)))

    gray = np.zeros((wid, hei), dtype=np.uint8)
    for y in range(hei):
        for x in range(wid):
            value = simplex.noise2d(x/feature_size, y/feature_size)
            color = int((value + 1) * 128)
            gray[y, x] = color
    return gray


aug = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0.1, 1))
])


def masks_CFA_Bayer(shape, pattern='RGGB'):
    """
    Returns the *Bayer* CFA red, green and blue masks for given pattern.

    Parameters
    ----------
    shape : array_like
        Dimensions of the *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    tuple
        *Bayer* CFA red, green and blue masks.

    Examples
    --------
    >>> from pprint import pprint
    >>> shape = (3, 3)
    >>> pprint(masks_CFA_Bayer(shape))
    (array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool))
    >>> pprint(masks_CFA_Bayer(shape, 'BGGR'))
    (array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool))
    """

    pattern = pattern.upper()

    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')

def create_CFA_image(shape, patterns='RGGB'):
    '''
    Parameters
    ----------
    shape : array_like
        Dimensions of the *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    CFA image - BGR
    '''
    BRIGHTNESS = random.randint(240, 255)
    all_patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG']
    pattern = random.choice(all_patterns)
    R_m, G_m, B_m = masks_CFA_Bayer(shape, pattern=pattern)
    R_m, G_m, B_m = R_m.astype(np.uint8)*BRIGHTNESS, G_m.astype(np.uint8)*BRIGHTNESS, B_m.astype(np.uint8)*BRIGHTNESS
    CFA_img = cv2.merge((B_m, G_m, R_m))
    return CFA_img


def gen(index):
    path = dataset[0]
    image = cv2.imread(path)
    image = cv2.imread('/home/geneous/Downloads/random/MicrosoftTeams-image.png')
    image = cv2.resize(image, (448, 448))
    H, W, _ = image.shape
    raw_image = image.copy()
    # image = np.ones(((448,448, 3)), dtype=np.uint8) * 255
    # dst_H, dst_W = image.shape[:2]
    dst_H, dst_W = 600, 800

    src_pt = np.zeros((4,2), dtype="float32")
    src_pt[0] = [W // 4, H // 4]
    src_pt[1] = [W // 4 * 3, H // 4]
    src_pt[2] = [W // 4 * 3, H // 4 * 3]
    src_pt[3] = [W // 4, H // 4 * 3]

    
    # Hyperparameter
    # gamma = random.uniform(0.4, 1.2)
    gamma = 1.2
    if random.random() < 0.8:
        gap_nl = 2
    else:
        gap_nl = random.randint(2, 4)

    gap_linear = random.randint(3, 7)

    # Vertical parameter
    v_moire = random.randint(1, 2)
    v_skew = [random.randint(2, 10) for _ in range(v_moire)]
    v_type = [random.choice(['f', 'g', 's']) for _ in range(v_moire)]
    v_cont = [random.randint(1, 15) for _ in range(v_moire)]
    v_dev = [random.randint(1, 2) for _ in range(v_moire)]
    # Horizontal parameter
    h_moire= random.randint(1, 2)
    h_skew = [random.randint(2, 10) for _ in range(h_moire)]
    h_type = [random.choice(['f', 'g', 's']) for _ in range(h_moire)]
    h_cont = [random.randint(1, 15) for _ in range(h_moire)]
    h_dev = [random.randint(1, 2) for _ in range(h_moire)]
    # Non-linear parameter
    nl_moire = True
    nl_dir = random.choice(['b', 'v', 'h'])
    nl_type = random.choice(['fixed', 'gaussian', 'sine'])
    if random.random() < 0.8:
        nl_skew = 0
    else:
        nl_skew = random.randint(-20, 20)
    nl_cont = random.randint(1, 5)
    nl_dev = random.randint(1, 20)
    nl_tb = random.uniform(0.05, 0.1)
    nl_lr = random.uniform(0.05, 0.1)

    # print("All parameters:")
    # print("v_moire:", v_moire)
    # print("v_skew:", v_skew)
    # print("v_type:", v_type)
    # print("v_cont:", v_cont)
    # print("v_dev:", v_dev)
    # print("h_moire:", h_moire)
    # print("h_skew:", h_skew)
    # print("h_type:", h_type)
    # print("h_cont:", h_cont)
    # print("h_dev:", h_dev)
    # print("nl_dir:", nl_dir)
    # print("nl_type:", nl_type)
    # print("nl_skew:", nl_skew)
    # print("nl_cont:", nl_cont)
    # print("nl_dev:", nl_dev)
    # print("nl_tb:", nl_tb)
    # print("nl_lr:", nl_lr)
    # print("gamma:", gamma)
    # print("gap:", gap)
    # print(":", )

    recap_module = RecaptureModule(dst_H, dst_W, 
                                   v_moire=v_moire, v_type=v_type, v_skew=v_skew, v_cont=v_cont, v_dev=v_dev,
                                   h_moire=h_moire, h_type=h_type, h_skew=h_skew, h_cont=h_cont, h_dev=h_dev,
                                   nl_moire=nl_moire, nl_dir=nl_dir, nl_type=nl_type, nl_skew=nl_skew,
                                   nl_cont=nl_cont, nl_dev=nl_dev, nl_tb=nl_tb, nl_lr=nl_lr,
                                   gamma=gamma, margins=None, seed=random.randint(1, 1000))

    canvas, canvas_nl, mask = recap_module(image,
                    gap_nl=gap_nl,
                    gap_linear=gap_linear,
                    new_src_pt = src_pt,
                    verbose=False,
                    show_mask=True)

    h, w = canvas.shape[:2]
    mask = cv2.resize(mask, (w,h))
    # Create CFA bayer - LCD sensor simulator
    CFA_img = create_CFA_image(mask.shape[:2])
    mask_bin = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Option 1 - best choice
    CFA_img = cv2.cvtColor(CFA_img, cv2.COLOR_BGR2HSV)
    CFA_img[:,:,2] = mask_bin
    CFA_img = cv2.cvtColor(CFA_img, cv2.COLOR_HSV2BGR)

    # mask_simplex = create_simplex_mask()
    # all_type_interpolate = [cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    # type_interpolate = random.choice(all_type_interpolate)
    # mask_simplex = cv2.resize(mask_simplex, (h, w), type_interpolate)

    # CFA_img = cv2.cvtColor(CFA_img, cv2.COLOR_BGR2HSV)
    # CFA_img[:,:,2] = mask_simplex
    # CFA_img = cv2.cvtColor(CFA_img, cv2.COLOR_HSV2BGR)

    # Option2 - but image looks not good when we did this
    # CFA_img[:,:,0] = CFA_img[:,:,0] * mask_bin
    # CFA_img[:,:,1] = CFA_img[:,:,1] * mask_bin
    # CFA_img[:,:,2] = CFA_img[:,:,2] * mask_bin

    merge = cv2.addWeighted(canvas, 1, CFA_img, 0.6, 0)
    # Merge full image
    ratio_merge = random.uniform(0.7, 1.2)
    merge = cv2.addWeighted(canvas, 0.5, merge, ratio_merge, 0)
    merge = aug.augment_image(merge)

    # Merge part of image
    # TODO

    output_img_path = os.path.join(output_folder, "{}_color_{}.jpg".format(output_img_name, index))
    cv2.imwrite(output_img_path, merge)

    if random.random() < 0.1:
        output_img_path_gray = os.path.join(output_folder, "{}_gray_{}.jpg".format(output_img_name, index))
        cv2.imwrite(output_img_path_gray, canvas_nl)

    return canvas, canvas_nl, merge


def main():
    global type_data
    type_data = "face"

    global output_img_name
    output_img_name = "invalid_moire_device_{}".format(type_data)

    front_dataset_path = "valid_face/valid_face"
    global dataset
    dataset = glob.glob(os.path.join(front_dataset_path, "*"))

    global output_folder
    output_folder = "result_data/{}".format(output_img_name)
    os.makedirs(output_folder, exist_ok=True)

    global simplex
    simplex = OpenSimplex()

    pool = mp.Pool(8)
    output = list(tqdm(pool.imap(gen, range(len(dataset))), total=len(dataset)))
    pool.terminate()

    # for i in tqdm(range(len(dataset))):
    #     canvas, nl_canvas, best_result = gen(i)

        # cv2.namedWindow("canvas", cv2.WINDOW_NORMAL)
        # cv2.imshow("canvas", canvas)

        # cv2.namedWindow("nl_canvas", cv2.WINDOW_NORMAL)
        # cv2.imshow("nl_canvas", nl_canvas)

        # cv2.namedWindow("merge", cv2.WINDOW_NORMAL)
        # cv2.imshow("merge", best_result)
        

        # key = cv2.waitKey(0)
        # if key == 27:
        #     break

if __name__ == "__main__":
    main()
