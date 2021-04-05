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


def gen(index):
    path = dataset[index]
    image = cv2.imread(path)
    H, W, _ = image.shape
    # dst_H, dst_W = image.shape[:2]
    dst_H, dst_W = 600, 800

    src_pt = np.zeros((4,2), dtype="float32")
    src_pt[0] = [W // 4, H // 4]
    src_pt[1] = [W // 4 * 3, H // 4]
    src_pt[2] = [W // 4 * 3, H // 4 * 3]
    src_pt[3] = [W // 4, H // 4 * 3]

    
    # Hyperparameter
    v_skew = random.randint(2, 10)
    v_type = random.choice(['f', 'g', 's'])
    v_cont = random.randint(1, 10)
    v_dev = random.randint(1, 2)

    h_skew = random.randint(2, 10)
    h_type = random.choice(['f', 'g', 's'])
    h_cont = random.randint(1, 10)
    h_dev = random.randint(1, 2)

    nl_moire = True
    if random.random() < 0.02:
        nl_moire = False
    nl_dir = random.choice(['v', 'b', 'h'])
    nl_type = random.choice(['fixed', 'gaussian', 'sine'])
    nl_skew = random.randint(1, 3)
    nl_cont = random.randint(1, 5)
    nl_dev = random.randint(1, 2)
    gamma = random.uniform(0.4, 0.6)
    
    gap = random.randint(5, 10)

    recap_module = RecaptureModule(dst_H, dst_W,
                                   v_moire=1, v_type=v_type, v_skew=v_skew, v_cont=v_cont, v_dev=v_dev,
                                   h_moire=1, h_type=h_type, h_skew=h_skew, h_cont=h_cont, h_dev=v_dev,
                                   nl_moire=nl_moire, nl_dir=nl_dir, nl_type=nl_dir, nl_skew=nl_skew,
                                   nl_cont=nl_cont, nl_dev=nl_dev, nl_tb=0.05, nl_lr=0.05,
                                   gamma=gamma, margins=None, seed=random.randint(1, 1000))

    canvas = recap_module(image,
                    gap=gap,
                    new_src_pt = src_pt,
                    verbose=False,
                    show_mask=False)
    
    output_img_path = os.path.join(output_folder, "{}_{}.jpg".format(output_img_name, index))
    cv2.imwrite(output_img_path, canvas)
    
    # cv2.namedWindow("", cv2.WINDOW_NORMAL)
    # cv2.imshow("", canvas)
    # cv2.waitKey(0)


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

    # pool = mp.Pool(8)
    # output = list(tqdm(pool.imap(gen, range(len(dataset))), total=len(dataset)))
    # pool.terminate()

    for i in tqdm(range(len(dataset))):
        gen(i)
        if i == 10:
            break



if __name__ == "__main__":
    main()
