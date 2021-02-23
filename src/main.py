import cv2
import numpy as np

from image_tools import *
from moire import linear_wave, dither
from basic_shapes import circles, radialShape
from module import RecaptureModule
import matplotlib.pyplot as plt

import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()

    # File I/O
    # parser.add_argument("--datapath", default='../data/sample_images',
    parser.add_argument("--datapath", default='./',
                        help="Path to the directory containing the source image.")
    parser.add_argument("--file", default='/home/geneous/Downloads/xxx (1).png',
                        help="Name of the source image file.")
    parser.add_argument("--savepath", default='../data/output',
                        help="Path to the output storage directory \
                                (automatically generated if not yet there).")
    parser.add_argument("--save", type=str, default=None,
                        help="Name of the output file storing the results \
                                (not saved if not provided).")
    parser.add_argument("--save-format", type=str, default='jpg',
                        help="File format of the output file (default: JPEG).")
    parser.add_argument("--show_mask", type=bool, default=True,
                        help="Show mask")

    # Image-related
    parser.add_argument("--canvas-dim", type=int, nargs='+', default=1024,
                        help="Dimensions (height, width) of the canvas to use. \
                                Provide a single value to produce a square canvas.")
    parser.add_argument('-e', "--empty", action='store_true',
                        help="Create a white blank canvas, instead of using an image")
    parser.add_argument('-g', "--gamma", type=float, default=0.5,
                        help="Do gamma correction on the given input (default: 1 => no correction)")
    parser.add_argument('-t', "--type", type=str, default='fixed',
                        help="Type of pattern to generate.")
    parser.add_argument('-rv', "--recapture-verbose", action='store_true',
                        help="Print the log of progress produced as \
                                RecaptureModule transforms the input image.")
    parser.add_argument("--psnr", action='store_true',
                        help="Compute the PSNR value of the output image.")

    # Others
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed value for 'np.random.seed'.")
    parser.add_argument('-m', "--show-mask", action='store_true',
                        help="Visualize the inserted nonlinear moire mask.")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.empty:
        canvas_dim = args.canvas_dim if type(args.canvas_dim) == list else [args.canvas_dim,] * 2
        canvas = np.ones(canvas_dim + [3,], np.uint8) * 255   # blank white image
        original = canvas.copy()
    else:
        canvas = cv2.imread(os.path.join(args.datapath, args.file), cv2.IMREAD_COLOR)
        original = canvas.copy()
    H, W, _ = canvas.shape

    # ================================== Add operations here =====================================
    #dst_H = 600; dst_W = 800
    dst_H, dst_W, _ = original.shape

    src_pt = np.zeros((4,2), dtype="float32")
    src_pt[0] = [W // 3, H // 3]
    src_pt[1] = [W // 5 * 3, H // 5]
    src_pt[2] = [W // 5 * 3, H // 5 * 3]
    src_pt[3] = [W // 2, H // 5 * 3]

    dst_pt = np.zeros((4,2), dtype="float32")
    dst_pt[0] = [dst_W // 4, dst_H // 4]    # top-left
    dst_pt[1] = [dst_W // 4 * 3, dst_H // 4]   # top-right
    dst_pt[2] = [dst_W // 4 * 3, dst_H // 4 * 3]   # bottom-right
    dst_pt[3] = [dst_W // 4, dst_H // 4 * 3]   # bottom-left

    t_margin = [0, 0.1]
    b_margin = [0, 0.1]
    l_margin = [0, 0.1]
    r_margin = [0, 0.1]
    sample_margins = [t_margin,b_margin,l_margin,r_margin]

    # recap_module = RecaptureModule(dst_H, dst_W,
    #                                v_moire=0, v_type='sg', v_skew=[5, 10], v_cont=2, v_dev=2,
    #                                h_moire=0, h_type='f', h_skew=[5, 10], h_cont=2, h_dev=2,
    #                                nl_moire=True, nl_dir='v', nl_type='sine', nl_skew=0,
    #                                nl_cont=5, nl_dev=2, nl_tb=0.05, nl_lr=0.05,
    #                                gamma=args.gamma, margins=None, seed=args.seed)

    recap_module = RecaptureModule(dst_H, dst_W,
                                   v_moire=4, v_type='g', v_skew=5, v_cont=1, v_dev=1,
                                   h_moire=10, h_type='g', h_skew=5, h_cont=1, h_dev=1,
                                   nl_moire=True, nl_dir='v', nl_type='sine', nl_skew=0,
                                   nl_cont=1, nl_dev=1, nl_tb=0.05, nl_lr=0.05,
                                   gamma=args.gamma, margins=None, seed=args.seed)

    if args.show_mask:
        canvas, nl_mask = recap_module(canvas,
                          new_src_pt = src_pt,
                          verbose=True,
                          show_mask=args.show_mask)
    else:
        canvas = recap_module(canvas,
                    new_src_pt = src_pt,
                    verbose=True,
                    show_mask=args.show_mask)
    ''''''
    # canvas = dither(canvas,gap=10, skew=50, pattern='rgb', color=(255, 255, 255),contrast=100, rowwise=True)

    # circles(canvas, [(H//2,W//2-64),(H//2,W//2+64)], max_rad=H//2,color=lightgray)
    # radialShape(canvas, (H//2-128,W//2-128), 512, 60, thick=2, color=lightgray)
    # radialShape(canvas, (H//2+128,W//2+128), 512, 60, thick=2, color=lightgray)
    ''''''
    # ===========================================================================================
    # plt.imshow(canvas)
    # plt.show()
    # Display result
    cv2.namedWindow("nonlinear mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("modified", cv2.WINDOW_NORMAL)
    cv2.imshow("modified", canvas)
    if args.show_mask:
        cv2.imshow("nonlinear mask", nl_mask)
    if not args.empty and False:        # TODO edit
        cv2.imshow("original", original)
        if original.shape == canvas.shape and args.psnr:
            psnr_val = psnr(canvas,original)
            print("PSNR value: %.4f db" % psnr_val)
    cv2.waitKey(0)
    cv2.imwrite("test.png", canvas)
    # Save output
    # if not args.empty and args.save:
    #     if not os.path.isdir(args.savepath):
    #         os.makedirs(args.savepath)
    #     save_name = args.save + '_g{}'.format(args.gamma)
    #     save_name += '_{:.4f}'.format(psnr_val)
    #     save_name += '.{}'.format(args.save_format)
    #     cv2.imwrite(os.path.join(args.savepath, save_name), canvas)

if __name__ == "__main__":
    main()
