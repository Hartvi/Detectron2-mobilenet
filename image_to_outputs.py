import os
import cv2
import tqdm
import json
import random
import pickle
import logging
import detectron2
import pycocotools
import torch, torchvision
from collections import OrderedDict
from train import setup
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import model_zoo
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from predictor import VisualizationDemo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

import numpy as np
from PIL import Image

# __all__ = []


def detectron2_outputs_to_mobile_inputs(predictor, image_names):
    # ret = list()
    im_names = list()
    mobile_inputs = list()
    detectron_outputs = list()
    for cnt, im_name in enumerate(image_names):
        # path = pred_dir + '/' + im_name
        im = cv2.imread(im_name)
        output = predictor(im)
        output = output["instances"].to("cpu")
        # """outputs["instances"]
        #    members: _image_size 2tuple, pred_boxes 4tuple,
        #    scores n-tuple, pred_classes n-tuple, pred_masks n-list of images"""
        # square_outputs, deformations = get_mobilenet_input(im, output)
        inputs = detectron2_output_to_mobile_input(im, output)  # this converts BGR to RGB
        im_names.append(im_name)
        mobile_inputs.append(inputs)
        detectron_outputs.append(output)
    return im_names, mobile_inputs, detectron_outputs


def detectron2_output_to_mobile_input(im, output, padding=30, outsize=362, max_deformation=3):
    """outputs["instances"]
       members: _image_size 2tuple, pred_boxes 4tuple,
       scores n-tuple, pred_classes n-tuple, pred_masks n-list of images"""
    # Boxes.tensor: (x1, y1, x2, y2)
    # outputs = outputs["instances"].to("cpu")
    # outputs should be in this format /\
    assert im.shape[2] == 3, "Image colour channels must be last"
    mobile_net_images = list()
    selected_indices = list()
    height, width = im.shape[0:2]
    print("before conversion to mobile inputs:", len(output.pred_masks))
    for i in range(len(output.pred_masks)):
        x1, y1, x2, y2 = pred_box_to_bounding_box(output.pred_boxes[i])
        # there might be discrepancies in the image size
        #   1. +-1 pixel
        #   2. some amount that was cut off due to the square reaching outside of the image
        x1, y1, x2, y2 = get_padded_extended_bbox(padding, width, height, x1, y1, x2, y2)
        dx = x2-x1
        dy = y2-y1
        deformation = round(float(deformation_score(dx, dy, outsize*outsize)), 2)
        if deformation < max_deformation:
            sub_image = im[y1:y2, x1:x2, ::-1]  # cut out instance bounding box, bgr => rgb
            resized_image = cv2.resize(sub_image, dsize=(outsize, outsize), interpolation=cv2.INTER_CUBIC)
            mobile_net_images.append(resized_image)
            # print("w:", sub_image.shape[1], "h:", sub_image.shape[0], "targetsize", outsize, deformation_score(sub_image.shape[1], sub_image.shape[0], outsize**2))
            # print(type(sub_image))
            # out_im = Image.fromarray(resized_image)
            # out_im.save("bbox_vis-{}-{}_{}_{}_{}.png".format(i, x1, y1, x2, y2))
            # out_im.save("bbox_vis-{}-{}_{}_{}_{}.png".format(deformation, x1, y1, x2, y2))
            # print(sub_image.shape)
            # for j in range(len(out_im)):
            #     for k in range(len(out_im[0])):
            #         out_im[j, k] = np.array(list(255 if outputs.pred_masks[i, j, k] else 0 for l in range(3)))
    return np.array(mobile_net_images),

# def overlap1D(xmin1, xmin2, xmax1, xmax2):
#     return xmax1 >= xmin2 and xmax2 >= xmin1
#
# def overlap2D(box1, box2):

def get_mobilenet_input(im, output, padding=30, outsize=362, max_deformation=3):
    """outputs["instances"]
       members: _image_size 2tuple, pred_boxes 4tuple,
       scores n-tuple, pred_classes n-tuple, pred_masks n-list of images"""
    # Boxes.tensor: (x1, y1, x2, y2)
    # outputs = outputs["instances"].to("cpu")
    # outputs should be in this format /\
    assert im.shape[2] == 3, "Image colour channels must be last"
    mobile_net_images = list()
    height, width = im.shape[0:2]
    square_ims = list()
    deformations = list()
    scores = list()
    """
    box1 = (xmin1, xmax1)
    box2 = (xmin2, xmax2)
    isOverlapping1D(box1, box2) = xmax1 >= xmin2 and xmax2 >= xmin1

    box1 = (x:(xmin1, xmax1), y:(ymin1, ymax1))
    box2 = (x:(xmin2, xmax2), y:(ymin2, ymax2))
    isOverlapping2D(box1, box2) = isOverlapping1D(box1.x, box2.x) and
    isOverlapping1D(box1.y, box2.y)
    """
    for i in range(len(output.pred_masks)):
        x1, y1, x2, y2 = pred_box_to_bounding_box(output.pred_boxes[i])
        x1, y1, x2, y2 = get_padded_extended_bbox(padding, width, height, x1, y1, x2, y2)
        dx = x2-x1
        dy = y2-y1
        deformation = round(float(deformation_score(dx, dy, outsize*outsize)), 2)
        print(output.pred_classes[i], "gonna get saved:", deformation < max_deformation)
        if deformation < max_deformation:
            sub_image = im[y1:y2, x1:x2, ::-1]  # cut out instance bounding box, bgr => rgb
            resized_image = cv2.resize(sub_image, dsize=(outsize, outsize), interpolation=cv2.INTER_CUBIC)
            square_ims.append(resized_image)
            deformations.append(deformation)
            # scores.append(score)
            # print("w:", sub_image.shape[1], "h:", sub_image.shape[0], "targetsize", outsize, deformation_score(sub_image.shape[1], sub_image.shape[0], outsize**2))
            # out_im = Image.fromarray(resized_image)
            # out_im.save("bbox_vis-{}-{}_{}_{}_{}.png".format(deformation, x1, y1, x2, y2))
    return square_ims, deformations


def detectron2_output_2_mask(output):
    """outputs["instances"]
       members: _image_size 2tuple, pred_boxes 4tuple,
       scores n-tuple, pred_classes n-tuple, pred_masks n-list of images"""
    (pred_num, height, width, channels) = (len(output.pred_masks), *output._image_size, 3)
    out_shape = (pred_num, height, width, channels)
    out_ims = np.zeros(out_shape, dtype=np.uint8)
    for i in range(len(output.pred_masks)):
        out_im = out_ims[i]
        for j in range(len(out_im)):
            for k in range(len(out_im[0])):
                out_im[j, k] = np.array(list(255 if output.pred_masks[i, j, k] else 0 for _ in range(3)))
    return out_ims


def save_interstage_io(predictor, image_names):
    for cnt, im_name in enumerate(image_names):
        # path = pred_dir + '/' + im_name
        im = cv2.imread(im_name)
        output = predictor(im)
        output = output["instances"].to("cpu")
        # """outputs["instances"]
        #    members: _image_size 2tuple, pred_boxes 4tuple,
        #    scores n-tuple, pred_classes n-tuple, pred_masks n-list of images"""
        square_outputs, deformations = get_mobilenet_input(im, output)
        for i, sq in enumerate(square_outputs):
            out_im = Image.fromarray(sq)
            out_im.save("im-{}-bbox_vis-{}-deform-{}.png".format(cnt, i, deformations[i]))
            print("saved im-{}-bbox_vis-{}-deform-{}.png".format(cnt, i, deformations[i]))
        masks = detectron2_output_2_mask(output)
        for i, mask in enumerate(masks):
            im = Image.fromarray(mask)
            im.save("im-{}-mask-{}.png".format(cnt, i))
            print("saved im-{}-mask-{}.png".format(cnt, i))

        print("visualized interstage for image",str(cnt)+"/"+str(len(image_names)))


def deformation_score(w1, h1, w2h2):
    # w1=100, h1=400, w2=362, h2=362 =>
    # w1=362, h1=1448, w2=362, h2=362 =>
    # (w2h2 / (w1 * h1)) * max(w1 / h1, h1 / w1) = 0.25*4 = 1
    return (w2h2 / (w1 * h1)) * max(w1 / h1, h1 / w1)


def get_padded_extended_bbox(padding, w, h, x1, y1, x2, y2):
    return extend_to_square(w, h, *get_padded_bbox(padding, w, h, x1, y1, x2, y2))


def get_padded_bbox(padding, w, h, x1, y1, x2, y2):
    return max(x1-padding, 0), max(y1-padding, 0), min(x2+padding, w), min(y2+padding, h)


def extend_to_square(w, h, x1, y1, x2, y2):
    dx = x2-x1
    dy = y2-y1
    if dy > dx:
        x1 = max(x1 - (dy-dx)//2, 0)
        x2 = min(x2 + (dy-dx)//2, w)
    else:
        y1 = max(y1 - (dx-dy)//2, 0)
        y2 = min(y2 + (dx-dy)//2, h)
    return x1, y1, x2, y2


def visualize_bounding_boxes(im, outputs):
    assert im.shape[2] == 3, "Image colour channels must be last"
    for i in range(len(outputs.pred_masks)):
        x1, y1, x2, y2 = pred_box_to_bounding_box(outputs.pred_boxes[i])
        sub_image = im[y1:y2, x1:x2, ::-1]  # cut out instance bounding box, bgr => rgb
        out_im = Image.fromarray(sub_image)
        out_im.save("bbox_vis-{}-{}_{}_{}_{}.png".format(i, x1, y1, x2, y2))


def get_sub_images_detectron2(im, outputs):
    assert im.shape[2] == 3, "Image colour channels must be last"
    ret = list()
    for i in range(len(outputs.pred_masks)):
        x1, y1, x2, y2 = outputs.pred_boxes[i].tensor.int()[0]
        sub_image = im[y1:y2, x1:x2, ::-1]  # cut out instance bounding box, bgr => rgb
        ret.append(sub_image)
    return ret


def pred_box_to_bounding_box(pred_box):
    x1, y1, x2, y2 = pred_box.tensor.int()[0]
    return x1, y1, x2, y2


def main():
    with open("/local/temporary/DATASET/info/id_to_OWN.json") as json_labels:
        new_dict = json.load(json_labels)

    # PATHS SETUP
    trn_data = '/local/temporary/DATASET/TRAIN'
    pred_dir = "images_input"
    img_output = "images_output"

    # video is having problems :/
    vid_dir = "videos"
    output = "videos_output"

    # lists all images to image_names list from the dictionary using os.walk
    image_names = []
    for (dirpath, dirnames, filenames) in os.walk(pred_dir):
        image_names.extend(filenames)
        break

    video_names = []
    for (dirpath, dirnames, filenames) in os.walk(vid_dir):
        video_names.extend(filenames)
        break

    # orders annotations for the METADATA
    ordered_list_of_names = []
    for i in range(len(new_dict)):
        ordered_list_of_names.append(new_dict[str(i)])

    ycb_metadata = MetadataCatalog.get(trn_data)

    # load annotations from any of the datasets (also train.data or val.data should work)
    def get_test_dict():
        with open("/local/temporary/DATASET/test.data", 'rb') as data:
            data = pickle.load(data)
        return data

    # choose the certainity threshold
    THRESHOLD = 0.2

    # translate threshold into a text form
    buff = ""
    for c in str(THRESHOLD):
        if c == '.':
            c = '_'
        buff += c
    THRESHOLD_TXT = buff

    cfg = setup()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # print(cfg.MODEL.WEIGHTS)
    # exit()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD
    cfg.DATASETS.TEST = (
    "/local/temporary/DATASET/TRAIN",)  # must be a tuple, for the inner workings of detectron2 (stupid, we know :/)
    predictor = DefaultPredictor(cfg)

    # testing image cutouts
    # im = cv2.imread(pred_dir + '/' + image_names[0])  # is numpy.ndarray
    # outputs = predictor(im)
    # detectron2_output_to_mobile_input(im, outputs)
    real_image_names = [pred_dir+"/"+im for im in image_names]
    # save_separate_masks(predictor, real_image_names)
    save_interstage_io(predictor, real_image_names)

    # # # ------------IMAGE PREDICTION-------------------
    # for cnt, im_name in enumerate(image_names):
    #     path = pred_dir + '/' + im_name
    #     im = cv2.imread(path)
    #     output = predictor(im)
    #     output = outputs["instances"].to("cpu")
    #     detectron2_output_to_mobile_input(im, output)


if __name__ == "__main__":
    main()
