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

from patch_based_material_recognition.detectron2_to_mobilenet import get_materials_from_patches
from patch_based_material_recognition.utils import ImageInfos
from image_to_outputs import detectron2_outputs_to_mobile_inputs


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
    THRESHOLD = 0.6

    # translate threshold into a text form
    buff = ""
    for c in str(THRESHOLD):
        if c == '.':
            c = '_'
        buff += c
    THRESHOLD_TXT = buff

    cfg = setup()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD
    cfg.DATASETS.TEST = (
        "/local/temporary/DATASET/TRAIN",)  # must be a tuple, for the inner workings of detectron2 (stupid, we know :/)
    predictor = DefaultPredictor(cfg)

    # testing image cutouts
    # im = cv2.imread(pred_dir + '/' + image_names[0])  # is numpy.ndarray
    # outputs = predictor(im)
    real_image_names = [pred_dir+"/"+im for im in image_names]
    # save_separate_masks(predictor, real_image_names)
    d2_out = detectron2_outputs_to_mobile_inputs(predictor, real_image_names)
    # `im_files`: list of names of input images. Shape: (len(im_files), )
    # `mobile_inputs`: imgs to be fed into mobilenet. Shape: (imlen, number of predicted bboxes, *img_dims)
    # `mobile_input_selection`: (len(im_files), bools field of len(mobile_inputs) where
    #                                                                      [deformation of mobile_inputs < threshold], )
    # `detectron_outputs`: list of standard detectron outputs. Shape: (len(im_files), )
    im_files, mobile_inputs, mobile_input_selection, detectron_outputs = d2_out
    infos = ImageInfos(im_files=im_files)
    infos.update_with_detectron_outputs(detectron_outputs)
    infos.update_with_mobile_input_selection(mobile_input_selection)

    # shape of `material_outputs`: (number of images, bboxes per image, materials per bbox)
    material_outputs = get_materials_from_patches(get_np_elements_a_layer_deeper(mobile_inputs, mobile_input_selection))
    infos.update_with_mobile_outputs(material_outputs)
    for info in infos.infos:
        print(info)
    # for im, mat in zip(im_files, material_outputs):
    #     print(im, mat)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "results.txt"), "w") as results:
        for im, mat in zip(im_files, material_outputs):
            results.write(im+":\n"+str(mat)+"\n")


def get_np_elements_a_layer_deeper(nparr, indices3D):
    return np.array([nparr[k][ind] for k, ind in enumerate(indices3D)])


if __name__ == "__main__":
    main()




