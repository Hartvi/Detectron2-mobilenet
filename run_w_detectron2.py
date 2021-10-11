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

from patch_based_material_recognition.image_utils import ImageInfos
from patch_based_material_recognition.intermediate_data import IntermediateOutputs
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
from patch_based_material_recognition.utils import *
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

    # DETECTRON INSTANCE SECTION
    real_image_names = [pred_dir+"/"+im for im in image_names]
    # save_separate_masks(predictor, real_image_names)
    intermediate_data = detectron2_outputs_to_mobile_inputs(predictor, real_image_names)
    # `im_files`: list of names of input images. Shape: (len(im_files), )
    # `mobile_inputs`: imgs to be fed into mobilenet. Shape: (imlen, number of predicted bboxes, *img_dims)
    # `detectron_outputs`: list of standard detectron outputs. Shape: (len(im_files), )
    im_names = intermediate_data.im_names
    detectron_instances = intermediate_data.outputs  # (number_of_images, number of instances per image)
    mobile_inputs = intermediate_data.inputs.mobile_inputs
    detectron_inputs = intermediate_data.inputs.detectron_inputs
    infos = ImageInfos(len(im_names))
    infos.update_with_im_names(im_names)
    infos.update_with_detectron_outputs(detectron_instances)
    print("Done: detectron instance detection")

    # MOBILE NET MATERIAL SECTION
    # shape of `material_outputs`: (number of images, bboxes per image, materials per bbox)
    material_outputs = get_materials_from_patches(mobile_inputs)
    infos.update_with_mobile_outputs(material_outputs)
    print("Done: material classification")

    sensitive_threshold = 0.10
    detectron_categories = get_detectron_categories(cfg, sensitive_threshold, detectron_inputs, detectron_instances)
    infos.update_with_detectron_categories(detectron_categories)
    print("Done: category classification")
    print(infos)

    # with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "results.txt"), "w") as results:
    #     for im, mat in zip(im_names, material_outputs):
    #         results.write(im+":\n"+str(mat)+"\n")


def get_detectron_categories(cfg, sensitivity, intermediate_outputs, detectron_instances: IntermediateOutputs) -> np.ndarray:
    temp_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = sensitivity
    predictor = DefaultPredictor(cfg)
    detectron_categories = list()
    for k, (ims, detectron_output) in enumerate(zip(intermediate_outputs, detectron_instances)):
        per_im_classes = list()
        for im, instance in zip(ims, detectron_output):
            outputs = predictor(im)
            instances = outputs["instances"]
            classes = gpu_to_numpy(instances.pred_classes)
            scores = gpu_to_numpy(instances.scores)
            bboxes = gpu_to_numpy(instances.pred_boxes.tensor)
            class_weights = get_category_weights_from_csb(classes, scores, bboxes, raw=True)
            # originally predicted by instance detection
            shortened_csb = [[instance.category, ], [instance.score, ], [instance.bbox, ]]
            original_instance_weights = get_category_weights_from_csb(*shortened_csb, raw=False)
            class_points = class_weights + original_instance_weights
            class_points = class_points / sum(class_points)
            per_im_classes.append(np.array(class_points))

        detectron_categories.append(np.array(per_im_classes))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = temp_thresh
    return np.array(detectron_categories)


if __name__ == "__main__":
    main()




