import argparse
import os

from PIL import Image
import numpy as np
import torch
from typing import Tuple, List

from .dataset_loader import MincDataset, MincLoader, get_free_gpu, category2id, id2category, ipalm_ignore_classes, categories
from .net import MobileNetV3Large

from .test import *


__all__ = ["get_materials_from_patches", "get_materials_from_patch"]


def get_materials_from_patches(patches_list) -> Tuple[Tuple[Tuple[Tuple[int, float]]], ...]:
    # list[image: list[patch:list[tuple[int,float]]], image: list[patch:list[tuple[int,float]]], ...]
    """

    Args:
        patches_list: list of N images, [N, H, W, C]

    Returns:
        (
        i:0  ((material_id, probability), (material_id, probability), ...),
        i:1  ((material_id, probability), (material_id, probability), ...),
        ...
        )
    """
    # ignore non-ipalm materials:
    ipalm_ids = [i for i in range(len(categories)) if categories[i] not in ipalm_ignore_classes]
    model_path = "saved_model.pth"

    model = MobileNetV3Large(n_classes=len(categories))
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("using cpu!")
    model.eval()     # Optional when not using Model Specific layer
    # print(ipalm_ids)
    per_image_materials = list()
    for k, patche_batch in enumerate(patches_list):
        # print(type(image))
        # N, H, W, C
        materials = list()
        for patch in patche_batch:
            material = get_materials_from_patch(model, patch, ipalm_ids)
            materials.append(material)
        per_image_materials.append(tuple(materials))
    return tuple(per_image_materials)


def get_materials_from_patch(model, image, selected_material_ids) -> Tuple[Tuple[int, float]]:
    """

    Args:
        model: classifier network w/ 23 inputs
        image: 362x362 image
        selected_material_ids: list of ids in [0,22] from which to calculate probabilities

    Returns:
        ((material_id, probability), (material_id, probability), ...)
    """
    # print(image.shape)
    data = np.transpose(np.array(image).astype('f4'), (2, 0, 1)) / 255.0
    data = torch.from_numpy(data)
    data.unsqueeze_(0)
    if torch.cuda.is_available():
        data = data.cuda()
    target = model(data)
    probs = get_probabilities_from_selection(target, selected_material_ids)
    class_probs = tuple(i for i in zip(map_from_selection(selected_material_ids, id2category), to_numpy_cpu(probs)))
    return tuple(get_classes_above_threshold(class_probs))

