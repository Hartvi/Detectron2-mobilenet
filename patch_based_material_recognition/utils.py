import numpy as np
import json
from typing import Tuple, Union, List

raw_id_to_catstr = {0: "baking tray", 1: "baking tray", 2: "ball", 3: "ball", 4: "ball", 5: "ball", 6: "blender", 7: "bottle", 8: "bowl", 9: "bowl", 10: "bowl", 11: "bowl", 12: "box", 13: "box", 14: "box", 15: "box", 16: "can", 17: "chopping board", 18: "chopping board", 19: "chopping board", 20: "clamp", 21: "coffee maker", 22: "coffee maker", 23: "cylinder", 24: "dice", 25: "dice", 26: "drill", 27: "food box", 28: "fork", 29: "fork", 30: "fork", 32: "fruit", 31: "fruit", 33: "glass", 34: "glass", 35: "hammer", 36: "kettle", 37: "kettle", 40: "knife", 38: "knife", 39: "knife", 41: "lego", 42: "mug", 43: "mug", 44: "mug", 45: "pan", 46: "pan", 47: "pen", 48: "pill", 49: "plate", 50: "plate", 51: "pot", 52: "scissors", 53: "screwdriver", 54: "soda can", 55: "spatula", 56: "spoon", 57: "spoon", 58: "spoon", 59: "thermos", 60: "thermos", 61: "toaster", 62: "toaster", 64: "wineglass", 63: "wineglass", 65: "wrench"}
raw_id_to_shortened_id = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 4, 10: 4, 11: 4, 12: 5, 13: 5, 14: 5, 15: 5, 16: 6, 17: 7, 18: 7, 19: 7, 20: 8, 21: 9, 22: 9, 23: 10, 24: 11, 25: 11, 26: 12, 27: 13, 28: 14, 29: 14, 30: 14, 32: 15, 31: 15, 33: 16, 34: 16, 35: 17, 36: 18, 37: 18, 40: 19, 38: 19, 39: 19, 41: 20, 42: 21, 43: 21, 44: 21, 45: 22, 46: 22, 47: 23, 48: 24, 49: 25, 50: 25, 51: 26, 52: 27, 53: 28, 54: 29, 55: 30, 56: 31, 57: 31, 58: 31, 59: 32, 60: 32, 61: 33, 62: 33, 64: 34, 63: 34, 65: 35}
shortened_id_to_str = {0: "baking tray", 1: "ball", 2: "blender", 3: "bottle", 4: "bowl", 5: "box", 6: "can", 7: "chopping board", 8: "clamp", 9: "coffee maker", 10: "cylinder", 11: "dice", 12: "drill", 13: "food box", 14: "fork", 15: "fruit", 16: "glass", 17: "hammer", 18: "kettle", 19: "knife", 20: "lego", 21: "mug", 22: "pan", 23: "pen", 24: "pill", 25: "plate", 26: "pot", 27: "scissors", 28: "screwdriver", 29: "soda can", 30: "spatula", 31: "spoon", 32: "thermos", 33: "toaster", 34: "wineglass", 35: "wrench"}


def get_category_weights_from_csb(classes, scores, bboxes, raw):
    """
    Count the number of pixels of each class in classes weighted by their scores/probabilities.
    Args:
        classes: list of categories
        scores: list of probabilities [0,1] for each category
        bboxes: list of 4 tuples for each category

    Returns:
        list of [class0: bbox0 area * prob0, class1: bbox1 area * prob1, ...]
    """
    ret = np.array([0 for _ in range(len(shortened_id_to_str))])
    if raw:
        for c, s, b in zip(classes, scores, bboxes):
            ret[raw_id_to_shortened_id[c]] = get_weight_from_bbox_score(b, s)
    else:
        for c, s, b in zip(classes, scores, bboxes):
            ret[c] = get_weight_from_bbox_score(b, s)
    return ret


# def get_category_points_from_csb(classes, scores, bboxes):
#     """
#
#     Args:
#         classes: category-only detectron output (0-35)
#         scores: scores for each classes[i] [0,1]
#         bboxes: 4 tuple (x1,y1,x2,y2) for each classes[i]
#
#     Returns:
#         num_categories(=36) tuple with normalized score for each class
#     """
#     ret = np.array([0 for _ in range(len(shortened_id_to_str))])
#     for c, s, b in zip(classes, scores, bboxes):
#         ret[raw_id_to_shortened_id[c]] = get_weight_from_bbox_score(b, s)
#     if len(classes) > 0:
#         ret = ret / sum(ret)
#     return ret


def get_weight_from_bbox_score(bbox: Union[Tuple, np.ndarray], score: float):
    """
    Area of bbox * probability
    Args:
        bbox: 4tuple bbox
        score: probability from instance detection

    Returns:
        Area of bbox * probability
    """
    return (bbox[2] - bbox[0])*(bbox[3] - bbox[1])*score


def gpu_to_numpy(item):
    return item.cpu().numpy()


if __name__ == "__main__":
    with open("id_to_OWN.json", "r") as f:
        id_dictionary = json.load(f)
        vals = id_dictionary.values()


