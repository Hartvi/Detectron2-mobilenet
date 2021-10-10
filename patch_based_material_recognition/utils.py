import numpy as np
import json
from typing import Tuple, Union, List
import collections

# class CustomArray(np.array):
#     pass
raw_id_to_catstr = {0: "baking tray", 1: "baking tray", 2: "ball", 3: "ball", 4: "ball", 5: "ball", 6: "blender", 7: "bottle", 8: "bowl", 9: "bowl", 10: "bowl", 11: "bowl", 12: "box", 13: "box", 14: "box", 15: "box", 16: "can", 17: "chopping board", 18: "chopping board", 19: "chopping board", 20: "clamp", 21: "coffee maker", 22: "coffee maker", 23: "cylinder", 24: "dice", 25: "dice", 26: "drill", 27: "food box", 28: "fork", 29: "fork", 30: "fork", 32: "fruit", 31: "fruit", 33: "glass", 34: "glass", 35: "hammer", 36: "kettle", 37: "kettle", 40: "knife", 38: "knife", 39: "knife", 41: "lego", 42: "mug", 43: "mug", 44: "mug", 45: "pan", 46: "pan", 47: "pen", 48: "pill", 49: "plate", 50: "plate", 51: "pot", 52: "scissors", 53: "screwdriver", 54: "soda can", 55: "spatula", 56: "spoon", 57: "spoon", 58: "spoon", 59: "thermos", 60: "thermos", 61: "toaster", 62: "toaster", 64: "wineglass", 63: "wineglass", 65: "wrench"}
raw_id_to_shortened_id = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 4, 10: 4, 11: 4, 12: 5, 13: 5, 14: 5, 15: 5, 16: 6, 17: 7, 18: 7, 19: 7, 20: 8, 21: 9, 22: 9, 23: 10, 24: 11, 25: 11, 26: 12, 27: 13, 28: 14, 29: 14, 30: 14, 32: 15, 31: 15, 33: 16, 34: 16, 35: 17, 36: 18, 37: 18, 40: 19, 38: 19, 39: 19, 41: 20, 42: 21, 43: 21, 44: 21, 45: 22, 46: 22, 47: 23, 48: 24, 49: 25, 50: 25, 51: 26, 52: 27, 53: 28, 54: 29, 55: 30, 56: 31, 57: 31, 58: 31, 59: 32, 60: 32, 61: 33, 62: 33, 64: 34, 63: 34, 65: 35}
shortened_id_to_str = {0: "baking tray", 1: "ball", 2: "blender", 3: "bottle", 4: "bowl", 5: "box", 6: "can", 7: "chopping board", 8: "clamp", 9: "coffee maker", 10: "cylinder", 11: "dice", 12: "drill", 13: "food box", 14: "fork", 15: "fruit", 16: "glass", 17: "hammer", 18: "kettle", 19: "knife", 20: "lego", 21: "mug", 22: "pan", 23: "pen", 24: "pill", 25: "plate", 26: "pot", 27: "scissors", 28: "screwdriver", 29: "soda can", 30: "spatula", 31: "spoon", 32: "thermos", 33: "toaster", 34: "wineglass", 35: "wrench"}


class TypedList(collections.MutableSequence):

    def __init__(self, oktypes, *args):
        self.oktypes = oktypes
        self.list = list()
        self.extend(list(args))

    def check(self, v):
        if not isinstance(v, self.oktypes):
            raise TypeError(v)

    def __len__(self): return len(self.list)

    def __getitem__(self, i): return self.list[i]

    def __delitem__(self, i): del self.list[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def __str__(self):
        return str(self.list)


class SimpleInstance:
    """
    Detected instance inside an image
    > processes like so: output.pred_classes, output.scores, output.pred_boxes.tensor
    """
    def __init__(self, output):
        self.categories = output.pred_classes
        self.bboxes = output.pred_boxes.tensor
        self.scores = output.scores


class IntermediateOutput:
    """
    Contains all instances for one image
    """
    def __init__(self, simple_output: List[SimpleInstance]):
        self.simple_output = simple_output

    def __getitem__(self, item):
        return self.simple_output[item]

    def __iter__(self):
        yield from self.simple_output


class IntermediateOutputs:
    """
    Contains intermediate outputs for a list of images
    """
    def __init__(self, intermediate_output: List[IntermediateOutput]):
        self.intermediate_output = intermediate_output

    def __getitem__(self, item):
        return self.intermediate_output[item]

    def __iter__(self):
        yield from self.intermediate_output


class IntermediateInput:
    """
    Contains intermediate inputs for one image
    """
    def __init__(self, intermediate_input):
        self.intermediate_input = intermediate_input

    def __getitem__(self, item):
        return self.intermediate_input[item]

    def __iter__(self):
        yield from self.intermediate_input


class IntermediateInputs:
    """
    Contains a list of intermediate inputs for a list of images
    """
    def __init__(self, mobile_inputs: List[IntermediateInput], detectron_inputs: List[IntermediateInput]):
        self.mobile_inputs = mobile_inputs
        self.detectron_inputs = detectron_inputs
        self.input_zip = tuple(zip(self.mobile_inputs, self.detectron_inputs))

    def __getitem__(self, item):
        return self.input_zip[item]

    def __iter__(self):
        yield from self.input_zip


class IntermediateData:
    """
    Contains a list of intermediate outputs for a list of images.
    And: a list of intermediate inputs for the same list of images.
    """
    def __init__(self, im_names: List[str], outputs: IntermediateOutputs, inputs: IntermediateInputs):
        self.im_names = im_names
        self.outputs = outputs
        self.inputs = inputs


def get_category_points_from_csb(classes, scores, bboxes):
    """

    Args:
        classes: category-only detectron output (0-35)
        scores: scores for each classes[i] [0,1]
        bboxes: 4 tuple (x1,y1,x2,y2) for each classes[i]

    Returns:
        num_categories(=36) tuple with normalized score for each class
    """
    ret = np.array([0 for _ in range(len(shortened_id_to_str))])
    for c, s, b in zip(classes, scores, bboxes):
        ret[raw_id_to_shortened_id[c]] = get_weight_from_bbox_score(b, s)
    if len(classes) > 0:
        ret = ret / sum(ret)
    return ret


def get_weight_from_bbox_score(bbox: Union[Tuple, np.ndarray], score: float):
    return (bbox[2] - bbox[0])*(bbox[3] - bbox[1])*score


def gpu_to_numpy(item):
    return item.cpu().numpy()


class ImageInfos:
    def __init__(self, num):
        # im_files, mobile_inputs, mobile_input_selection, detectron_outputs = middle_step
        self.infos = list()
        for i in range(num):
            self.infos.append(ImageInfo())

    def update_with_im_names(self, im_names):
        for k, im_name in enumerate(im_names):
            self.infos[k].name = im_name

    def update_with_detectron_outputs(self, simple_outputs: IntermediateOutputs):
        for info, output in zip(self.infos, simple_outputs):
            info.detectron_output = output  # iterable

    # def update_with_mobile_input_selection(self, mobile_input_selection):
    #     for info, selection in zip(self.infos, mobile_input_selection):
            # info.mobile_selection = selection
            # info.bboxes_w_material = info.bboxes[info.mobile_selection]
            # info.raw_categories_w_material = info.raw_categories[info.mobile_selection]

    def update_with_mobile_outputs(self, material_outputs):
        for info, material_output in zip(self.infos, material_outputs):
            tmp_mat_out = list()
            for i in range(len(material_output)):
                tmp_mat_out.append(sorted(material_output[i], key=lambda x: x[1], reverse=True))
            # print(type(tmp_mat_out[0][0][1]))
            info.materials = tmp_mat_out
            # print(info.name, ":", np.array(info.materials).shape)

    def update_with_detectron_categories(self, detectron_classes):
        """

        Args:
            detectron_classes: 3D array. Shape: (num_input_ims, instances_per_image, num of categories (36))
        """
        for info, categories in zip(self.infos, detectron_classes):
            info.categories = np.array(categories)

    def __getitem__(self, item):
        return self.infos[item]

    def __iter__(self):
        yield from self.infos


class ImageInfo:
    def __init__(self):
        self.name: str = ""
        # self.bboxes: np.ndarray = np.array([])
        # self.scores: np.ndarray = np.array([])
        # self.raw_categories: np.ndarray = np.array([])
        # self.mobile_selection: np.ndarray = np.array([])
        # shortened parameters: len
        self.materials: np.ndarray = np.array([])
        self.detectron_output = None
        # self.bboxes_w_material: np.ndarray = np.array([])
        # self.raw_categories_w_material: np.ndarray = np.array([])
        self.categories: np.ndarray = np.array([])

    def __repr__(self):
        ret = "name: "+self.name+"\n"
        # print(len(self.bboxes))
        # print(len(self.materials))
        # print(len(self.bboxes_w_material))
        for i in range(len(self.bboxes_w_material)):
            temp = f"  item{i+1}: \n"
            temp += "    class: " + raw_id_to_catstr[self.raw_categories_w_material[i]]
            temp += f"\n    bbox: "
            temp += str([int(ffff) for ffff in self.bboxes_w_material[i]])
            temp += "\n    material: "
            materials = ""
            for mat in self.materials[i]:
                # print(type(mat[1]))
                materials += "\n      " + mat[0] + ": " + str(round(mat[1], 4))
            temp += materials
            temp += "\n    category distribution: "
            categories = "\n"
            sorted_categories = np.argsort(self.categories[i])[::-1]
            for prob_i in sorted_categories:
                if self.categories[i][prob_i] != 0.0:
                    categories += "      " + shortened_id_to_str[prob_i] + ": " + str(round(self.categories[i][prob_i], 4)) + "\n"
            temp += categories
            temp += "\n"
            ret += temp
        return ret


if __name__ == "__main__":
    with open("id_to_OWN.json", "r") as f:
        id_dictionary = json.load(f)
        vals = id_dictionary.values()


