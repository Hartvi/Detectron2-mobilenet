import numpy as np
import json

# class CustomArray(np.array):
#     pass
raw_id_to_catstr = {0: "baking tray", 1: "baking tray", 2: "ball", 3: "ball", 4: "ball", 5: "ball", 6: "blender", 7: "bottle", 8: "bowl", 9: "bowl", 10: "bowl", 11: "bowl", 12: "box", 13: "box", 14: "box", 15: "box", 16: "can", 17: "chopping board", 18: "chopping board", 19: "chopping board", 20: "clamp", 21: "coffee maker", 22: "coffee maker", 23: "cylinder", 24: "dice", 25: "dice", 26: "drill", 27: "food box", 28: "fork", 29: "fork", 30: "fork", 32: "fruit", 31: "fruit", 33: "glass", 34: "glass", 35: "hammer", 36: "kettle", 37: "kettle", 40: "knife", 38: "knife", 39: "knife", 41: "lego", 42: "mug", 43: "mug", 44: "mug", 45: "pan", 46: "pan", 47: "pen", 48: "pill", 49: "plate", 50: "plate", 51: "pot", 52: "scissors", 53: "screwdriver", 54: "soda can ", 55: "spatula", 56: "spoon ", 57: "spoon", 58: "spoon", 59: "thermos", 60: "thermos", 61: "toaster", 62: "toaster", 64: "wineglass", 63: "wineglass", 65: "wrench"}
raw_id_to_shortened_id = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 4, 10: 4, 11: 4, 12: 5, 13: 5, 14: 5, 15: 5, 16: 6, 17: 7, 18: 7, 19: 7, 20: 8, 21: 9, 22: 9, 23: 10, 24: 11, 25: 11, 26: 12, 27: 13, 28: 14, 29: 14, 30: 14, 32: 15, 31: 15, 33: 16, 34: 16, 35: 17, 36: 18, 37: 18, 40: 19, 38: 19, 39: 19, 41: 20, 42: 21, 43: 21, 44: 21, 45: 22, 46: 22, 47: 23, 48: 24, 49: 25, 50: 25, 51: 26, 52: 27, 53: 28, 54: 29, 55: 30, 56: 31, 57: 32, 58: 32, 59: 33, 60: 33, 61: 34, 62: 34, 64: 35, 63: 35, 65: 36}


class ImageInfos:
    def __init__(self, im_files):
        # im_files, mobile_inputs, mobile_input_selection, detectron_outputs = middle_step
        self.infos = list()
        # print(len(detectron_outputs))
        # print(len(im_files))
        for im_name in im_files:
            info = ImageInfo(im_name)
            self.infos.append(info)

    def update_with_detectron_outputs(self, detectron_outputs):
        for info, detectron_output in zip(self.infos, detectron_outputs):
            info.bboxes = np.array(detectron_output.pred_boxes.tensor.cpu())
            info.raw_categories = np.array(detectron_output.pred_classes.cpu())

    def update_with_mobile_input_selection(self, mobile_input_selection):
        for info, selection in zip(self.infos, mobile_input_selection):
            info.mobile_selection = selection
            info.bboxes_w_material = info.bboxes[info.mobile_selection]
            info.raw_categories_w_material = info.raw_categories[info.mobile_selection]

    def update_with_mobile_outputs(self, material_outputs):
        for info, material_output in zip(self.infos, material_outputs):
            tmp_mat_out = list()
            for i in range(len(material_output)):
                tmp_mat_out.append(tuple(sorted(material_output[i], key=lambda x: x[1], reverse=True)))
            info.materials = tuple(tmp_mat_out)
            # print(info.name, ":", np.array(info.materials).shape)


class ImageInfo:
    def __init__(self, name):
        self.name = name
        self.bboxes: np.ndarray = np.array([])
        self.scores: np.ndarray = np.array([])
        self.raw_categories: np.ndarray = np.array([])
        self.mobile_selection: np.ndarray = np.array([])
        # shortened parameters: len
        self.materials: np.ndarray = np.array([])
        self.bboxes_w_material: np.ndarray = np.array([])
        self.raw_categories_w_material: np.ndarray = np.array([])

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
            temp += "\n    material: " + str(self.materials[i]) + "\n"
            ret += temp
        return ret


if __name__ == "__main__":
    with open("id_to_OWN.json", "r") as f:
        id_dictionary = json.load(f)
        vals = id_dictionary.values()


