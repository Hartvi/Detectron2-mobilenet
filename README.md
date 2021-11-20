Part of the IPALM project, this is a fusion of a MobilenetV3 trained on the smaller MINC2500 dataset and the default Detectron2 InstanceSegmentor trained on COCO, [ShopVRB](https://michaal94.github.io/SHOP-VRB/), [YCB](https://www.ycbbenchmarks.com/) and a few custom images.

### Project structure
The base of the project is the detectron2 framework's instance segmentation backbone by [facebookresearch](https://github.com/facebookresearch/detectron2). 
Initially we tried to add a material segmentation/classification head to the instance segmentation network, however that proved to be exceedingly confusing because of the structure of the project. See the [gitlab progress log](https://gitlab.fel.cvut.cz/body-schema/ipalm/ipalm-vir2020-object-category-from-image/-/blob/master/code/PROGRESS.md) for details on how it was unfeasible (for me lol) given our constraints. Basically the detectron2's structure is such that I couldn't even discover any modifications of it **not made by facebook employees**.

We ended up using just the default instance segmentor from VIR, retrained because something in either a newer detectron2 or pytorch version changed something in the background. The bounding boxes gathered from detectron2 are then plugged back into detectron2 this time used as a category classifier and into a MobileNetV3 material classifier trained on the [MINC 2500](http://opensurfaces.cs.cornell.edu/publications/minc/) dataset.

# TODO: 
- insert images explaining the information flow
- short summaries of the ipalm project files

## Information flow
The high-level structure of the project is the following. The input image is fed into Detectron2 which is first used to locate objects of interest and its output data is saved. The bounding boxes gained from the first pass are extracted and plugged into Detectron2 (again) and also into MobileNet.
<div align=center>
    <img src="https://i.imgur.com/JcbV39e.png" alt="drawing" width="500"/><br>
    Figure 1. Information flow in project structure.
</div>
<br>

The following picture contains an explanation how categories are weighted. There are in total 2 passes of each bounding box through detectron. Therefore there is 1 bounding box that is then plugged back into detectron to get some more bounding boxes. The weight of the class initially detected by detectron is then `area_of_first_bbox*first_probability` + `area_of_nested_bbox*second_probability`. The weights of other classes are simply just `area_of_nested_bbox_of_class_i*second_probability_of_class_i`.


<div align=center>
    <img src="https://i.imgur.com/IpxOxNd.png" alt="drawing" width="500"/><br>
    Figure 2. Category and material probability calculation.
</div>

### Prerequisites
- Versions of packages used:
  - `torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0`
  - `OpenCV/3.4.8-fosscuda-2019b-Python-3.7.4`
  - `scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4`
  - `scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4`
  - and dependencies

### How to install:
1. Go to some folder A and: `git clone https://github.com/Hartvi/Detectron2-mobilenet`
    - This will create the folder called `Detectron2-mobilenet
2. Rename `Detectron2-mobilenet` to `detectron2`
3. In folder A: `python -m pip install -e detectron2 --user`
4. ???
5. Profit 1 billion dollars


### Short demo:

```
from detectron2 import andrej_logic

megapredictor = CatmatPredictor(threshold=0.6)
# folder with images: "images_input/[some_images]"
input_imgs = ["images_input/" + f for f in listdir("images_input") if isfile(join("images_input", f))]
# CatMatPredictor.get_andrej(raw_image[arr]/image_path[str]) returns a list of dictionaries for 
for inp_img in input_imgs:
    # this is a list of dicts in andrej format, see ipalm/andrej_output_format
    # optional argument: output_target="your_file_name.json" to save the dicts in json format
    predictions = megapredictor.get_andrej(inp_img)  
    # andrej plot:
    quick_plot_bboxes(predictions, inp_img)
```



## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
