Part of the IPALM project, this is a fusion of a MobilenetV3 trained on the smaller MINC2500 dataset and the default Detectron2 InstanceSegmentor trained on COCO, ShopVRB, YCB and a few custom images.

### Project structure
The base of the project is the detectron2 framework's instance segmentation backbone by [facebookresearch](https://github.com/facebookresearch/detectron2). 
Initially we tried to add a material segmentation/classification head to the instance segmentation network, however that proved to be exceedingly confusing because of the structure of the project. See the [progress log](https://gitlab.fel.cvut.cz/body-schema/ipalm/ipalm-vir2020-object-category-from-image/-/blob/master/code/PROGRESS.md) for details on how and why it was unfeasible given our constraints.

We ended up using just the default instance segmentor from VIR, retrained because something in either a newer detectron2 or pytorch version changed something in the background. The bounding boxes gathered from detectron2 are then plugged in a mobilenetv3 classifier trained on the MINC 2500 dataset.

### How to install:
1. Go to some folder A and: `git clone https://github.com/Hartvi/Detectron2-mobilenet`
    - This will create the folder called `Detectron2-mobilenet
2. Rename `Detectron2-mobilenet` to `detectron2`
3. In folder A: `python -m pip install -e detectron2 --user`
4. ???
5. Profit 1 billion dollars

### Short demo (in folder A):

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
