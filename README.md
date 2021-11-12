Part of the IPALM project, this is a fusion of a MobilenetV3 trained on the smaller MINC2500 dataset and the default Detectron2 InstanceSegmentor trained on COCO, ShopVRB, YCB and a few custom images.


### How to install:
1. Go to some folder A and: `git clone https://github.com/Hartvi/Detectron2-mobilenet`
    - This will create the folder called `Detectron2-mobilenet
2. Rename `Detectron2-mobilenet` to `detectron2`
3. In folder A: `python -m pip install -e detectron2 --user`
4. ???
5. Profit 1 billion dollars

### Short demo (in folder A):

(Not sure if it works exactly this way)
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
