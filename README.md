# openimages2coco
Python script to convert open images instance segmentation masks to coco annotation format

## Prerequisites
- Download the desired images and the associated png masks from the [open images dataset](https://storage.googleapis.com/openimages/web/download.html) and extract them in seperate folders
- Also download the [class names](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv) and [train mask data](https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv) (and/or [validation](https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv) and [test](https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv) mask data) to the directory of the script
- Install `pycocotools`, `opencv-python` and `imagesize`
```Shell
pip install pycocotools opencv-python imagesize
```
On Windows, the original pycocotools is not working. You have to install [this fork](https://github.com/philferriere/cocoapi) instead:
```Shell
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

## Usage
```Shell
python convert_annotations.py -i IMAGE_FOLDER -m MASK_FOLDER
```
To see additional options, type:
```Shell
python convert_annotations.py -h
```

## Output
The script generates a file `coco_annotations.json` that contains the coco-style annotations. Note that compressed RLEs are used to store the binary masks. Since the json format cannot store the compressed byte array, they are base64 encoded. This is not COCO standard. Therefore, if you want to import the annotations using the [COCO API](https://github.com/cocodataset/cocoapi), you have to decode the base64 RLEs. Python example:

```python
from pycocotools.coco import COCO
import base64

def decode_base64_rles(coco):
    for ann in coco.dataset['annotations']:
        segm = ann['segmentation']
        if type(segm) != list and type(segm['counts']) != list:
            segm['counts'] = base64.b64decode(segm['counts'])

coco = COCO('coco_annotations.json')
decode_base64_rles(coco)
```
