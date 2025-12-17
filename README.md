# TextRecognitionDataGenerator [![CircleCI](https://circleci.com/gh/Belval/TextRecognitionDataGenerator/tree/master.svg?style=svg)](https://circleci.com/gh/Belval/TextRecognitionDataGenerator/tree/master) [![PyPI version](https://badge.fury.io/py/trdg.svg)](https://badge.fury.io/py/trdg) [![codecov](https://codecov.io/gh/Belval/TextRecognitionDataGenerator/branch/master/graph/badge.svg)](https://codecov.io/gh/Belval/TextRecognitionDataGenerator) [![Documentation Status](https://readthedocs.org/projects/textrecognitiondatagenerator/badge/?version=latest)](https://textrecognitiondatagenerator.readthedocs.io/en/latest/?badge=latest)

A synthetic Thai data generator for text recognition

## What is it for?

Generating Thai text image samples to train an OCR software. Now supporting thai text! and export COCO format (BBOX CHARACTER LEVEL) For a more thorough tutorial see [the official documentation](https://textrecognitiondatagenerator.readthedocs.io/en/latest/index.html).

## What do I need to make it work?
```bash
pip install lmdb pillow torchvision nltk natsort six fire
```

### Basic (CLI)
```bash
python run.py -dt dicts/th_en_words.txt -oc -ft "fonts/th/THSarabunNew.ttf" -b 1 -k 0 -rk -d 0 -do 0 -bl 0 -stw 0
```

### Command Line Arguments

Here is an explanation of the flags used in the example command:

| Argument | Description | Recommended for Clean Data |
| :--- | :--- | :--- |
| `-dt` | Path to the dictionary file (text source). | `dicts/th_en_words.txt` |
| `-oc` | **Output COCO:** Enable COCO JSON export mode. | (Required) |
| `-ft` | Path to the TTF font file. | `fonts/th/THSarabunNew.ttf` |
| `-b` | Background type (0=Gaussian Noise, 1=Plain White, 2=Quasicrystal). | `1` (White) |
| `-k`, `-rk` | Skew angle and Random Skew. Set to 0 to keep text straight. | `0` |
| `-d`, `-do` | Distortion and Distortion Orientation. | `0` |
| `-bl` | Blur radius. | `0` (No blur) |
| `-stw` | Stroke width (thickness of outline). | `0` (No outline) |

### COCO Output Format

When using the `-oc` flag, the generator creates an `dataset/` folder with the following structure:

```text
dataset/
├── annotations/
│   ├── train.json
│   └── val.json
└── [images generated]
```

### Example

![Sample Image](samples/กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย_8.jpg "Sample Output")

![Sample Image](samples/การนอนหลับพักผ่อนให้เพียงพอเป็นสิ่งสำคัญ_10.jpg "Sample Output")

![Sample Image](samples/ช่วย%20Check%20บิลโต๊ะห้าให้หน่อยครับลูกค้าจะกลับแล้ว_18.jpg "Sample Output")

### Bounding Box Visualization

![Visual Image](samples/visual1.png "Visual Image")

![Visual Image](samples/visual2.png "Visual Image")

### Dictionary

The text is loaded from a dictionary file (that can be found in the *dicts* folder). By default, all lines from the dictionary are used. If the `-c` parameter is specified, the text is chosen at random from the dictionary. The text is drawn on a white background made with Gaussian noise (configurable). The resulting image is saved as [text]_[index].jpg
