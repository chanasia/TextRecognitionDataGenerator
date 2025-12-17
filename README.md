# TextRecognitionDataGenerator [![CircleCI](https://circleci.com/gh/Belval/TextRecognitionDataGenerator/tree/master.svg?style=svg)](https://circleci.com/gh/Belval/TextRecognitionDataGenerator/tree/master) [![PyPI version](https://badge.fury.io/py/trdg.svg)](https://badge.fury.io/py/trdg) [![codecov](https://codecov.io/gh/Belval/TextRecognitionDataGenerator/branch/master/graph/badge.svg)](https://codecov.io/gh/Belval/TextRecognitionDataGenerator) [![Documentation Status](https://readthedocs.org/projects/textrecognitiondatagenerator/badge/?version=latest)](https://textrecognitiondatagenerator.readthedocs.io/en/latest/?badge=latest)

A synthetic Thai data generator for text recognition

## What is it for?

Generating Thai text image samples to train an OCR software. Now supporting thai text! and export COCO format (BBOX CHARACTER LEVEL) For a more thorough tutorial see [the official documentation](https://textrecognitiondatagenerator.readthedocs.io/en/latest/index.html).

## What do I need to make it work?
```bash
pip install pillow nltk natsort six fire uharfbuzz fonttools opencv-python tqdm diffimg arabic-reshaper python-bidi wikipedia
```

### Basic (CLI)
```bash
python run.py -dt dicts/th_en_words.txt -oc -ft "fonts/th/THSarabun.ttf" -b 1 -k 0 -rk -d 0 -do 0 -bl 0 -stw 0 -f 64 -e png
```
```bash
python run.py -dt dicts/th_en_words.txt -oc -ft "fonts/th/THSarabun.ttf" -b 1 -k 0 -rk -d 0 -do 0 -bl 0 -stw 0 -d 3 -do 10 -f 64 -e png
```

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

![Visual Image](samples/visual2.png "Visual Image")

### Dictionary

The text is loaded from a dictionary file (that can be found in the *dicts* folder). By default, all lines from the dictionary are used. If the `-c` parameter is specified, the text is chosen at random from the dictionary. The text is drawn on a white background made with Gaussian noise (configurable). The resulting image is saved as [text]_[index].jpg
