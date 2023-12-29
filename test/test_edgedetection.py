import includes
import logging
import pathlib
from PIL import Image
from pixelartificer.ExtractorClass.SketchExtractor import SketchExtractor

def _get_test_image(imgname="piano_brackets_drow.png"):
    img_path = pathlib.Path(__file__).parent / "_test_images" / imgname
    return Image.open(img_path)

def test_edge_detection():
    """open test image and run edge detection
    verify that the modifier function is working correctly"""
    
    img = _get_test_image().convert("HSV")
    subject_extractor = SketchExtractor(img)
    subject_extractor.edge_detection(input_img=img)
    
    zz_pixel = img.getpixel((0,0))
    test_list = [
        {'pixel':tuple([222, 222, 255]), 'output':1.0},
        {'pixel':tuple([222, 222, 150]), 'output':0.85588235},
        {'pixel':tuple([222, 222, 100]), 'output':0.9539215},
        {'pixel':tuple([222, 222, 50]), 'output':1.05196078},
        {'pixel':tuple([222, 222, 0]), 'output':1.15},
        ]
    for test in test_list:
        input = test['pixel']
        output = subject_extractor.get_edge_detection_multiplier(input)
        # print(f"{output} {test['output']}")
        diff = abs(output - test['output'])
        assert diff < 0.0001

    