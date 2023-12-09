#! python
import os
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import pathlib
import sys
import argparse
import colorsys
import math
import multiprocessing
import logging
from .ParallelPixelProcessor import ParallelPixelProcessor

class AlphaExtractor():
    """this one just does the brightness to transparency thing"""
    def __init__(self, image, output=None, mode="RGBA"):
        self.image = image
        self.output = output
        self.mode = mode
    
    @staticmethod
    def brightness_to_opacity(rgba):
        """ gets brightness from rgba tuple
        determine V value from HSV for that value
        return RGBA with that value as alpha
        """
        colorsys_rgba = tuple(rgba[i]/255 for i in range(3))
        hls = colorsys.rgb_to_hls(*colorsys_rgba[:3])
        darkness = 1.0 - hls[1]
        rgba2 = rgba[:3] + (darkness*255,)
        return tuple(int(i) for i in rgba2)
    
    
    def do_main(self):
        """open image"""
        old_img = Image.open(self.image)
        new_img = ParallelPixelProcessor.transform_image(old_img, self.brightness_to_opacity)
        new_img.save(self.output)

    @classmethod
    def main(cls, image, output=None, mode="RGBA"):
        cls(image, output, mode).do_main()