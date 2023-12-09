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

from . import Defaults
from .Rectangle import Rectangle

class ParallelPixelProcessor():
    def __init__(self, pxl_func=None, num_processes=Defaults.default_process_count, mode="RGBA"):
        self.pxl_func = pxl_func if pxl_func is not None else self.do_nothing
        self.num_processes = num_processes
        self.mode = mode

    def main_loop(self, input_img, num_processes):
        """split into sub images to get
        divide into rectangles for each
        spawn processes
        divide into rectangles
        send rectangles to processes
        wait for processes
        combine images
        """
        rects = self.split_into_rectangles(input_img, num_processes)
        processes = []
        return_dict = multiprocessing.Manager().dict()
        for rect in rects:
            return_dict[rect.key] = None
        for rect in rects:
            processes.append(
                multiprocessing.Process(
                    target=self.sub_loop, args=(input_img, rect, return_dict)
                )
            )
        
        for process in processes:
            process.start()
        for process in processes:
            process.join()  
        
        # new image of the same size
        new_image = Image.new(input_img.mode, input_img.size)
        combine_image = self.combine_image(new_image, return_dict)
        
        return new_image

    def split_into_rectangles(self, input_img, num_processes):
        """split into rectangles"""
        int_sqrt = math.floor(math.sqrt(num_processes))
        sub_image_width = input_img.width // int_sqrt
        sub_image_height = input_img.height // int_sqrt
        rects = []
        for i in range(int_sqrt):
            for j in range(int_sqrt):
                rects.append(
                    Rectangle(
                        x=i * sub_image_width,
                        y=j * sub_image_height,
                        w=sub_image_width,
                        h=sub_image_height,
                    )
                )
                if i == int_sqrt - 1:
                    rects[-1].w = input_img.width - rects[-1].x
                if j == int_sqrt - 1:
                    rects[-1].h = input_img.height - rects[-1].y
        
        return rects

    def sub_loop(self, input_img, rect, target_dict):
        """make a new image in the rectangle
        return that value"""
        sub_image=Image.new(input_img.mode, (rect.w, rect.h))
        for x in range(rect.x, rect.x+rect.w):
            for y in range(rect.y, rect.y+rect.h):
                _x, _y = x-rect.x, y-rect.y
                sub_image.putpixel((_x, _y), self.pxl_func(input_img.getpixel((x, y))))
        target_dict[rect.key] = sub_image


    def combine_image(self, new_image, return_dict):
        for key, sub_image in return_dict.items():
            rect = Rectangle.from_key(key)
            x,y = rect.x, rect.y
            new_image.paste(sub_image, (x,y))
        return new_image
    
    def do_nothing(self, pxl):
        return pxl     
        
    @classmethod
    def transform_image(cls, old_img, pxl_func):
        """instatiate the class and run the main loop"""
        return cls(pxl_func).main_loop(old_img, num_processes=16)


