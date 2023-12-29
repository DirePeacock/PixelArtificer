import os
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps
import pathlib
import sys
import argparse
import colorsys
import math
import multiprocessing
import logging

from . import Defaults 
from .ParallelPixelProcessor import ParallelPixelProcessor
#todo
# bug some of the darker pixel are having thier opacity reduced?
# check the gradient stuff
# maybe change selection to use pencilyness of a pixel to set alpha
# pencilyness 
#   uses hue differenece from paper color
#   lightness & saturation
class SketchExtractor(ParallelPixelProcessor):
    def __init__(self, 
                 image, 
                 output=None, 
                 pxl_func=None, 
                 num_processes=Defaults.default_process_count, 
                 mode="RGBA", 
                 contrast_factor=Defaults.default_contrast_selection_factor, 
                 cut_off_percentile=Defaults.default_cutoff_percentile, 
                 darkening_factor=Defaults.default_darkening_factor, 
                 cutoff_gradient=Defaults.default_gradient_selection, 
                 blur_radius=Defaults.default_blur_radius,
                 blur_opacity=Defaults.default_blur_opacity,
                 use_edge_detection=True):
        self.image = image
        self.output = output
        self.contrast_factor = contrast_factor
        self.cut_off_percentile = cut_off_percentile
        self.darkening_factor = darkening_factor
        self.cutoff_gradient = cutoff_gradient
        self.blur_radius = blur_radius
        self.blur_opacity = blur_opacity
        self.use_edge_detection = use_edge_detection
        super().__init__(pxl_func, num_processes, mode)

    def edge_detection(self, input_img):
        """use edge detection to find edges
        return an image that is darker around areas of contrast"""
        _img = input_img.convert("L")
 
        # Calculating Edges using the passed laplacian Kernel
        # idk how to modify these values to change anything but whatever
        final = _img.filter(ImageFilter.Kernel((3, 3), 
                                               (-1, -1, -1,
                                                -1, 8, -1, 
                                                -1, -1, -1), 
                                                1, 0))
        
        final = ImageOps.invert(final)
        final = self.image_denoise(final, passes=2)

        return final
    
    def image_denoise(self, input_img, passes=1):
        "use filter to get an image that is less noisy from the paper"
        # this blur radius is 0.02 of image shortest side
        blur_radius = int(min(input_img.width, input_img.height) * 0.02)
        retval = ImageEnhance.Contrast(input_img).enhance(2)
        for i in range(passes):    
            
            retval = retval.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            retval = retval.filter(ImageFilter.MinFilter(3))
            retval = retval.filter(ImageFilter.MaxFilter)
            retval = retval.filter(ImageFilter.MedianFilter(3))
            #increase contrast
            retval = ImageEnhance.Contrast(retval).enhance(1.2)
            
        return retval
    

    def main_loop(self, input_img, num_processes):
        """same as the old one with some extra parts
        NEW:
            enhance image contrast, save as new intermediate image
            use intermediate image to get the brightness to alpha
            stores those values to make a percentile for the alpha
            set the alpha to 0 for all pixels below that percentile
        """
        hc_image = ImageEnhance.Contrast(input_img).enhance(1.0+self.contrast_factor)

        edge_detection_image = self.edge_detection(hc_image)
        edge_detection_image = edge_detection_image.convert("HSV")
        # edge_detection_image.show()
        # print("edge detection done")
        # exit(0)
        rects = self.split_into_rectangles(input_img, num_processes)
        collection_processes = []
        
        if self.blur_radius is None:
            self.blur_radius = Defaults.default_blur_radius_percent * (input_img.width + input_img.height) / 2


        # oh yeah baby use all that space for dicts and arrays
        # we got plenty of flops and bits to spare
        return_dict = multiprocessing.Manager().dict()
        value_dict = multiprocessing.Manager().dict()
        
        for rect in rects:
            return_dict[rect.key] = None
            value_dict[rect.key] = {}
                
        for rect in rects:
            collection_processes.append(
                multiprocessing.Process(
                    target=self.sub_loop_collection, args=(input_img, hc_image, rect, return_dict, value_dict, edge_detection_image)
                )
            )
        
        print(f"step 1:\t Collecting img info")

        for process in collection_processes:
            process.start()
        for process in collection_processes:
            process.join()  

        hc_alpha_img = Image.new(input_img.mode, input_img.size)
        self.combine_image(hc_alpha_img, return_dict)
        value_array = [0 for i in range(256)]
        for key, value in value_dict.items():
            for i, v in value.items():
                value_array[i] += v
        total_pixels = input_img.width * input_img.height
        
        #pixels_to_find_cutoff = total_pixels * (1-self.cut_off_percentile)
        # we want to find the percentile of pixels that have highest alpha
        pixels_to_find_cutoff = total_pixels * self.cut_off_percentile
        pixels_found = 0
        cutoff_value = 0
        
        for i, value in enumerate(value_array[::-1]):
            pixels_found += value
            if pixels_found > pixels_to_find_cutoff:
                cutoff_value = i
                break
        
        print(f"step 2:\t Processing image")

        selection_processes = []
        for rect in rects:
            selection_processes.append(
                multiprocessing.Process(
                    target=self.sub_loop_selection, args=(input_img, hc_alpha_img, rect, return_dict, cutoff_value)
                )
            )

        for process in selection_processes:
            process.start()
        for process in selection_processes:
            process.join()

        print(f"step 3:\t Combining image")

        # new image of the same size
        new_image = Image.new(input_img.mode, input_img.size)
        self.combine_image(new_image, return_dict)
        
        
        if self.blur_radius > 0 and self.blur_opacity > 0:
            print(f"step 4:\t blurring image")
            new_image = self.blur_image(new_image, self.blur_radius, self.blur_opacity)
        
        return new_image
    
    def blur_image(self, image, blur_radius, blur_opacity):
        """blur image"""
        image = image

        if blur_radius > 0 and blur_opacity > 0:
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # create new blank image of the same size
            blank_image = Image.new(image.mode, image.size, (0,0,0,0))

            # blend blurred image 
            blurred_image = Image.blend(image, blurred_image, blur_opacity)

            # paste blurred image into blank image
            image = Image.alpha_composite(image, blurred_image)

        return image
      
    def get_edge_detection_multiplier(self, ed_pxl):
        """this is to add a modifier to the darkness of the original image
        
        """    
        if not self.use_edge_detection:
            return 1.0
        
        edgyness = 1.0 - ed_pxl[2]/255
        min_edgyness = 0.0
        mid_edgyness = 0.23
        multiplier_magnitude = 0.7
        
        multiplier = 1.0
        if edgyness >= min_edgyness:
            multiplier = 1.0 + multiplier_magnitude * (edgyness - mid_edgyness)

        return multiplier

    def sub_loop_collection(self, input_img, intermediate_image, rect, target_dict, value_dict, ed_image):
        """make a new image in the rectangle
        return that value"""
        sub_image=Image.new(input_img.mode, (rect.w, rect.h))
        tmp_value_dict = {i:0 for i in range(256)}
        for x in range(rect.x, rect.x+rect.w):
            for y in range(rect.y, rect.y+rect.h):
                _x, _y = x-rect.x, y-rect.y
                pixel = self.brightness_to_opacity(intermediate_image.getpixel((x, y)))
                
                ed_image_pixel_hsv = ed_image.getpixel((x, y))
                ed_modifier = self.get_edge_detection_multiplier(ed_image_pixel_hsv)
                new_alpha = int(pixel[3] * ed_modifier)
                new_alpha = max(0, min(new_alpha, 255))
                pixel = tuple([pixel[0], pixel[1], pixel[2], new_alpha])

                tmp_value_dict[pixel[3]] += 1
                sub_image.putpixel((_x, _y), pixel)
        
        # write this at the end so as to avoid mutext time delays
        value_dict[rect.key] = {key:value for key, value in tmp_value_dict.items() if value > 0}
        target_dict[rect.key] = sub_image
    
    def sub_loop_selection(self, input_img, intermediate_image, rect, target_dict, cutoff_value):
        """make a new image in the rectangle
        return that value"""
        sub_image=Image.new(input_img.mode, (rect.w, rect.h))
        for x in range(rect.x, rect.x+rect.w):
            for y in range(rect.y, rect.y+rect.h):
                # full_img coord
                _x, _y = x-rect.x, y-rect.y
                left_rgba = input_img.getpixel((x, y))
                right_rgba = intermediate_image.getpixel((x, y))
                new_pixel = self.magic_pixel_combinator(left_rgba, right_rgba, cutoff_value, darkening_factor=self.darkening_factor, cutoff_gradient=self.cutoff_gradient)
                sub_image.putpixel((_x, _y), new_pixel)
        target_dict[rect.key] = sub_image

    
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
    
    @classmethod
    def magic_pixel_combinator(cls, left, right, cutoff, darkening_factor=0.4, cutoff_gradient=0.2):
        """
        darken rgba by darkening factor
        if right alpha is below cutoff, set to 0
        if right alpha is above_gradient but below cutoff
            get percentage distance from cutoff
            multiply alpha by that 1.0 - percentage
        """
        secondary_cutoff = cutoff * cutoff_gradient
        secondary_cutoff = max(secondary_cutoff, 0)
        if right[3] < cutoff:
            if right[3] > secondary_cutoff:
                # get percentage distance from cutoff
                percentage = (right[3] - secondary_cutoff) / (cutoff - secondary_cutoff)
                # multiply alpha by that 1.0 - percentage
                new_alpha = right[3] * (percentage)
                right = right[:3] + (new_alpha,)
            else:
                right = (0,0,0,0)
                left = (0,0,0,0)
        if right[3] > 0:
            left = cls.darken_rgba(left, darkening_factor)
        new_pixel = cls.apply_right_alpha_to_left_rgb(left, right)
        new_pixel = tuple([int(i) for i in new_pixel])
        return new_pixel
    
    def test_gradient(self):
        rgba_list = [ ]
        for i in range(0,10):
            lightness = i/10
            j=int(255*lightness)
            rgba_list.append((j,j,j,j))
        
        outputs = []
        for rgba in rgba_list:
            #left = right but with alpha 255
            left = rgba[:3] + (255,)
            output = self.magic_pixel_combinator(left, rgba, 255, darkening_factor=0.0, cutoff_gradient=0.5)
            outputs.append(output)
            # verify that the output gradient gets more opacity as it is darker and closer to the cutoff
        """127 should be 0
        152 should be 0.2*0.6
        178 should be 0.4*0.7
        """
        print(outputs)


    @classmethod
    def darken_rgba(cls, rgba, darkening_factor):
        """make rgb values darker by multiplying 1.0 - darkening_factor to make the values smaller
        """
        multiplier = 1.0 - darkening_factor
        new_rgba = (rgba[0]*multiplier, rgba[1]*multiplier, rgba[2]*multiplier, rgba[3])
        return new_rgba
    
    @staticmethod
    def apply_right_alpha_to_left_rgb(left, right):
        """ return rbga with right alpha applied to left rgb
        """
        return (left[0], left[1], left[2], right[3])
    

    def do_main(self):
        """open image"""
        old_img = Image.open(self.image)
        old_img = old_img.convert(self.mode)
        new_img = self.main_loop(old_img, self.num_processes)
        new_img.save(self.output)
        
def test_brightness():    
    rgba_list = [ ]
    for i in range(0,10):
        lightness = i/10
        j=int(255*lightness)
        rgba_list.append((j,j,j,255))
    for rgba in rgba_list:
        print(rgba, SketchExtractor.brightness_to_opacity(rgba))

