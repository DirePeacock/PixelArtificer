#! python
import os
from PIL import Image
from PIL import ImageEnhance
import pathlib
import sys
import argparse
import colorsys
import math
import multiprocessing
import logging

# test_img = pathlib.Path(
#     "D:\\pics\\reference\\trythis\\sprites\\loopHeroPortraits_trans.png"
# )
test_img = pathlib.Path(
    "D:\\files\\cod\\cool_cli_stuff\\quickies\\pixelart\\_test_images\\piano_brackets_drow.png"
)


class Rectangle():
    def __init__(self, x,y,w,h):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
    
    @property
    def key(self):
        return (f"{self.x},{self.y},{self.w},{self.h}")
    def __repr__(self):
        return self.key
    def __str__(self):
        return self.__repr__()
    @classmethod
    def from_key(cls, key):
        x,y,w,h = key.split(",")
        return cls(int(x), int(y), int(w), int(h))

class ParallelPixelProcessor():
    def __init__(self, pxl_func=None, num_processes=16, mode="RGBA"):
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


    def combine_image(self, new_image,  return_dict):
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


#todo
# bug some of the darker pixel are having thier opacity reduced?
# check the gradient stuff
# maybe change selection to use pencilyness of a pixel to set alpha
# pencilyness 
#   is hue differenece from paper color
#   lightness
class SketchExtractor(ParallelPixelProcessor):
    def __init__(self, image, output=None, pxl_func=None, num_processes=16, mode="RGBA", contrast_factor=0.6, cut_off_percentile=0.7, darkening_factor=0.4, cutoff_gradient=0.3):
        self.image = image
        self.output = output
        self.contrast_factor = 0.6
        self.cut_off_percentile = 0.6
        self.darkening_factor = 0.4
        self.cutoff_gradient = 0.3
        self.paper_lightness_cutoff_value = 0.65
        super().__init__(pxl_func, num_processes, mode)

    # TODO    
    def determine_paper_color(self, paper_dict):
        """ 
        track things about paper colors by pixels above the lightness cutoff
        median of things above the lightness cutoff maybe?
        """
        paper_color = [0,0,0,0]
        
        self.paper_color = (int(i) for i in paper_color)
    
    # TODO
    def determine_pencilyness(self, rgba):
        # based on how different the things are from the paper color
        hsv = colorsys.rgb_to_hsv(*rgba[:3])
        determined_pencilyness = 0
        pass
    
    def main_loop(self, input_img, num_processes):
        """same as the old one with some extra parts
        NEW:
            enhance image contrast, save as new intermediate image
            use intermediate image to get the brightness to alpha
            stores those values to make a percentile for the alpha
            set the alpha to 0 for all pixels below that percentile
        """
        
        hc_image = ImageEnhance.Contrast(input_img).enhance(1.0+self.contrast_factor)
        rects = self.split_into_rectangles(input_img, num_processes)
        collection_processes = []
        
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
                    target=self.sub_loop_collection, args=(input_img, hc_image, rect, return_dict, value_dict)
                )
            )
        
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
        print(f"cut off value: {cutoff_value}")
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

        # new image of the same size
        new_image = Image.new(input_img.mode, input_img.size)
        self.combine_image(new_image, return_dict)
        
        return new_image
    
    

    def sub_loop_collection(self, input_img, intermediate_image, rect, target_dict, value_dict):
        """make a new image in the rectangle
        return that value"""
        sub_image=Image.new(input_img.mode, (rect.w, rect.h))
        tmp_value_dict = {i:0 for i in range(256)}
        for x in range(rect.x, rect.x+rect.w):
            for y in range(rect.y, rect.y+rect.h):
                _x, _y = x-rect.x, y-rect.y
                pixel = self.brightness_to_opacity(intermediate_image.getpixel((x, y)))
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

    
    def combine_image(self, new_image,  return_dict):        
        for key, sub_image in return_dict.items():
            rect = Rectangle.from_key(key)
            x,y = rect.x, rect.y
            # paste sub_image into new_image
            new_image.paste(sub_image, (x,y))
        # new_image.show()
        return new_image

    
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
        secondary_cutoff = cutoff * (1.0-cutoff_gradient)
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
        expected_output = [ ]
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
        

class AlphaExtractor():
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
        hsv = colorsys.rgb_to_hsv(*rgba[:3])
        rgba2 = rgba[:3] + (hsv[2],)
        return rgba2
    
    def do_main(self):
        """open image"""
        old_img = Image.open(self.image)
        new_img = ParallelPixelProcessor.transform_image(old_img, self.brightness_to_opacity)
        new_img.save(self.output)

    @classmethod
    def main(cls, image, output=None, mode="RGBA"):
        cls(image, output, mode).do_main()



def main(args):
    print(__file__, args.__dict__)
    _output = args.output
    suffix = "_alpha.png" if not args.sketch_extractor else "_extracted.png"
    if _output is None:
        _output = os.path.splitext(args.image)[0] + "_alpha.png"
    if args.sketch_extractor:
        runner = SketchExtractor(
            image=args.image, output=_output)
        runner.do_main()
    else:
        runner = AlphaExtractor(
            image=args.image, output=_output)
        runner.do_main()

def test_brightness():
    
    rgba_list = [ ]
    for i in range(0,10):
        lightness = i/10
        j=int(255*lightness)
        rgba_list.append((j,j,j,255))
    for rgba in rgba_list:
        print(rgba, SketchExtractor.brightness_to_opacity(rgba))

def parse_args(args_):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default=test_img,
        help="image path to apply palette to",
    )
    parser.add_argument(
        "-r", "--rgb", default=False, action="store_true", help="use rgb instead of hsv"
    )
    parser.add_argument(
        "-s", "--sketch-extractor", default=True, action="store_true", help="use settings for sketch extractor"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="output path for palette image",
    )

    return parser.parse_args(args_)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
