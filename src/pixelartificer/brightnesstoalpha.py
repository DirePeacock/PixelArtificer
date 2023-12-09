#! python
import os
from PIL import Image
import sys
import argparse

from ExtractorClass import SketchExtractor
from ExtractorClass import AlphaExtractor
from ExtractorClass import Defaults

def main(args):
    print(__file__, args.__dict__)
    _output = args.output
    
    suffix = "_alpha.png" if args.just_opacity else "_extracted.png"
    if _output is None:
        _output = os.path.splitext(args.image)[0] + suffix
    
    
    if args.interactive:
        ExtractorDemo.do_main(
            image=args.image, 
            output=_output, 
            num_processes=args.process_count,
            contrast_factor=args.contrast_selection_factor,
            cut_off_percentile=args.percentile,
            darkening_factor=args.darkening_factor,
            cutoff_gradient=args.gradient_selection,
            blur_radius=args.blur_radius,
            blur_opacity=args.blur_opacity)
        
    elif not args.just_opacity:
        runner = SketchExtractor.SketchExtractor(
            image=args.image, 
            output=_output, 
            contrast_factor=args.contrast_selection_factor,
            cut_off_percentile=args.percentile,
            darkening_factor=args.darkening_factor,
            cutoff_gradient=args.gradient_selection,
            blur_radius=args.blur_radius,
            blur_opacity=args.blur_opacity
            )
        runner.do_main()
    else:
        runner = AlphaExtractor.AlphaExtractor(
            image=args.image, output=_output)
        runner.do_main()


class ExtractorDemo():
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
                 blur_opacity=Defaults.default_blur_opacity):
        self.image = image
        self.output = output
        self.mode=mode
        self.contrast_factor = contrast_factor
        self.cut_off_percentile = cut_off_percentile
        self.darkening_factor = darkening_factor
        self.cutoff_gradient = cutoff_gradient
        self.paper_lightness_cutoff_value = 0.65
        self.blur_radius = blur_radius
        self.blur_opacity = blur_opacity
    
    def demo(self):
        """open image"""
        old_img = Image.open(self.image)
        old_img = old_img.convert(self.mode)
        
        largest_dimension = max(old_img.width, old_img.height)
        scale_factor = 512 / largest_dimension
        old_img = old_img.resize((int(old_img.width*scale_factor), int(old_img.height*scale_factor)))

        args_dict = {
            "contrast_factor":self.contrast_factor,
            "cut_off_percentile":self.cut_off_percentile,
            "darkening_factor":self.darkening_factor,
            "cutoff_gradient":self.cutoff_gradient,
            "blur_radius":self.blur_radius,
            "blur_opacity":self.blur_opacity
        }

        good_settings = False
        while not good_settings:
            runner = SketchExtractor.SketchExtractor(
                image=self.image, 
                output=self.output, 
                contrast_factor=args_dict["contrast_factor"],
                cut_off_percentile=args_dict["cut_off_percentile"],
                darkening_factor=args_dict["darkening_factor"],
                cutoff_gradient=args_dict["cutoff_gradient"],
                blur_radius=args_dict["blur_radius"],
                blur_opacity=args_dict["blur_opacity"]
                )
            print("\ntesting the settings\n")
            new_img = runner.main_loop(old_img, runner.num_processes)
            # new_img.save(self.output)
            demo_image = self.make_img_demo(new_img)
            demo_image.show()
            print("current settings:")
            for key, value in args_dict.items():
                print(f"\t{key}: {value}")
            good_settings = self.ask_yes_no("do you like these settings?\n")
            if not good_settings:
                args_dict = self.get_new_settings(args_dict)

        # run sketch extractor
        runner = SketchExtractor.SketchExtractor(
            image=self.image,
            output=self.output,
            contrast_factor=args_dict["contrast_factor"],
            cut_off_percentile=args_dict["cut_off_percentile"],
            darkening_factor=args_dict["darkening_factor"],
            cutoff_gradient=args_dict["cutoff_gradient"],
            blur_radius=args_dict["blur_radius"],
            blur_opacity=args_dict["blur_opacity"]
        )
        print("\nlaunching sketch extractor\n".upper())
        new_img = runner.main_loop(Image.open(self.image), runner.num_processes)
        new_img.save(self.output)
        print(f"saved to {self.output}")

    
    def make_img_demo(self, new_img):
        """make a demo image"""
        white_image = Image.new(new_img.mode, new_img.size, (255, 255, 255, 255))
        transparent_image = Image.new(new_img.mode, new_img.size, (0, 0, 0, 0))
        demo_image = Image.new(new_img.mode, (new_img.width*2, new_img.height), (0,0,0,0))
        alpha_image = Image.new(new_img.mode, (new_img.width*2, new_img.height), (0,0,0,0))


        demo_image.paste(white_image, (0, 0))
        demo_image.paste(transparent_image, (new_img.width, 0))

        alpha_image.paste(new_img, (0, 0))
        alpha_image.paste(new_img, (new_img.width, 0))

        composted_img = Image.alpha_composite(demo_image, alpha_image)

        return composted_img

    def get_new_settings(self, args_dict):
        
        making_changes = True
        keys = list(args_dict.keys())
        
        while making_changes:
            last_arg_dict_lines = []
            for key, value in args_dict.items():
                last_arg_dict_lines.append(f"{key}: {value}")
            last_arg_dict_lines.append("done")

            print("changning settings settings:")
            number = self.ask_number(last_arg_dict_lines)
            if number == len(last_arg_dict_lines)-1:
                making_changes = False
                continue
            else:
                key = keys[number]
                old_value = args_dict[key]
                new_value = self.get_new_number(old_value, f"\nnew value for {key}")
                args_dict[key] = new_value
        return args_dict
            

    def ask_yes_no(self, question):
        print(question)
        answer = input("y/n: ")
        if answer.lower() == "y":
            return True
        return False
    
    def ask_number(self, lines):
        for i, line in enumerate(lines):
            print(f"[{i}] {line}")
        answer = input("number: ")
        
        return int(answer)
        
    def get_new_number(self, old_number, question):
        print(question)
        answer = input(f"current value: {old_number}\nnew value: ")
        if answer.replace(".", "").isnumeric():
            return float(answer)
        else:
            return int(answer)
    @classmethod
    def do_main(cls, * args, ** kwargs):
        cls(*args, **kwargs).demo()
        

def parse_args(args_):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image",
        type=str,
        default=Defaults.test_img,
        help="image path to apply palette to",
    )
    parser.add_argument(
        "--just-opacity", dest='just_opacity', default=False, action="store_true", help="use settings for sketch extractor"
    )
    parser.add_argument(
        "-i", "--interactive", dest='interactive', default=False, action="store_true", help="interactive mode for the sketch extractor"
    )
    parser.add_argument(
        "--process-count",
        type=float,
        default=Defaults.default_process_count,
        help=f"number of processes to use\ndefault: {Defaults.default_process_count}",
    )
    parser.add_argument(
        "-p",
        "--percentile",
        type=float,
        default=Defaults.default_cutoff_percentile,
        help=f"the percitile of the darkes pixels to keep\ndefault: {Defaults.default_cutoff_percentile}",
    )
    parser.add_argument(
        "-g",
        "--gradient-selection",
        type=float,
        default=Defaults.default_gradient_selection,
        help=f"\nkeeps some extra pixels below the cutoff at a lower opacity\ndefault: {Defaults.default_gradient_selection}",
    )
    parser.add_argument(
        "-d",
        "--darkening-factor",
        type=float,
        default=Defaults.default_darkening_factor,
        help=f"darkens the RGB of pixels that are kept\ndefault: {Defaults.default_darkening_factor}",
    )
    parser.add_argument(
        "-c",
        "--contrast-selection-factor",
        type=float,
        default=Defaults.default_contrast_selection_factor,
        help=f"keeps some extra pixels below the cutoff at a lower opacity\ndefault: {Defaults.default_contrast_selection_factor}",
    )
    parser.add_argument(
        "--blur-radius",
        type=int,
        default=Defaults.default_blur_radius,
        help=f"radius of blur effect\ndefault: {Defaults.default_blur_radius}",
    )
    parser.add_argument(
        "--blur-opacity",
        type=float,
        default=Defaults.default_blur_opacity,
        help=f"opacity of blurred layer added\ndefault: {Defaults.default_blur_opacity}",
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
