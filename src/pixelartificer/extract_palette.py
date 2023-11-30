#! python
import os
from PIL import Image
import pathlib
import sys
import argparse
import colorsys
import math
from . import color

test_img = pathlib.Path(
    "E:\\pics\\reference\\trythis\\sprites\\loopHeroPortraits_trans.png"
)


class PaletteExtractor:
    def __init__(self, image, output=None, using_hsv=True):
        # self.image = image
        self.image_path = pathlib.Path(image)
        self.output = output
        self.palette_image = None
        self.using_hsv = using_hsv
        self.colors = []
        self.hsv_palette = []

    def update_percent(self):
        self.i_px += 1
        new_percent_complete = int(self.i_px / self.num_px * 100)
        if new_percent_complete > self.percent_complete:
            self.percent_complete = new_percent_complete
            print(f"%{self.percent_complete}")
            # os.system("clear")

    def _load_palette(self):
        """open image file to rgba list"""

        with Image.open(self.image_path) as image_obj:
            self.num_px = image_obj.width * image_obj.height
            self.i_px = 0
            self.percent_complete = 0
            for x in range(0, image_obj.width):
                for y in range(0, image_obj.height):
                    pixel = image_obj.getpixel((x, y))

                    if pixel[3] == 0:
                        continue
                    new_pixel = (pixel[0], pixel[1], pixel[2], 255)
                    new_hsv = colorsys.rgb_to_hsv(
                        pixel[0] / 255, pixel[1] / 255, pixel[2] / 255
                    )
                    if new_pixel not in self.colors:
                        self.colors.append(new_pixel)
                    if new_hsv not in self.hsv_palette:
                        self.hsv_palette.append(new_hsv)
                    self.update_percent()

    def main(self):
        self._load_palette()
        self._create_palette_image()
        self._save_palette_image()

    def _create_palette_image(self):
        """create a new image with 16x16 squares of each color sorted by hue
        each row has 4 squares of 16x16
        """
        sq_size = 16
        colors_list = self.hsv_palette if self.using_hsv else self.colors
        mode = "RGBA"  # "HSV" if self.using_hsv else "RGBA"
        num_columns = 4
        num_rows = math.ceil(len(self.colors) / num_columns)

        self.palette_image = Image.new(
            mode, (sq_size * num_columns, sq_size * num_rows), (0, 0, 0, 0)
        )
        if self.using_hsv:
            colors_list = sorted(colors_list, key=lambda x: x[0])

        for i, color in enumerate(colors_list):
            x = (i % num_columns) * sq_size
            y = (i // num_columns) * sq_size
            this_color = color
            if self.using_hsv:
                this_color = colorsys.hsv_to_rgb(*color)
                this_color = (
                    int(this_color[0] * 255),
                    int(this_color[1] * 255),
                    int(this_color[2] * 255),
                    255,
                )

            new_square = Image.new("RGBA", (sq_size, sq_size), this_color)
            self.palette_image.paste(new_square, (x, y))
            # for x_ in range(x, x + 16):
            #     for y_ in range(y, y + 16):
            #         self.palette_image.putpixel((x_, y_), color)

    def _save_palette_image(self):
        """save palette image to file named with palette_ prefixed to it"""
        output_path = self.output
        if not output_path:
            output_path = os.path.join(
                os.path.dirname(self.image_path),
                f"palette_{os.path.basename(self.image_path)}",
            )
        self.palette_image.save(output_path)

    def extract_palette():
        """return list of color objects used in the image"""
        palette = []
        with Image.open(self.image_path) as image_obj:
            for x in range(0, image_obj.width):
                for y in range(0, image_obj.height):
                    pixel = image_obj.getpixel((x, y))

                    if pixel[3] == 0:
                        continue
                    new_color = color.Color(rgba=pixel)
                    if new_color not in palette:
                        palette.append(new_color)

                    # new_pixel = (pixel[0], pixel[1], pixel[2], 255)
                    # new_hsv = colorsys.rgb_to_hsv(
                    #     pixel[0] / 255, pixel[1] / 255, pixel[2] / 255
                    # )
                    # if new_pixel not in self.colors:
                    #     self.colors.append(new_pixel)
                    # if new_hsv not in self.hsv_palette:
                    #     self.hsv_palette.append(new_hsv)


def main(args):
    print(__file__, args.__dict__)
    runner = PaletteExtractor(
        image=args.image, output=args.output, using_hsv=(not args.rgb)
    )
    runner.main()


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
        "-o",
        "--output",
        type=str,
        default=None,
        help="output path for palette image",
    )

    return parser.parse_args(args_)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
