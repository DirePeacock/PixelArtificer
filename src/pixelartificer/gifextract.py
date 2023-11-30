#! python
import logging
import sys
import os
import argparse
from PIL import Image
import pathlib
import enum

outline_types = enum.Enum("outline_types", "full dots none")
""" outline type is for the texture output to have divisions in it, to help user devide it
"""

"""
Feb 2023
now you can call this script on an image, force frame disposal mode and also store as a texture png
"""

"""
copied & updated from https://gist.github.com/BigglesZX/4016539

I searched high and low for solutions to the "extract animated GIF frames in Python"
problem, and after much trial and error came up with the following solution based
on several partial examples around the web (mostly Stack Overflow).

There are two pitfalls that aren't often mentioned when dealing with animated GIFs -
firstly that some files feature per-frame local palettes while some have one global
palette for all frames, and secondly that some GIFs replace the entire image with
each new frame ('full' mode in the code below), and some only update a specific
region ('partial').

This code deals with both those cases by examining the palette and redraw
instructions of each frame. In the latter case this requires a preliminary (usually
partial) iteration of the frames before processing, since the redraw mode needs to
be consistently applied across all frames. I found a couple of examples of
partial-mode GIFs containing the occasional full-frame redraw, which would result
in bad renders of those frames if the mode assessment was only done on a
single-frame basis.

Nov 2012
"""


def analyseImage(path):
    """
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    """
    im = Image.open(path)
    results = {
        "size": im.size,
        "mode": "full",
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results["mode"] = "partial"
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results


def processImageToTexture(
    path,
    partial=None,
    texture=True,
    outline_type=outline_types.none,
    px_pad=1,
    odd_color=(2, 210, 69, 255),
):
    """the same as the last one but it makes a texture instead of a sequence of images"""

    mode = analyseImage(path)["mode"]

    im = Image.open(path)

    i = 0
    p = im.getpalette()

    last_frame = im.convert("RGBA")
    if texture:
        new_size = (
            px_pad + ((px_pad + im.width) * im.n_frames),
            im.height + px_pad * 2,
        )
        new_image = Image.new("RGBA", new_size)
    good_job = False
    new_files = []
    try:
        while True:
            if not im.getpalette() and im.mode in ("L", "LA", "P", "PA"):
                im.putpalette(p)
            bg_color = (0, 0, 0, 0)  # (0, 0, 127, 255)
            new_frame = Image.new("RGBA", im.size, bg_color)

            if mode == "partial" and partial is None or partial:
                new_frame.paste(last_frame)

            new_frame.paste(im, (0, 0), im.convert("RGBA"))

            if texture:
                x_offset = px_pad + i * (im.width + px_pad)
                new_image.paste(new_frame, (x_offset, px_pad))

            else:
                new_path = (
                    pathlib.Path(path).parent / f"{pathlib.Path(path).stem}-{i}.png"
                )
                new_frame.save(new_path, "PNG")
                new_files.append(new_path)

            i += 1
            good_job = i >= im.n_frames

            last_frame = new_frame
            im.seek(im.tell() + 1)

    except EOFError:
        if good_job:
            logging.debug("good job, we got all the frames")

    if good_job and texture:
        # outlining the texture frames
        horizontal_bar = Image.new("RGBA", (new_size[0], 1), odd_color)
        vertical_bar = Image.new("RGBA", (1, new_size[1]), odd_color)
        # outline_type = outline_types.full
        if outline_type == outline_types.full:
            new_image.paste(horizontal_bar, (0, 0))
            new_image.paste(horizontal_bar, (0, new_size[1] - 1))

        for i in range(im.n_frames + 1):
            x_offset = px_pad + i * (im.width + px_pad)
            if outline_type == outline_types.full:
                new_image.paste(vertical_bar, (x_offset - px_pad, px_pad))
            elif outline_type == outline_types.dots:
                for y_val in [0, new_size[1] - 1]:
                    new_image.putpixel((x_offset - px_pad, y_val), odd_color)
            # new_image.paste(vertical_bar, (x_offset, new_size[1] - 1))

        # saving the texture
        new_path = pathlib.Path(path).parent / f"{pathlib.Path(path).stem}_texture.png"

        logging.debug(new_path)
        new_image.save(new_path, "PNG")
        new_files.append(new_path)


test_img = "E:\\files\\cod\\vandal5\\src\\vandal5\\artassets\\crit_slash.gif"


def main(args):
    processImageToTexture(
        args.image,
        partial=args.partial,
        texture=(not args.seq),
        outline_type=outline_types[args.outline],
    )
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)


def parse_args(args_):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default=test_img,
        help="image path to apply palette to",
    )
    parser.add_argument("-s", "--seq", action="store_true", help="sequence of images")
    parser.add_argument(
        "-p", "--partial", action="store_true", help="use partial replace mode"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="debug output")
    parser.add_argument(
        "-o",
        "--outline",
        type=str,
        default="dots",
        help="outline type",
        choices=["full", "dots", "none"],
    )
    return parser.parse_args(args_)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
