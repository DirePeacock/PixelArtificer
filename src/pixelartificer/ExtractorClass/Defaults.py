#! python 
import pathlib

test_img = pathlib.Path(__file__).parent.parent / "test" / "_test_images" /"piano_brackets_drow.png"

default_process_count = 8 
default_cutoff_percentile = 0.66
default_gradient_selection = 0.5
default_darkening_factor = 0.4
default_contrast_selection_factor = 0.6
default_blur_radius=None
default_blur_opacity=0.2
# if not passed use this percent of the image width
default_blur_radius_percent=0.01

# for using edge detection
default_min_edgyness = 0.0
default_mid_edgyness = 0.23
default_multiplier_magnitude = 0.7