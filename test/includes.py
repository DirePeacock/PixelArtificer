import os
import sys
import pathlib
basedir = pathlib.Path(__file__).parent.parent
srcdir = basedir / 'src'
pixelartiferdir = srcdir / 'pixelartifer'

sys.path.append(str(srcdir))
sys.path.append(str(pixelartiferdir))