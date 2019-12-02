# Author: Daniel Lindberg

# Native python modules
import os
import matplotlib.pyplot as plt
import numpy as np

# Native python submodules
from PIL import Image
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters.rank import entropy

all_files = os.listdir(".")
finalized_list = all_files
# get all of the jpg
for a_f in all_files:
	if not a_f.endswith(".jpg"):
		finalized_list.remove(a_f)
for each_item in finalized_list:
	img = Image.open(each_item)
	entropy_img = entropy(img)
	print entropy
