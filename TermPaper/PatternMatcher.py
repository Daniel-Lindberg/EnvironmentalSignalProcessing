# Author: Daniel Lindberg
# ECEN: Stochastic/Environmental Signal Processing
# Native python modules
import os
import pathlib
import PIL
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#Native python submodules
from PIL import Image

#Support Vector Machine Code
#----------------------------------------
# Get data points
test_dir = "tests/"
# get all of our classification names
test_sub_dirs = os.listdir(test_dir)

# Arrays for data points and classification
all_test_points = []

print "Start loading all data points in testing"
for img_index in range(len(test_sub_dirs)):
	print "Starting test:", test_sub_dirs[img_index]
	im = Image.open(test_dir+test_sub_dirs[img_index])
	# Create the image to be displayed
	pattern_matcher = im.copy()	
	#pattern_matcher = np.zeros((im.size[0], im.size[1], 3))
	copy_img = np.zeros((im.size[0], im.size[1], 3))
	for x in range(im.size[0]):
		for y in range(im.size[1]):
			r,g,b = im.getpixel((x,y))
			#print r,g,b
			tmp_pixel = [0, 0, 0]
			r_g_close = abs(r-g) < 23
			r_b_close = abs(r-b) < 23
			g_b_close = abs(b-g) < 23
			high_b = (b-g > 20) and (b-r > 20)
			high_g = (g-b > 20) and (g-r > 20)
			# Brown is typically ratio r,g,b 3:2:1
			is_brown = (b* 2.3 <= r) and (b * 1.4 <= g) and (g*1.4 <= r)
			if high_b:
				# Body water
				tmp_pixel = [0, 0, 255]
			elif high_g:
				# agriculture
				tmp_pixel = [0, 255, 0]
			elif is_brown:
				# agriculture
				tmp_pixel = [0, 255, 0]
			elif r_g_close and r_b_close and g_b_close:
				# Civilization
				tmp_pixel = [130, 130, 130]
			pattern_matcher.putpixel((x, y), (tmp_pixel[0], tmp_pixel[1], tmp_pixel[2]))
			"""
			for t in range(3):
				pattern_matcher[x,y,0] = tmp_pixel[0]
				pattern_matcher[x,y,1] = tmp_pixel[1]
				pattern_matcher[x,y,2] = tmp_pixel[2]
				copy_img[x,y,0] = r
				copy_img[x,y,1] = g
				copy_img[x,y,2] = b
			"""
	#print pattern_matcher.shape
	#print pattern_matcher.transpose((1,0,2)).shape
	#pattern_matcher = np.ascontiguousarray(pattern_matcher.transpose(1,0,2))
	#pattern_matcher = Image.fromarray(pattern_matcher, 'RGB')
	#copy_img = np.ascontiguousarray(copy_img.transpose(1,0,2))
	#copy_img = Image.fromarray(copy_img, 'RGB')
	#pattern_matcher = pattern_matcher.transpose(Image.TRANSPOSE)
	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(im)
	plt.title("Original Image")
	plt.xlabel('Pixels X-Direction')
	plt.ylabel('Pixels Y-Direction')
	plt.subplot(1,2,2)
	plt.title("Test Neural Network")
	plt.xlabel('Pixels X-Direction')
	plt.ylabel('Pixels Y-Direction')
	plt.imshow(pattern_matcher)
	blue_patch = mpatches.Patch(color='blue', label='Water-Body')
	gray_patch = mpatches.Patch(color='gray', label='Civilization')
	green_patch = mpatches.Patch(color='green', label='Vegetation')
	plt.legend(handles=[blue_patch, gray_patch, green_patch])
	#plt.subplot(1,3,3)
	#plt.imshow(copy_img)
	#plt.show()
	#sys.exit()
	print test_sub_dirs
	print img_index
	print "Pred-"+test_sub_dirs[img_index].split(".")[0]+"-patternmatcher.png"
	plt.savefig("Pred-"+test_sub_dirs[img_index].split(".")[0]+"-patternmatcher.png")


