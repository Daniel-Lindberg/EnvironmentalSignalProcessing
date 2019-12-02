#Author: Daniel Lindberg
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
from sklearn.svm import LinearSVC, SVC

# Smallest image width and height in training
IMG_WIDTH=299
IMG_HEIGHT=168

#Support Vector Machine Code
#----------------------------------------
# Get data points
img_dir = "imgs/"
test_dir = "tests/"
# get all of our classification names
final_sub_dirs = os.listdir(img_dir)
final_sub_dirs.sort()
test_sub_dirs = os.listdir(test_dir)

# Create array for all data points
all_data_points = []
all_classification_points = []
all_test_points = []

test_indices = [0]
class_names = np.array(final_sub_dirs)
print "Start loading image files into array for training"
for x in range(len(final_sub_dirs)):
	sub_dir_files = os.listdir(img_dir+final_sub_dirs[x])
	while len(sub_dir_files) > 3:
		sub_dir_files.pop()
	#while(len(sub_dir_files) > 1):
	#	sub_dir_files.pop()
	for s_d_f in sub_dir_files:
		im = Image.open(img_dir+final_sub_dirs[x]+os.sep+s_d_f)
		# Resize to smaller array for processing
		im = im.resize((IMG_WIDTH, IMG_HEIGHT), PIL.Image.ANTIALIAS)
		pix = np.array(im.getdata())
		img_arr_length = pix.shape[0]
		for i in range(img_arr_length):
			all_classification_points.append(x)
			all_data_points.append((pix[i][0],pix[i][1],pix[i][2]))
		print "Finished loading:", s_d_f
print "Finished loading all data points in training"
print "Start loading all data points in testing"
#while len(test_sub_dirs) > 1:
#	test_sub_dirs.pop()
for x in range(len(test_sub_dirs)):
	im = Image.open(test_dir+test_sub_dirs[x])
	pix = np.array(im.getdata())
	img_arr_length = pix.shape[0]
	test_indices.append(img_arr_length + test_indices[-1])
	for i in range(img_arr_length):
		all_test_points.append((pix[i][0],pix[i][1],pix[i][2]))
	print "Finishing loading:", test_sub_dirs[x]
print "Finished loading all data points in testing"
all_data_points = np.array(all_data_points)
all_classification_points = np.array(all_classification_points)
all_test_points = np.array(all_test_points)
print "Creating Support Vector machine"
rbf_svm_model = LinearSVC()
rbf_svm_model.fit(all_data_points, all_classification_points)
#print "Accuracy:",rbf_svm_model.score(all_data_points, all_classification_points)
#print "support:",rbf_svm_model.n_support_
print "params:",rbf_svm_model.get_params()
print "Creating prediction for data points"
svm_predictions = rbf_svm_model.predict(all_test_points)
for x in range(0, len(test_sub_dirs)):
	tmp_test_img = Image.open(test_dir+test_sub_dirs[x])
	prediction_img = tmp_test_img.copy()
	pixel_x = 0
	pixel_y = 0
	tmp_test_width, tmp_test_height = tmp_test_img.size
	print x
	print test_indices
	print test_indices[x]
	print test_indices[x+1]
	for sub_pred in svm_predictions[test_indices[x]: test_indices[x+1]]:
		tmp_triple = [0, 0, 0]
		if sub_pred == 0:
			# First is civilizations
			tmp_triple = [130, 130, 130]
		elif sub_pred == 1:
			# Second is landscapes
			tmp_triple = [0, 255, 0]
		elif sub_pred == 2:
			# Third is water bodies
			tmp_triple = [0, 0, 255]
		prediction_img.putpixel((pixel_x, pixel_y), (tmp_triple[0], tmp_triple[1], tmp_triple[2]))
		pixel_x += 1
		if pixel_x == tmp_test_width:
			pixel_x = 0
			pixel_y += 1
	#plt.imsave("RBF.png", rbf_Z)
	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(tmp_test_img)
	plt.title("Original Image")
	plt.xlabel('Pixels X-Direction')
	plt.ylabel('Pixels Y-Direction')
	plt.subplot(1,2,2)
	plt.title("Test SVM")
	plt.xlabel('Pixels X-Direction')
	plt.ylabel('Pixels Y-Direction')
	plt.imshow(prediction_img)
	blue_patch = mpatches.Patch(color='blue', label='Water-Body')
	gray_patch = mpatches.Patch(color='gray', label='Civilization')
	green_patch = mpatches.Patch(color='green', label='Vegetation')
	plt.legend(handles=[blue_patch, gray_patch, green_patch])
	plt.savefig("Pred-"+test_sub_dirs[x].split(".")[0]+"-supportvector-2.png")
