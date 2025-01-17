# Author: Daniel Lindberg

# Native python modules
import os
import skimage
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.io
import xgboost as xgb

# Native python submodules
from PIL import Image
from scipy.misc import imread
from skimage.filters.rank import entropy
from skimage.io import imread
from skimage.morphology import disk
from sklearn.decomposition import PCA

def getCovariances(tmp_img):
	new_img = np.zeros((tmp_img.size[0], tmp_img.size[1]))
	for x in range(tmp_img.size[0]):
		for y in range(tmp_img.size[1]):
			new_img[x,y] = sum(tmp_img.getpixel((x,y)))/3.0
	covariance = np.cov(new_img)
	return covariance


def takeEntropy(some_list, some_dir):
	eb_list = []
	all_covariances = []
	for each_item in some_list:
		img = Image.open(some_dir+each_item)
		img_covariance = getCovariances(img)
		all_covariances.append(img_covariance)
		pix = np.array(img.getdata(), dtype=np.uint8)	
		#img = np.array(img)	
		entropy_img = entropy(pix, disk(300, dtype=np.uint8))
		expected_bits = entropy_img/math.log(2)
		temp_PCA = PCA()
		temp_PCA.fit(expected_bits)
		fig = plt.figure()
		plt.subplot(3,1,1)
		plt.title("Original Image")
		plt.imshow(img)
		plt.subplot(3,1,2)
		plt.title("Entropy Expected Bits")
		plt.plot(expected_bits)
		plt.subplot(3,1,3)
		plt.title("PCA Components")
		plt.ylabel("Principal Components")
		plt.xlabel("Planes")
		plt.plot(temp_PCA.components_)
		blue_patch = mpatches.Patch(color='blue', label='1st plane')
		orange_patch = mpatches.Patch(color='orange', label='2nd plane')
		green_patch = mpatches.Patch(color='green', label='3rd plane')
		plt.legend(handles=[blue_patch, orange_patch, green_patch])
		fig.subplots_adjust(hspace=0.5)
		plt.savefig(some_dir+(each_item.split(".")[0])+"_expected_bits.png")
		print each_item, "[bits, pca]:", expected_bits.shape, ",", temp_PCA.components_.shape
		eb_list.append(expected_bits.shape[0])
	return eb_list, all_covariances

# Our two Directories
random_images_dir = "random_images/"
original_images_dir = "original_images/"

# get all of the files listed
all_files = os.listdir(random_images_dir)
all_files.sort()

original_files = os.listdir(original_images_dir)
original_files.sort()

# Refine all of the lists
finalized_list = []
for x in range(len(all_files)):
	if (all_files[x].endswith("512.jpg")):
		finalized_list.append(all_files[x])

original_finalized_list = []
for x in range(len(original_files)):
	if (original_files[x].endswith("512.jpg")):
		original_finalized_list.append(original_files[x])

# List of expected bits
random_eb_list = []
original_eb_list = []

random_eb_list, random_covariances = takeEntropy(finalized_list, random_images_dir)
original_eb_list, original_covariances = takeEntropy(original_finalized_list, original_images_dir)

# This is for part 1
print random_eb_list
print original_eb_list

indexer = 0
for sub_cov in random_covariances:
	print "------------------\nWorking on image:", finalized_list[indexer].split(".")[0]
	print "Part 2:", sub_cov.shape, np.average(sub_cov)
	temp_PCA = PCA()
	temp_PCA.fit(sub_cov)
	temp_components = temp_PCA.components_
	plt.figure()
	plt.subplot(1,2,1)
	plt.title("Image form Principal Components")
	plt.imshow(temp_components)
	plt.subplot(1,2,2)
	plt.title("Scatterplot form Principal Components")
	plt.plot(temp_components)
	plt.xlabel("Principal Components over each voariance component")
	plt.ylabel("Intensity of Component")
	plt.savefig(random_images_dir+finalized_list[indexer].split(".")[0]+"_PCA_Cov.png")
	print "Part 3:", finalized_list[indexer].split(".")[0], ":",  np.std(temp_components)
	temp_xy_components = []
	for x in range(temp_components.shape[0]):
		for y in range(temp_components.shape[1]):
			temp_xy_components.append((temp_components[x,y], (x,y)))
	temp_xy_components.sort(key=lambda x:x[0])	
	print "Part 4 Three largest:", temp_xy_components[-1], temp_xy_components[-2], temp_xy_components[-3]
	rgb_utility = np.zeros((temp_components.shape[0], temp_components.shape[1], 3))
	for x in range(temp_components.shape[0]):
		for y in range(temp_components.shape[0]):
			rgb_utility[x,y] = [temp_xy_components[-1][0]*255, temp_xy_components[-2][0]*255, temp_xy_components[-3][0]*255]
	plt.figure()
	plt.imshow(rgb_utility)
	plt.savefig(random_images_dir+finalized_list[indexer].split(".")[0]+"_RGB_UTIL.png")
	indexer +=1
	
print "doing originals now"
indexer = 0
for sub_cov in original_covariances:
	print "------------------\nWorking on image:", original_finalized_list[indexer].split(".")[0]
	print "Part 2:", sub_cov.shape, np.average(sub_cov)
	temp_PCA = PCA()
	temp_PCA.fit(sub_cov)
	temp_components = temp_PCA.components_
	plt.figure()
	plt.subplot(1,2,1)
	plt.title("Image form Principal Components")
	plt.imshow(temp_components)
	plt.subplot(1,2,2)
	plt.title("Scatterplot form Principal Components")
	plt.plot(temp_components)
	plt.xlabel("Principal Components over each voariance component")
	plt.ylabel("Intensity of Component")
	plt.savefig(original_images_dir+original_finalized_list[indexer].split(".")[0]+"_PCA_Cov.png")
	print "Part 3:", original_finalized_list[indexer].split(".")[0], ":",  np.std(temp_components)
	temp_xy_components = []
	for x in range(temp_components.shape[0]):
		for y in range(temp_components.shape[1]):
			temp_xy_components.append((temp_components[x,y], (x,y)))
	temp_xy_components.sort(key=lambda x:x[0])	
	print "Part 4 Three largest:", temp_xy_components[-1], temp_xy_components[-2], temp_xy_components[-3]
	rgb_utility = np.zeros((temp_components.shape[0], temp_components.shape[1], 3))
	for x in range(temp_components.shape[0]):
		for y in range(temp_components.shape[0]):
			rgb_utility[x,y] = [temp_xy_components[-1][0]*255, temp_xy_components[-2][0]*255, temp_xy_components[-3][0]*255]
	plt.figure()
	plt.imshow(rgb_utility)
	plt.savefig(original_images_dir+original_finalized_list[indexer].split(".")[0]+"_RGB_UTIL.png")
	indexer +=1


print "Starting part 6"

dtrain_train_data = np.array([])
dtrain_label_data = np.array([])
red_data = []
blue_data = []
green_data = []
has_read = False
for eti_image in original_finalized_list:
	if "eit" in eti_image and not has_read:
		dtrain_train_data = imread(original_images_dir+eti_image)
		im = Image.open(original_images_dir+eti_image)
		pix = im.getdata()
		for x in range(len(pix)):
			red_data.append(pix[x][0])
			green_data.append(pix[x][1])
			blue_data.append(pix[x][2])
		#dtrain_train_data = np.append(dtrain_train_data, im)
		has_read = True
	if "hmiigr" in eti_image:
		dtrain_label_data = imread(original_images_dir+eti_image)
		#dtrain_label_data = np.append(dtrain_label_data, im)
dtrain_train_data = np.array(dtrain_train_data)
dtrain_label_data = np.array(dtrain_label_data)
print dtrain_train_data.shape
print dtrain_label_data.shape
new_train = np.zeros((dtrain_train_data.shape[0], dtrain_train_data.shape[1]))
new_label = np.zeros((dtrain_train_data.shape[0], dtrain_train_data.shape[1]))
for x in range(dtrain_train_data.shape[0]):
	for y in range(dtrain_train_data.shape[1]):
		new_train[x,y] = sum(dtrain_train_data[x,y])/3.0
		new_label[x,y] = sum(dtrain_label_data[x,y])/3.0
dtrain = xgb.DMatrix(new_train, label=new_label)
dtrain.save_binary('train.buffer')
plt.figure()
plt.plot(red_data)
plt.plot(green_data)
plt.plot(blue_data)
plt.title("Dtrain Data")
plt.savefig("Dtrain_scatter.png")
plt.figure()
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
num_round = 10

bst = xgb.train(param, dtrain, num_round)
ypred = bst.predict(dtrain)
plt.figure()
plt.plot(ypred)
plt.title("Prediction HMI Continum)")
plt.savefig("HMI_Continum.png")
#xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees=3)
plt.savefig("HMI_TREE.png")

bst.__del__()


print "Starting part 6"

dtrain_train_data2 = np.array([])
dtrain_label_data2 = np.array([])
has_read = False
for eti_image in finalized_list:
	if "eit" in eti_image and not has_read:
		dtrain_train_data2 = imread(random_images_dir+eti_image)
		#dtrain_train_data = np.append(dtrain_train_data, im)
		has_read = True
	if "hmiigr" in eti_image:
		dtrain_label_data2 = imread(random_images_dir+eti_image)
		#dtrain_label_data = np.append(dtrain_label_data, im)
dtrain_train_data2 = np.array(dtrain_train_data2)
dtrain_label_data2 = np.array(dtrain_label_data2)
print dtrain_train_data2.shape
print dtrain_label_data2.shape
new_train2 = np.zeros((dtrain_train_data2.shape[0], dtrain_train_data2.shape[1]))
new_label2 = np.zeros((dtrain_label_data2.shape[0], dtrain_label_data2.shape[1]))
for x in range(dtrain_train_data2.shape[0]):
	for y in range(dtrain_train_data2.shape[1]):
		new_train2[x,y] = sum(dtrain_train_data2[x,y])/3
for x in range(dtrain_label_data2.shape[0]):
	for y in range(dtrain_label_data2.shape[1]):
		new_label2[x,y] = sum(dtrain_label_data2[x,y])/3
dtrain2 = xgb.DMatrix(new_train2, label=new_label)
dtrain2.save_binary('train2.buffer')

plt.figure()
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
num_round = 10
bst2 = xgb.train(param, dtrain2, num_round)
ypred2 = bst2.predict(dtrain2)
plt.figure()
plt.plot(ypred2)
plt.title("Prediction HMI Continum)")
plt.savefig("HMI_Continum2.png")
#xgb.plot_importance(bst)
xgb.plot_tree(bst2, num_trees=4)
plt.savefig("HMI_TREE2.png")



