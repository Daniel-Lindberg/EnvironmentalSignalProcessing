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
import tensorflow as tf

#Native python submodules
from collections import Counter
from PIL import Image
from tensorflow import keras
from ann_visualizer.visualize import ann_viz

# Smallest image width and height in training, set it small for processing
IMG_WIDTH=299
IMG_HEIGHT=168

# Function to find the highest count in a list
# Used on neural network predictions
def mostFrequent(input_list): 
    return max(set(input_list), key = input_list.count) 

def secondFrequent(input): 
    # Convert given list into dictionary 
    # it's output will be like {'ccc':1,'aaa':3,'bbb':2} 
    dict = Counter(input) 
    # Get the list of all values and sort it in ascending order 
    value = sorted(dict.values(), reverse=True) 
    # Pick second largest element 
    second_largest = value[1] 
    # Traverse dictionary and print key whose 
    # value is equal to second large element 
    for (key, val) in dict.iteritems(): 
        if val == second_largest: 
            return key 

#Support Vector Machine Code
#----------------------------------------
# Get data points
img_dir = "imgs/"
test_dir = "tests/"
# get all of our classification names
final_sub_dirs = os.listdir(img_dir)
test_sub_dirs = os.listdir(test_dir)

# Get the test_indices for the test
test_indices = [0]

# Arrays for data points and classification
all_data_points = []
all_classification_points = []
all_test_points = []

class_names = np.array(final_sub_dirs)
print "Start loading image files into array for training"
for x in range(len(final_sub_dirs)):
	sub_dir_files = os.listdir(img_dir+final_sub_dirs[x])
	sub_dir_files.sort()
	while len(sub_dir_files) > 1:
		sub_dir_files.pop()
	for s_d_f in sub_dir_files:
		im = Image.open(img_dir+final_sub_dirs[x]+os.sep+s_d_f)
		im = im.resize((IMG_WIDTH, IMG_HEIGHT), PIL.Image.ANTIALIAS)		
		pix = np.array(im.getdata())
		img_arr_length = pix.shape[0]
		for i in range(img_arr_length):
			all_classification_points.append(x)
			all_data_points.append(pix[i])
		print "Finished loading:", s_d_f
print "Finished loading all data points in training"
print "Start loading all data points in testing"
# Un comment this if we wanna limit the amount of tests we have
while len(test_sub_dirs) > 1:
	test_sub_dirs.pop()
for x in range(len(test_sub_dirs)):
	print "Starting test:", test_sub_dirs[x]
	im = Image.open(test_dir+test_sub_dirs[x])
	pix = np.array(im.getdata())
	img_arr_length = pix.shape[0]
	test_indices.append(img_arr_length)
	for i in range(img_arr_length):
		all_test_points.append(pix[i])
	print "Finishing loading:", test_sub_dirs[x]
print "Finished loading all data points in testing"

all_data_points = np.array(all_data_points)
all_classification_points = np.array(all_classification_points)
all_test_points = np.array(all_test_points)

# get the initial test shape
test_shape = all_test_points.shape


print "Training Data shape:", all_data_points.shape
print "Classification Data shape:", all_classification_points.shape
print "Test Data shape:", all_test_points.shape

# Get the full length of the data, used if we want to modify the trianing size
full_length = all_data_points.shape[0]
print "Length of training array:",full_length
print "Shape of training matrix:",all_data_points.shape
all_data_points = all_data_points[0:full_length,:]
print "Updated shape of training matrix:", all_data_points.shape
print "Shape of classification matrix:", all_classification_points.shape
all_classification_points = all_classification_points[0:full_length]
print "Updated shape of classification matrix:", all_classification_points.shape
# Expand the dimensions so it will fit into the neural network
all_data_points = np.expand_dims(all_data_points, axis=2)
all_test_points = np.expand_dims(all_test_points, axis=2)
# Set the training shape to the new expansion
train_shape = all_data_points.shape
# Let's get the original array size
train_shape = (train_shape[1], train_shape[2])

print "Creating keras sequential"
model = keras.Sequential()

# Add the keras Conv1D to express 1 dimension triple of a pixel
model.add(keras.layers.Conv1D(filters=1, kernel_size=1, activation='relu', input_shape=train_shape))
# Make sure to set the input shape to the training shape
model.add(keras.layers.Conv1D(filters=1, kernel_size=1, activation='relu'))
# a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
model.add(keras.layers.Dropout(0.5))
# Flattens out input into the training array, removes 3rd dimension
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(2, activation='relu', input_shape=(train_shape[0],)))
model.add(keras.layers.Dense(test_shape[0], activation='softmax'))
print "Compiling keras model"
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print "Training Neural Network"
history = model.fit(all_data_points, all_classification_points, validation_split=0.25, epochs=5, batch_size=512, verbose=1)
print "Neural Network fully trained"
keras.utils.plot_model(model, show_shapes=True, to_file='NeuralNetwork.png')
# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("Model Accuracy")

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("ModelLoss.png")

ann_viz(model, title="Neural Network Visualizer")
sys.exit()
print "Making predictions model"
predictions = model.predict(all_test_points)
print "Predictions were finished:", predictions.shape
possiblities = list(set(predictions[0]))
# Sorting list of floats in descending 
possiblities.sort(reverse = True) 
# set([0.35695216, 0.20141406, 0.44163376, 0.0])
#sys.exit()
for x in range(len(test_sub_dirs)):
	print "Starting test image:", test_sub_dirs[x]
	tmp_img = Image.open(test_dir+test_sub_dirs[x])
	actual_predictions = tmp_img.copy()
	"""
	We need to refactor the predictions.
	The Neural network returns a list of predictions per data point
	through the various input layers and hidden layers. We need to look
	at evaluations and determine which one is the most frequent.

	Sample output:
	#all_test_points = (50460, 3, 1)
	#predictions = (50460, 1506)
	#new_set = set([0.0017316031, 0.0006632962])
	"""
	new_set = set()
	print test_indices
	print test_indices[x], "-", test_indices[x+1]
	print set(predictions[7])
	indexer = 0
	span = test_indices[x+1] - test_indices[x]
	tmp_test_width, tmp_test_height = tmp_img.size
	#actual_predictions = np.zeros((tmp_img.size[0]*tmp_img.size[1], 3))
	pixel_x = 0
	pixel_y = 0
	for sub_pred in predictions[test_indices[x]:test_indices[x+1]]:	
		if indexer%500 == 0:
			print indexer, "/", span
		#print "sub_list size:", sub_pred.shape
		#pixel_classification = mostFrequent(sub_pred.tolist())
		#if pixel_classification < 0.0002:		
		#	pixel_classification = secondFrequent(sub_pred.tolist())
		pixel_classification = sub_pred[indexer]		
		tmp_triple = [0,0,0]
		if pixel_classification == possiblities[0]:
			# Civilization
			# Set to gray pixel
			tmp_triple = [130,130,130]			
		elif pixel_classification == possiblities[1]:
			# Agriculture
			# Set to green pixel
			tmp_triple = [0, 255, 0]
		elif pixel_classification == possiblities[2]:
			# Waterbody
			# Set to blue pixel
			tmp_triple = [0, 0, 255]		
		#print pixel_x, ",", pixel_y, " of:", tmp_test_width, ",", tmp_test_height
		#else:
		#	# Not valid
		#	tmp_triple = [130,130,130]
		#tmp_triple = np.array(tmp_triple)
		#actual_predictions[indexer] = tmp_triple
		actual_predictions.putpixel((pixel_x,pixel_y), (tmp_triple[0], tmp_triple[1], tmp_triple[2]))
		for s_p in sub_pred:
			new_set.add(s_p)
		indexer += 1
		pixel_x += 1
		if pixel_x == tmp_test_width:
			pixel_y += 1
			pixel_x = 0
	#actual_predictions = np.array(actual_predictions)
	#print "Prediction-Shape:", actual_predictions.shape
	print "Img size:", tmp_img.size
	"""
	Print now that we actually got the actual predictions
	We need to reconstruct them into an image and display it
	"""
	#actual_predictions = np.reshape(actual_predictions, (tmp_img.size[1], tmp_img.size[0], 3))
	#actual_predictions = Image.fromarray(actual_predictions, 'RGB')
	#actual_predictions = actual_predictions.transpose(Image.TRANSPOSE)
	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(tmp_img)
	plt.title("Original Image")
	plt.xlabel('Pixels X-Direction')
	plt.ylabel('Pixels Y-Direction')
	plt.subplot(1,2,2)
	plt.title("Test Neural Network")
	plt.xlabel('Pixels X-Direction')
	plt.ylabel('Pixels Y-Direction')
	plt.imshow(actual_predictions)
	blue_patch = mpatches.Patch(color='blue', label='Water-Body')
	gray_patch = mpatches.Patch(color='gray', label='Civilization')
	green_patch = mpatches.Patch(color='green', label='Vegetation')
	plt.legend(handles=[blue_patch, gray_patch, green_patch])
	#plt.savefig("Pred-"+test_sub_dirs[x].split(".")[0]+"-neuralnet.png")
	print new_set




