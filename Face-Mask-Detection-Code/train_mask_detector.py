import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths


Initial_Learning_Rate = 1e-5 #Initial_Learning_Rate, we are making our learning rate very less, due to this the loss will be calculated accurately, and accuracy will be maintained
PASS = 15 # so this is basically  the number of passes or the periods in which we train our dataset
Batch_Size = 49 #Batch_Size  so this refers to the number of training examples trained in a single pass.

DIRECTORY = r"D:\CODE\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]


Data_Sets = [] # Data_Set
labels = []

for cat in CATEGORIES:
    path = os.path.join(DIRECTORY, cat)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)
    	Data_Sets.append(image)
    	labels.append(cat)
		
Label_Binarizer = LabelBinarizer() #Label_Binarizer
labels = Label_Binarizer.fit_transform(labels)
labels = to_categorical(labels)

Data_Sets = np.array(Data_Sets, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(Data_Sets, labels,	test_size=0.15, stratify=labels, random_state=30)

# whenever we are having lesser dataset we use ImagedataGenerator
aug = ImageDataGenerator(    # so basically this creates many images with single image by adding properties to it, so we can create more dataset with this.
	rotation_range=30,    # so ImageDataGenerator is basically used to generate many images with single image by changing some properties.
	zoom_range=0.20,
	width_shift_range=0.3,
	height_shift_range=0.3,
	shear_range=0.17,
	horizontal_flip=True,
	fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False, # here we are creating a model, in which there is a parameter i.e. weights="imagenet". so this imagenet is nothing but there are pretrained models specifically for images, so when we use this imagenet those weights will be initialized for us and which will us better results.
	input_tensor=Input(shape=(224, 224, 3))) # inputtensor is nothing but the shape of the image (we are inputing images of 224 by 224(height and width respectively) and 3 is nothing but 3 channels of colors as we are inputing colored images so these 3 denoted 3 channels mainly RGB) 
# include_top=flase is nothing but a boolean value which is to say whether to include fully connected layer at the top of the network, so we connect the fully connected layer

# so the below code is to construct the fully connected layer by pooling.
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel) #so here the pool size is 7 by 7
headModel = Flatten(name="flatten")(headModel) # flatenning the layers(flatenning is nothing but converting the data into 1D array to input it to the next layer)
headModel = Dense(138, activation="relu")(headModel)  # adding the dense layer (dense layer is a layer of neurons in which each neuron recieves input from all neurons of previous layer)  here 'relu' is the go to activation function for non linear images, mostly this is used for worst case of images (also we are using 138 neurons to train dataset)
headModel = Dropout(0.5)(headModel)   # to avoide overfitting of the model we use dropout layer(badically overfitting is nothing but it occurs when our statistical model doesn't fits exactly as our training model.)
headModel = Dense(2, activation="sigmoid")(headModel)  # this is for output model, our output model is of 2 layers i.e. one for with mask and other for without mask, ( we are using sigmoid activation function also we can use softmax activation function as they are probability based 0s or  1s values) since here we are dealing with binary classification we have taken sigmoid function


model = Model(inputs=baseModel.input, outputs=headModel) # calling of model function, so it accepts 2 parameters one is inputs and other is output so input is basemodel.input and output will be the headmodel


for layer in baseModel.layers:   # here we are freezing the layers in the base model, the reason for doing this is they will not be updated during the first running process, as they are just replacement for our convolution neural networks(CNN) so we are just freezing them for training
	layer.trainable = False

# once it is done we are giving our initial learning rate to 1 e-5 (1 e to the power -5) 
# optimization
optimize_model = Adam(lr=Initial_Learning_Rate, decay=Initial_Learning_Rate / PASS) # this decay function is for training neural network, basically it starts training the network with a learning rate and then slowly reducing /decaying until local minima is obtained.
model.compile(loss="binary_crossentropy", optimizer=optimize_model, # the loss function is for evaluating how well specific algorithm models the given data. and the binarycrossentropy basically compares each of the predicted probabilities to actual class output which can be either 0 or 1. it basically determines how close or accurate we are to our model, and we are using the adam optimizer, which is similar to relu which commonly used optimizer for any image prediction methods
	metrics=["accuracy"])  # accuracy mertix is the simplest way to calculate the accuracy of our model, it calculates the accuracy using number of correclty classified point in test data and total number of points in the test data. this accuracy matrix is called Confusion matrix6

# training_head
# so now we are fitting our model and the image data generator that we used in the beginning is flowed here so we get more training data to train images
training_head = model.fit(
	aug.flow(trainX, trainY, batch_size=Batch_Size),
	steps_per_epoch=len(trainX) // Batch_Size,
	validation_data=(testX, testY),   # for data validation we are using testX and testY which are the testing datasets
	validation_steps=len(testX) // Batch_Size,
	epochs=PASS) # the number of passes in which our datasets will be trained, we've given 15.



predIdxs = model.predict(testX, batch_size=Batch_Size) # evaluating our network by using model.predict, so it accepts one argument i.e. the new data and returns learned label for each object or the data in the array


predIdxs = np.argmax(predIdxs, axis=1) # for each image in the dataset we need to find the index of the label corresponding to the largest predicted probability, so basically it returns indices of the max element of the array in a particular axis.

# from here comments are left. 
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=Label_Binarizer.classes_))



model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
Number = PASS  # Number
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, Number), training_head.history["loss"], label="train_loss")
plt.plot(np.arange(0, Number), training_head.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, Number), training_head.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, Number), training_head.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Pass #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")