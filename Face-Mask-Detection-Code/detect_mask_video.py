# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
def mask_prediction(frame, fNet, mNet):
	# grab the dimensions of the frame and then construct a binary large object
	# from it
	(h, w) = frame.shape[:2]
	binary_large_object = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the face detections
	fNet.setInput(binary_large_object)
	detect = fNet.forward()
	print(detect.shape)
	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locations = []
	predictions = []
	# loop over the detections
	for i in range(0, detect.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		probability = detect[0, 0, i, 2]
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if probability > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			boundingbox = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = boundingbox.astype("int")
			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locations.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on all
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		predictions = mNet.predict(faces, batch_size=49)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locations, predictions)

# load our serialized face detector model from disk
txtPath = r"D:\CODE\Face-Mask-Detection-master\face_detector\deploy.prototxt"
weightPath = r"D:\CODE\Face-Mask-Detection-master\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
fNet = cv2.dnn.readNet(txtPath, weightPath)

# load the face mask detector model from disk
mNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locations, predictions) = mask_prediction(frame, fNet, mNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (boundingbox, pred) in zip(locations, predictions):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = boundingbox     
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		lbl = "Mask" if mask > withoutMask else "No Mask"
		clr = (255, 0, 0) if lbl == "Mask" else (0, 0, 255)

		# include the probability in the label
		#label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, lbl, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), clr, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()