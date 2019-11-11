#!/usr/bin/env python
# coding:utf-8

import os
import cv2
import time
import utils
import collections
import requests
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np

from PIL import Image
from tflite_runtime.interpreter import Interpreter


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


#from profilehooks import profile # pip install profilehooks


class Camera(object):
    __metaclass__ = utils.Singleton

    def __init__(self, width=640, height=480):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
		
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')        
        self.video.set(3, width)
        self.video.set(4, height)
        self.width = int(self.video.get(3))
        self.height = int(self.video.get(4))
        print('%sx%s' % (self.width, self.height))


    #@profile
    def classify_image(interpreter, image, top_k=1):
		"""Returns a sorted array of classification results."""
		set_input_tensor(interpreter, image)
		interpreter.invoke()
		output_details = interpreter.get_output_details()[0]
		output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
		if output_details['dtype'] == np.uint8:
			scale, zero_point = output_details['quantization']
			output = scale * (output - zero_point)

		ordered = np.argpartition(-output, top_k)
		return [(i, output[i]) for i in ordered[:top_k]]

	#@profile
	def encode_image_to_jpeg(self, image):
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.
		ret, jpeg = cv2.imencode('.jpg', image,(cv2.IMWRITE_JPEG_QUALITY, 80))
		return jpeg.tobytes()
   	
	#@profile
	def image_request():
		ret,image=self.video.read()
		final=self.encode_image_to_jpeg(self, image)
		return final
	
	#@profile
	def image_classification_render():
		#model configuration
		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser.add_argument('--model', help='File path of .tflite file.', required=True)
		parser.add_argument('--labels', help='File path of labels file.', required=True)
		args = parser.parse_args()
		labels = load_labels(args.labels)
		interpreter = Interpreter(args.model)
		interpreter.allocate_tensors()
		_, height, width, _ = interpreter.get_input_details()[0]['shape']
		
		#image request
		self.image_request()
		
		#image classification
		final_image = Image.open(final).convert('RGB').resize((width, height),Image.ANTIALIAS)
		start_time = time.time()
		results = classify_image(interpreter, final_image)
		elapsed_ms = (time.time() - start_time) * 1000
		label_id, prob = results[0]
		cv2.putText(final_image, str(labels[label_id])+' '+str(prob)+' '+str(elapsed_ms), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(250,25,250), 2)
		self.video.release()
		return final_image
	
	
def main():

	#parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	#parser.add_argument('--model', help='File path of .tflite file.', required=True)
	#parser.add_argument('--labels', help='File path of labels file.', required=True)
	#args = parser.parse_args()
	#labels = load_labels(args.labels)
	#interpreter = Interpreter(args.model)
	#interpreter.allocate_tensors()
	#_, height, width, _ = interpreter.get_input_details()[0]['shape']

    #ret,image=Camera().video.read()
	#final=encode_image_to_jpeg(self, image)
	#final_image = Image.open(final).convert('RGB').resize((width, height),Image.ANTIALIAS)
	#start_time = time.time()
    #results = classify_image(interpreter, final_image)
    #elapsed_ms = (time.time() - start_time) * 1000
    #label_id, prob = results[0]
	#cv2.putText(final, str(labels[label_id])+' '+str(prob)+' '+str(elapsed_ms), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(250,25,250), 2)
	#Camera().video.release()
	print(Camera().image_classification_render())


if __name__ == "__main__":
    main()

	
