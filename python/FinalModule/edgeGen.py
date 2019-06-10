#!/usr/bin/env python

import torch

import cv2
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
from PIL import Image
import gangan
import gc
from time import sleep


import os
from io import BytesIO

import numpy as np
from PIL import Image

import tensorflow as tf
import sys
import datetime

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.cuda.device(1) # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'bsds500'
arguments_strIn = './images/sample.png'
arguments_strOut = './out.png'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		self.moduleVggOne = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleVggTwo = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleVggThr = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleVggFou = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleVggFiv = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

		self.moduleCombine = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			torch.nn.Sigmoid()
		)


		self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))

	# end

	def forward(self, tensorInput):
		tensorBlue = (tensorInput[:, 0:1, :, :] * 255.0) - 104.00698793
		tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762
		tensorRed = (tensorInput[:, 2:3, :, :] * 255.0) - 122.67891434

		tensorInput = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 1)

		tensorVggOne = self.moduleVggOne(tensorInput)
		tensorVggTwo = self.moduleVggTwo(tensorVggOne)
		tensorVggThr = self.moduleVggThr(tensorVggTwo)
		tensorVggFou = self.moduleVggFou(tensorVggThr)
		tensorVggFiv = self.moduleVggFiv(tensorVggFou)

		tensorScoreOne = self.moduleScoreOne(tensorVggOne)
		tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
		tensorScoreThr = self.moduleScoreThr(tensorVggThr)
		tensorScoreFou = self.moduleScoreFou(tensorVggFou)
		tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)

		tensorScoreOne = torch.nn.functional.interpolate(input=tensorScoreOne, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreTwo = torch.nn.functional.interpolate(input=tensorScoreTwo, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreThr = torch.nn.functional.interpolate(input=tensorScoreThr, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreFou = torch.nn.functional.interpolate(input=tensorScoreFou, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreFiv = torch.nn.functional.interpolate(input=tensorScoreFiv, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)

		return self.moduleCombine(torch.cat([ tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv ], 1))
	# end
# end

moduleNetwork = Network().cuda().eval()

##########################################################

def estimate(tensorInput):
	intWidth = tensorInput.size(2)
	intHeight = tensorInput.size(1)

	assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	return moduleNetwork(tensorInput.cuda().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
# end

##########################################################



import os
from io import BytesIO

import numpy as np
from PIL import Image

import tensorflow as tf
import sys
import datetime


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    start = datetime.datetime.now()

    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    end = datetime.datetime.now()

    diff = end - start
    print("Time taken to evaluate segmentation is : " + str(diff))

    return resized_image, seg_map

def drawSegment(baseImg, matImg):
  width, height = baseImg.size
  dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
  for x in range(width):
            for y in range(height):
                color = matImg[y,x]
                (r,g,b) = baseImg.getpixel((x,y))
                if color == 0:
                    dummyImg[y,x,3] = 0
                else :
                    dummyImg[y,x] = [r,g,b,255]
  img = Image.fromarray(dummyImg)
  filenames3 = os.listdir('userOutput')
  filenames3.sort()
  img.save(outputFilePath+filenames3[0].split('.')[0]+'.png')

inputFilePath = 'userOutput/'
outputFilePath = 'userResult/'

def run_visualization(filepath):
  """Inferences DeepLab model and visualizes result."""
  try:
	  filenames4 = os.listdir('userOutput')
	  filenames4.sort()
	  f = open(filepath + filenames4[0])
	  jpeg_str = open(filepath + filenames4[0], "rb").read()
	  orignal_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check file: ' + filepath)
    return

  print('running deeplab on image %s...' % filepath)
  resized_im, seg_map = MODEL.run(orignal_im)

  # vis_segmentation(resized_im, seg_map)
  drawSegment(resized_im, seg_map)












if __name__ == '__main__':

	while True :

		iosInput = os.listdir('/home/chanil/Website_Django/Vcsite/mainapp/media/input')
		iosInput.sort()
		if len(iosInput) == 1 :
			img = cv2.imread('/home/chanil/Website_Django/Vcsite/mainapp/media/input' + iosInput[0],)
			cv2.imwrite('input'+iosInput[0], img)


		initfilenames = os.listdir('input')
		initfilenames.sort()
		print(initfilenames)
		print(len(initfilenames))
		ohyeah = gangan
		if len(initfilenames) == 1 :
			sleep(1)

			print("들어옴")
			filenames = os.listdir('input')
			filenames.sort()
			userFile = ""
			if len(filenames) == 0:
				print("Not Exist")

			print(filenames)
			count9 = 0
			for filename in filenames:
				img9 = cv2.imread('input/' + filename, )
				userFile = filename

				img9 = cv2.resize(img9, (256, 256))
				cv2.imwrite('input3/' + filename , img9)
				count9 = count9 + 1

			for filename in filenames:
				tensorInput = torch.FloatTensor(
					numpy.array(cv2.resize(cv2.imread("input/" + filename, ), (480, 320)))[:, :, ::-1].transpose(2, 0,1).astype(numpy.float32) * (1.0 / 255.0))

				tensorOutput = estimate(tensorInput)

				PIL.Image.fromarray(
					(tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(
					"output_HED/" + filename)

			filenames2 = os.listdir('output_HED')
			filenames2.sort()

			if len(filenames2) == 0:
				print("Not Exist")

			count = 0
			for filename in filenames2:
				img0 = cv2.imread('output_HED/' + filename, )
				hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
				low_Val = numpy.array([0, 0, 0])
				high_Val = numpy.array([200, 200, 200])
				maskWhite = cv2.inRange(hsv, low_Val, high_Val)
				img0 = cv2.bitwise_and(img0, img0, mask=maskWhite)

				# the Threshold for Edge Detection Function
				# TODO : make the threshold about image

				thresholdLow = [100]
				thresholdHigh = [400]

				# GaussianBlur : maybe it already exist in canny alg. check and erase.
				# img = cv2.GaussianBlur(img0,(3,3),0)

				for j in range(0, 1):
					for i in range(0, 1):
						edges = cv2.Canny(img0, thresholdLow[i], thresholdHigh[j])
						edges = numpy.invert(edges)
						edges = cv2.resize(edges, (256, 256))
						cv2.imwrite('output_Canny/' + filename, edges)
						print('output_' + str(count) + '_' + str(j * 10 + i))
				result = Image.new("RGB", (512, 256))
				imgB = Image.open('output_Canny/' + filename)
				imgA = Image.open('input3/' + filename)
				result.paste(im=imgA, box=(0, 0))
				result.paste(im=imgB, box=(256, 0))
				id = ''
				if filename.split('.')[0] == '11111' :
					id = 'bicycle'
				if filename.split('.')[0] == '99999' :
					id = 'car'

				result.save('datasets/'+id+'/val/' + filename )


			ohyeah.test()
			filenames3 = os.listdir('userOutput')
			filenames3.sort()

			if len(filenames3) == 0:
				print("Not Exist")

			print(filenames3)

			for filename in filenames3:
				MODEL = DeepLabModel("xception_model")
				run_visualization(inputFilePath)
				img9 = cv2.imread('userResult/' + filename.split('.')[0] + '.png', cv2.IMREAD_UNCHANGED)

				# make mask of where the transparent bits are
				trans_mask = img9[:, :, 3] == 0

				# replace areas of transparency with white and not transparent
				img9[trans_mask] = [255, 255, 255, 255]

				# new image without alpha channel...
				new_img = cv2.cvtColor(img9, cv2.COLOR_BGRA2BGR)
				cv2.imwrite('/home/chanil/Website_Django/Vcsite/mainapp/media/' + userFile.split(".")[0] + '.' + "jpg", new_img)

			file1 = 'userOutput/' + initfilenames[0]
			file2 = 'output_Canny/' + initfilenames[0]
			file3 = 'output_HED/' + initfilenames[0]
			file4 = 'image3/' + initfilenames[0]
			file5 = 'input/' + initfilenames[0]
			file6 = 'datasets/'+ 'car' +'/val/' + initfilenames[0]
			file7 = '/home/chanil/Website_Django/Vcsite/mainapp/media/input/' + initfilenames[0]
			file8 = 'datasets/'+ 'bicycle' +'/val/' + initfilenames[0]
			file9 = 'datasets/'+ 'bag' +'/val/' + initfilenames[0]
			file10 = 'datasets/'+ 'shoes' +'/val/' + initfilenames[0]



			filenames = os.listdir('input')
			filenames.sort()
			inputname = filenames[0]
        
			if os.path.isfile(file1):
				os.remove(file1)

			if os.path.isfile(file2):
				os.remove(file2)

			if os.path.isfile(file3):
				os.remove(file3)

			if os.path.isfile(file4):
				os.remove(file4)

			if os.path.isfile(file5):
				os.remove(file5)

			if os.path.isfile(file6):
				os.remove(file6)

			if os.path.isfile(file7):
				os.remove(file7)
			if os.path.isfile(file8):
				os.remove(file8)
			if os.path.isfile(file9):
				os.remove(file9)
			if os.path.isfile(file10):
				os.remove(file10)
			
		else :
			print("도는중 기모찌")
			sleep(1)
			continue



