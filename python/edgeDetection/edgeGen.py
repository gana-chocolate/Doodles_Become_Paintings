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

if __name__ == '__main__':


	filenames = os.listdir('input')
	filenames.sort()

	if len(filenames) == 0:
	    print
	    "Not exist"
	print(filenames)
	for filename in filenames :

		tensorInput = torch.FloatTensor(numpy.array(cv2.resize(cv2.imread("input/"+filename,),(480,320)))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

		tensorOutput = estimate(tensorInput)

		PIL.Image.fromarray((tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save("output_HED/"+filename)


	filenames2 = os.listdir('output_HED')
	filenames2.sort()

	if len(filenames2) == 0:
	    print
	    "Not exist"

	for filename in filenames2 :
	    img0 = cv2.imread('output_HED/' + filename,)
	    #hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
	    #low_Val = numpy.array([0,0,0])
	    #high_Val = numpy.array([200,200,200])
	    #maskWhite = cv2.inRange(hsv,low_Val,high_Val)
	    #img0 = cv2.bitwise_and(img0,img0,mask=maskWhite)
	    
	    
	    

	    # the Threshold for Edge Detection Function
	    # TODO : make the threshold about image

	    thresholdLow = [100,200,300,400]
	    thresholdHigh = [100,150,200,250,300,350,400,450,500,550,600,650,700,750,800]

	    # GaussianBlur : maybe it already exist in canny alg. check and erase.
	    #img = cv2.GaussianBlur(img0,(3,3),0)

	    for i in range(0 , 4):
	        for j in range(0 , 15):
	    	    edges = cv2.Canny(img0, thresholdLow[i], thresholdHigh[j])
	    	    edges = numpy.invert(edges)
	    	    cv2.imwrite('output_Canny/' + 'output_' +filename.split(".")[0]+'_'+ str(i*100+j) + '.' + filename.split(".")[1],edges)
# end
