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
import argparse
import os
import scipy.misc
import numpy as np

from model import pix2pix
import tensorflow as tf



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
	count = 0
	for filename in filenames2 :
	    img0 = cv2.imread('output_HED/' + filename,)
	    hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
	    low_Val = numpy.array([0,0,0])
	    high_Val = numpy.array([200,200,200])
	    maskWhite = cv2.inRange(hsv,low_Val,high_Val)
	    img0 = cv2.bitwise_and(img0,img0,mask=maskWhite)
	    
	    
	    

	    # the Threshold for Edge Detection Function
	    # TODO : make the threshold about image

	    thresholdLow = [100,150,200]
	    thresholdHigh = [200,300]

	    # GaussianBlur : maybe it already exist in canny alg. check and erase.
	    #img = cv2.GaussianBlur(img0,(3,3),0)

	    for j in range(0 , 2):
	        for i in range(0 , 3):
	    	    edges = cv2.Canny(img0, thresholdLow[i], thresholdHigh[j])
	    	    edges = numpy.invert(edges)
	    	    edges = cv2.resize(edges,(256,256))
	    	    #cv2.imwrite('output_Canny/' +filename.split(".")[0]+'_'+ str(i*100+j) + '.' + filename.split(".")[1],edges)
	    	    cv2.imwrite('output_Canny/' + str(count) +'_'+ str(j*10+i) + '.' + filename.split(".")[1],edges)
	    	    print('output_' + str(count) +'_'+ str(j*10+i))
	    result = Image.new("RGB",(512, 256))
	    imgA = Image.open('output_Canny/' + str(count) +'_'+ str(0) + '.' + filename.split(".")[1])
	    imgB = Image.open('output_Canny/' + str(count) +'_'+ str(12) + '.' + filename.split(".")[1])
	    result.paste(im=imgA, box=(0, 0))
	    result.paste(im=imgB, box=(256, 0))
	    result.save('input2/' + str(count) + '.jpg')
	    count = count + 1
# end



parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='car2', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')

args = parser.parse_args()
args.phase = 'test'
args.dataset_name = 'car'
args.test_dir = './output'
def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    
    with tf.Session() as sess:
        model = pix2pix(sess, image_size=args.fine_size, batch_size=args.batch_size,
                        output_size=args.fine_size, dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)
            
            if args.phase == 'train':
                model.train(args)
                    else:
                        model.test(args)

if __name__ == '__main__':
    tf.app.run()


