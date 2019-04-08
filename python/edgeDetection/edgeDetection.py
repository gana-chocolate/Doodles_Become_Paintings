import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


# TODO : HED detection 앞으로 몰아오기.

# input 들어오는 파일 이름 받고, mac 환경일 경우 예외 파일 걸러주기
filenames = os.listdir('input')
filenames.sort()
filenames.remove('.DS_Store')

if len(filenames) == 0:
    print
    "Not exist"
print(filenames)
for filename in filenames :
    # image input
    # TODO : compatibility 'jpg / png / jpeg'
    img0 = cv2.imread('input/' + filename,)

    # image size : width / height
    imgWidth, imgHeight, _ = img0.shape

    # the Threshold for Edge Detection Function
    # TODO : make the threshold about image

    thresholdLow = [100,100,200,200,300,300,400,400]
    thresholdHigh = [200,300,300,400,400,500,500,600]

    # GaussianBlur : maybe it already exist in canny alg. check and erase.
    #img = cv2.GaussianBlur(img0,(3,3),0)

    for i in range(0 , 8):
        edges = cv2.Canny(img0, thresholdLow[i], thresholdHigh[i])
        edges = np.invert(edges)
        cv2.imwrite('output/' + 'output_' + str(i) + '_' + filename,edges)

    # threshold array is needed for edge examples.
    #edges = cv2.Canny(img0,200,500)

    # swap black and white
    #edges = np.invert(edges)

    # image OUTPUT
    # TODO : autosaving function
    #cv2.imwrite('images/'+'output_'+filename,edges)



# Show window. for testing
#plt.subplot(121),plt.imshow(img0,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()