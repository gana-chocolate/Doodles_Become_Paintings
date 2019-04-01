import cv2
import numpy as np
from matplotlib import pyplot as plt


# image input
# TODO : compatibility 'jpg / png / jpeg'
img0 = cv2.imread('images/1.jpg',)

# image size : width / height
imgWidth, imgHeight, _ = img0.shape

# the Threshold for Edge Detection Function
# TODO : make the threshold about image

threshold = 0

if imgWidth > imgHeight:
    threshold = imgWidth
else:
    threshold = imgHeight

# GaussianBlur : maybe it already exist in canny alg. check and erase.
#img = cv2.GaussianBlur(img0,(3,3),0)

# threshold array is needed for edge examples.
edges = cv2.Canny(img0,200,500)

# swap black and white
edges = np.invert(edges)

# image OUTPUT
# TODO : autosaving function
cv2.imwrite('images/7.png',edges)



# Show window. for testing
plt.subplot(121),plt.imshow(img0,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()