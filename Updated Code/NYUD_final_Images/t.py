import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2
def transform_image(img,shear_range,trans_range):
	rows,cols,ch = img.shape 
	tr_x = trans_range*np.random.uniform()-trans_range/2
	tr_y = trans_range*np.random.uniform()-trans_range/2
	Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
	pts1 = np.float32([[5,5],[20,5],[5,20]])
	pt1 = 5+shear_range*np.random.uniform()-shear_range/2
	pt2 = 20+shear_range*np.random.uniform()-shear_range/2
	pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
	shear_M = cv2.getAffineTransform(pts1,pts2)
	img = cv2.warpAffine(img,Trans_M,(cols,rows))
	img = cv2.warpAffine(img,shear_M,(cols,rows))
	return img
image = mpimg.imread('NYU0001_rgb.jpg')
img = transform_image(image,7,5)
plt.imshow(img)
plt.axis('off')
plt.show()
