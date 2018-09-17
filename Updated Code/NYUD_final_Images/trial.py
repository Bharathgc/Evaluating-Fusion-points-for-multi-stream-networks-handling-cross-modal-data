import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
def transform_image(img,shear_range,trans_range):
	tr_x = trans_range*np.random.uniform()-trans_range/2
	tr_y = trans_range*np.random.uniform()-trans_range/2
	Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
	pts1 = np.float32([[5,5],[20,5],[5,20]])
	pt1 = 5+shear_range*np.random.uniform()-shear_range/2
	pt2 = 20+shear_range*np.random.uniform()-shear_range/2
image = mpimg.imread('NYU0001_rgb.jpg')
img = transform_image(image,10,5)
plt.imshow(image);
plt.axis('off');
plt.show()
