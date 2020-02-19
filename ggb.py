import cv2
import matplotlib.pyplot as plt
import numpy as np

#read original
ori_img = cv2.imread('path_to_image')

#OpenCV display image using BGR space by default.
#Therefore. we should it to RGB to achieve orignal image.
#Simply switch B and R matrix column.
temp = ori_img[:, :, 0].copy()
ori_img[:, :, 0] = ori_img[:, :, 2].copy()
ori_img[:, :, 2] = temp

#save R, G and B channel value
r = ori_img[:, :, 0].copy()
g = ori_img[:, :, 1].copy()
b = ori_img[:, :, 2].copy()

#contrast enhance g channel with constant of 1.2
for i in range(g.shape[0]) :
    for j in range(g.shape[1]) :
        if g[i][j] * 1.2 > 255 :
            g[i][j] = 255
        else :
            g[i][j] = g[i][j] * 1.2

#output image
output_img = np.ndarray(shape=ori_img.shape, dtype=int)
output_img[:, :, 0] = b
output_img[:, :, 1] = b
output_img[:, :, 2] = g

#display original image and its output
images = [ori_img, output_img]

for item in images :
    plt.figure()
    plt.imshow(item)
plt.show()
