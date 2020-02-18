import cv2
import matplotlib.pyplot as plt
import numpy as np

bgr_img = cv2.imread('paper.png')
rgb_img_ori = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
rgb_img_proc = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

r = rgb_img_proc[:,:,0]
g = rgb_img_proc[:,:,1]
b = rgb_img_proc[:,:,2]

for i in range(g.shape[0]) :
    for j in range(g.shape[1]) :
        if g[i][j] * 1.2 > 255 :
            g[i][j] = 255
        else :
            g[i][j] = g[i][j] * 1.2

images = [rgb_img_ori, cv2.merge((g, g, b))]
for item in images :
    plt.figure()
    plt.imshow(item)

plt.show()
