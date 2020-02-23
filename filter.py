import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def rgb_to_hsv(rgb_img) :
    height = rgb_img.shape[0]
    width = rgb_img.shape[1]
    
    hsv_img = np.ndarray(shape=rgb_img.shape, dtype=rgb_img.dtype)
    for i in range (height) :
        for j in range (width) :
            r = rgb_img[i][j][0].copy() / 255.0
            g = rgb_img[i][j][1].copy() / 255.0
            b = rgb_img[i][j][2].copy() / 255.0

            cmax = max(r, max(g, b))
            cmin = min(r, min(g, b))
            diff = cmax - cmin
            h = -1
            s = -1

            if cmax == cmin :
                h = 0
                
            elif cmax == r :
                h = (60 * ( (g - b) / diff) + 360) % 360

            elif cmax == g :
                h = (60 * ( (b - r) / diff) + 120) % 360

            elif cmax == b :
                h = (60 * ( (r - g) / diff) + 240) % 360

            if cmax == 0 :
                s = 0

            else :
                s = (diff / cmax) * 100

            v = cmax * 100
            
            hsv_img[i][j][0] = h / 2
            hsv_img[i][j][1] = s * 2.55
            hsv_img[i][j][2] = v * 2.55

    return hsv_img

def hsv_to_rgb(hsv_img):
    height = hsv_img.shape[0]
    width = hsv_img.shape[1]

    rgb_img = np.ndarray(shape=hsv_img.shape, dtype=hsv_img.dtype)

    for i in range (height) :
        for j in range (width) :
            h = hsv_img[i, j, 0].copy() * 2
            s = hsv_img[i, j, 1].copy() / 255
            v = hsv_img[i, j, 2].copy() / 255

            c = v * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = v - c

            if h >= 0 and h < 60 :
                r = (c + m) * 255
                g = (x + m) * 255
                b = m * 255

            elif h >= 60 and h < 120 :
                r = (x + m) * 255
                g = (c + m) * 255
                b = m * 255

            elif h >= 120 and h < 180 :
                r = m * 255
                g = (c + m) * 255
                b = (x + m) * 255

            elif h >= 180 and h < 240 :
                r = m * 255
                g = (x + m) * 255
                b = (c + m) * 255

            elif h >= 240 and h < 300 :
                r = (x + m) * 255
                g = m * 255
                b = (c + m) * 255

            else :
                r = (c + m) * 255
                g = m * 255
                b = (x + m) * 255

            rgb_img[i][j][0] = r
            rgb_img[i][j][1] = g
            rgb_img[i][j][2] = b
            
    return rgb_img

def median_filter(img, ksize=5):
    if ksize % 2 == 0 :
        print("Kernel size must be odd number")
        return

    if ksize > img.shape[0] or ksize > img.shape[1]:
        print("Kernel size can not be higher than img")
        return
    
    sliding_height = img.shape[0] - ksize + 1
    sliding_width = img.shape[1] - ksize + 1
    center = int((ksize - 1) / 2)

    for i in range (sliding_height) :
        for j in range (sliding_width) :
            arr = []
            center_x = j + center
            center_y = i + center
            for k in range (i, i + ksize) :
                for l in range (j, j + ksize) :
                    if k == center_y and l == center_x :
                        continue
                    
                    arr.append(img[k][l])

            img[center_y][center_x] = np.median(arr)
            
    return img

if __name__ == "__main__" :
    
    ori_img = cv2.imread('images/img_1.png')

    temp = ori_img[:, :, 0].copy()
    ori_img[:, :, 0] = ori_img[:, :, 2].copy()
    ori_img[:, :, 2] = temp

    hsv_img = rgb_to_hsv(ori_img)
    tes_bener = cv2.cvtColor(ori_img, cv2.COLOR_RGB2HSV)

    h_img = hsv_img[:, :, 0].copy()
    s_img = hsv_img[:, :, 1].copy()
    v_img = hsv_img[:, :, 2].copy()
    
    filtered_hsv = np.ndarray(shape=hsv_img.shape, dtype=hsv_img.dtype)
    filtered_hsv[:, :, 0] = median_filter(h_img)
    filtered_hsv[:, :, 1] = median_filter(s_img)
    filtered_hsv[:, :, 2] = median_filter(v_img)

    tes_bener[:, :, 0] = cv2.medianBlur(tes_bener[:, :, 0], 5)
    tes_bener[:, :, 1] = cv2.medianBlur(tes_bener[:, :, 1], 5)
    tes_bener[:, :, 2] = cv2.medianBlur(tes_bener[:, :, 2], 5)

    filtered_rgb = hsv_to_rgb(filtered_hsv)
    filtered_rgb_bener = cv2.cvtColor(tes_bener, cv2.COLOR_HSV2RGB)

    images = [ori_img, filtered_rgb, filtered_rgb_bener]

    for item in images :
        plt.figure()
        plt.imshow(item)
        plt.xticks([]), plt.yticks([])

    plt.show()
