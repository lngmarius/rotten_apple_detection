import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from colorthief import ColorThief


path =r'path'
img = cv.imread(path)

img = cv.medianBlur(img,7)
cv.imshow('blurred image',img)

Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# convert back into uint8, and make original image
center = np.uint8(center)
print(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv.imshow('KMeans Clustering',res2)


#cv.imwrite(r'path', res2)
#color_thief = ColorThief(r'path')
#dominant_color = color_thief.get_color(quality=1)
#palette = color_thief.get_palette(color_count=6)
#print(dominant_color)
#print(palette)


gri = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
(thresh1, I1) = cv.threshold(gri,127,255,cv.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(I1,kernel,iterations = 0)
dilation = cv.dilate(I1,kernel,iterations = 1)
plt.subplot(131), plt.imshow(img, 'gray'),plt.title('Imagine originala')
plt.subplot(132), plt.imshow(erosion,'gray'),plt.title('Imagine erodata')
plt.subplot(133), plt.imshow(dilation,'gray'),plt.title('Imagine dilatata')
plt.show()
cv.waitKey(1000)

#with np.printoptions(threshold=np.inf):
#    print(erosion)



number_of_white_pix = np.sum(erosion == 255)
number_of_black_pix = np.sum(erosion == 0)
number_of_pixels=number_of_white_pix+number_of_black_pix

print('Number of pixels:', number_of_pixels)
print('Number of white pixels:', number_of_white_pix)
print('Number of black pixels:', number_of_black_pix)

procent_white=number_of_white_pix/number_of_pixels*100
procent_black=number_of_black_pix/number_of_pixels*100

#print(procent_white,'% of whiel pixels')
#print(procent_black,'% of black pixels')


if(procent_black > 5 and procent_black != 0):
    img = cv.putText(img, 'Rotten!', (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 1, cv.LINE_AA)
else:
    img = cv.putText(img, 'Healthy!', (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                     0.7, (255, 0, 0), 1, cv.LINE_AA)

cv.imshow('Output', img)

cv.waitKey(0)
