import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image

aimg = cv2.imread('pic.png', cv2.IMREAD_UNCHANGED)
simg = cv2.resize(aimg,(600,450))
cv2.imwrite('new.png', simg)

img = cv2.imread('new.png', 0)
                  #resize the picture and set it to (600,450) this should change due to different whiteboard size ratio
img = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            cv2.THRESH_BINARY, 11, 2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]


# numpy 2-d array of the binary image
x, y = images[1].shape
z = np.maximum(x, y)
bimg = np.zeros((x, y), np.int)

for i in range(x):
    bimg[i] = images[1][i]                 #this could change to 3 for a better one for there are a lot more words on it----

cv2.imwrite('color_img.jpg', bimg)

# create the window loop, do one for 2*2 and one for 3*3
win2 = np.zeros([2, 2])
after2 = np.zeros((x, y), np.int)
win3 = np.zeros([3, 3])
after3 = np.zeros((x, y), np.int)

for i in range(x - 2 + 1):
    for j in range(y - 2 + 1):
        if (bimg[i, j] == 0 and bimg[i, j + 1] == 0 and bimg[i + 1, j] == 0 and bimg[i + 1, j + 1] == 0):  # 0 is black
            after2[i, j] = 4
        elif (bimg[i, j] == 0 and bimg[i, j + 1] == 0 and bimg[i + 1, j] == 0):
            after2[i, j] = 3
        elif (bimg[i, j] == 0 and bimg[i, j + 1] == 0):
            after2[i, j] = 2
        elif (bimg[i, j] == 0 and bimg[i + 1, j] == 0):
            after2[i, j] = 2

        elif (bimg[i, j] == 0):
            after2[i, j] = 1

cv2.imwrite('after2.jpg', 255 - after2 * 50)

for i in range(x - 3 + 1):
    for j in range(y - 3 + 1):
        w3value = np.array(bimg[i:i + 2, j:j + 2])
        w3add = np.sum(w3value)
        after3[i, j] = w3add

cv2.imwrite('after3.jpg', after3 * 25)


def windowfilter(arr, size):  # perhaps use this to find the centre
    size_plus = int(np.floor(size / 2))
    x, y = arr.shape
    after = np.zeros((x, y), np.int)
    afterr = np.zeros((x + 2 * size_plus, y + 2 * size_plus), np.int)
    afterrr = np.zeros((x, y), np.int)
    for i in range(x):
        for j in range(y):
            if arr[i, j] == 0:
                afterrr[i, j] = 1
            elif arr[i, j] == 255:
                afterrr[i, j] = 0

    # decide the with window plus new arrary --- afterr
    afterr[size_plus:size_plus + x, size_plus:size_plus + y] = afterrr

    for i in range(x):
        for j in range(y):
            w3value = np.zeros((size, size))
            i1 = i + size_plus
            j1 = j + size_plus

            w3value[0:size, 0:size] = afterr[i1 - size_plus:i1 + size_plus, j1 - size_plus:j1 + size_plus]
            w3add = np.sum(w3value)
            after[i, j] = w3add

    # cv2.imwrite('new.jpg', after *5)
    return np.max(after), np.min(after)


# print(windowfilter(bimg,400))


# delete complete white sppace
def whitespace(arr, size):  # perhaps use this to find the centre

    x, y = arr.shape
    after = np.zeros((x - size + 1, y - size + 1),
                     np.int)  # notice this arraysize is changeable this is for the area value not every single  point value

    afterr = np.zeros((x, y), np.int)
    for i in range(x):
        for j in range(y):
            if arr[i, j] == 0:
                afterr[i, j] = 1
            elif arr[i, j] == 255:
                afterr[i, j] = 0

    for i in range(x - size + 1):
        for j in range(y - size + 1):
            w3value = np.zeros((size, size))

            w3value[0:size, 0:size] = afterr[i:i + size, j:j + size]
            w3add = np.sum(w3value)
            after[i, j] = w3add

    # cv2.imwrite('new.jpg', after *5)
    return np.max(after), np.min(after)


# print(whitespace(bimg,321))

# first using log2 to iterate then using 2 or 4 times to iterate
def iterwhite(arr):
    x, y = arr.shape
    max = np.maximum(x, y)
    min = np.minimum(x, y)
    iters = int(np.floor(np.log2(min)))
    print(iters)
    start = 0
    find = 0
    for k in range(iters - 3):
        start = int(np.power(2, k + 4))

        print("---start----")
        print(start)
        a, b = whitespace(arr, start)
        print("-------finish--------")
        if b > 0:
            find = 1
            break
        elif b == 0:
            find = 0

    print(find)
    if find == 1:
        end = int(start)
        start = int(np.power(2, (np.log2(end) - 1)))
    elif find == 0:
        start = start
        end = min

    steps = end - start

    start1 = 0
    # make step to be 40
    iter = int(np.floor((steps / 40)) + 1)
    for k in range(iter):
        start1 = start + (k + 1) * 40

        print("---start----")
        print(start1)
        a, b = whitespace(arr, start1)
        print("-------finish--------")
        if b > 0:
            break

    end1 = start1
    start1 = start1 - 40
    steps = end1 - start1

    start2 = 0
    # make step to be 5
    iter2 = int(np.floor((steps / 5)) + 1)
    for k in range(iter2):
        start2 = start1 + (k + 1) * 5

        print("---start----")
        print(start2)
        a, b = whitespace(arr, start2)
        print("-------finish--------")
        if b > 0:
            break

    end2 = start2
    start2 = start2 - 5
    steps = end2 - start2
    # make step to be 1
    iter2 = int(np.floor((steps / 1)) + 1)
    for k in range(iter2):
        kk = start2 + (k + 1) * 1

        print("---start----")
        print(kk)
        a, b = whitespace(arr, kk)
        print("-------finish--------")
        if b > 0:
            return kk, b


# iterwhite(bimg)
# print(iterwhite(bimg))

# --------------using again whitespace as defined before but this time return different value
def whitespaceplus(arr, size):  # perhaps use this to find the centre

    x, y = arr.shape
    after = np.zeros((x - size + 1, y - size + 1),
                     np.int)  # notice this arraysize is changeable this is for the area value not every single  point value

    afterr = np.zeros((x, y), np.int)
    for i in range(x):
        for j in range(y):
            if arr[i, j] == 0:
                afterr[i, j] = 1
            elif arr[i, j] == 255:
                afterr[i, j] = 0

    for i in range(x - size + 1):
        for j in range(y - size + 1):
            w3value = np.zeros((size, size))

            w3value[0:size, 0:size] = afterr[i:i + size, j:j + size]
            w3add = np.sum(w3value)
            after[i, j] = w3add

    # cv2.imwrite('new.jpg', after *5)]
    c, d = after.shape
    return c, d, after


# define the draw rectangular function: notice in our current case its box not rec
def drawrec(filename, coor, x, y, color):
    im = np.array(Image.open(filename), dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    rect = patches.Rectangle(coor, x, y, linewidth=1, edgecolor=color, facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


a=321
#drawrec('new.png', (least_cor[1],least_cor[0]), a, a, 'b')

#another window size decrease by power of 2
win1_size=int(np.floor(a/8))
c1, d1, arr1 = whitespaceplus(bimg, win1_size)
result1 = np.where(arr1 == np.min(arr1))
coor1 = list(zip(result1[0], result1[1]))  ##minor change in this step change x and y !!!
print(coor1)
print(c1)
print(d1)

#find the minimum b in the coordinate
mini1=1000
least_cor1=(0,0)
for cor1 in coor1:

    addvalue1=arr1[cor1]
    if addvalue1<mini1:

        mini1=addvalue1
        least_cor1=cor1
print(least_cor1)


cimg = np.ones((x, y), np.int)
#binary image either 0 for white or 1 for black  --- used for calculate centre of mass
#loop through all the coordinates and if find one then make all its window content to be 0, else if 1
for corr in coor1:
    xi=corr[0]
    yi=corr[1]
    cimg[xi:xi+win1_size,yi:yi+win1_size]=0

def mass(biarr):
    xa,xb=biarr.shape




