import time
import numpy as np
import cv2 as cv
import os
from scipy import signal

curdir = os.path.dirname(os.path.realpath(__file__))
stru = '\yy'
curdir = curdir.replace(stru[0], '/') + '/'
im_name = 'Capture1.PNG'
vid_name = '2.mp4'

blur_arr1 = np.array([[0.003, 0.013, 0.022, 0.013, 0.003],
                      [0.013, 0.05, 0.078, 0.05, 0.098],
                      [0.022, 0.078, 0.114, 0.078, 0.022],
                      [0.013, 0.05, 0.078, 0.05, 0.098],
                      [0.003, 0.013, 0.022, 0.013, 0.003]
                      ])

blur_arr3 = np.array([
    [0.003, 0.013, 0.022, 0.013, 0.003],
    [0.013, 0.06, 0.098, 0.06, 0.013],
    [0.022, 0.098, 0.162, 0.098, 0.022],
    [0.013, 0.06, 0.098, 0.06, 0.013],
    [0.003, 0.013, 0.022, 0.013, 0.003]
])
num3x3 = 1/9
blur_arr2 = np.array([[num3x3, num3x3, num3x3],
                      [num3x3, num3x3, num3x3],
                      [num3x3, num3x3, num3x3]])
blur_arr5 = np.ones((5, 5))/25

blur_arr4 = np.array([[1/16, 1/8, 1/16],
                      [1/8, 1/4, 1/8],
                      [1/16, 1/8, 1/16]])

conv_arr2 = np.array([[-0.25, -0.5, -0.25],
                      [0, 0, 0],
                      [0.25, 0.5, 0.25]])

conv_arr1 = np.array([[0.125, 0.125, 0.3, 0.125, 0.125],
                     [0, 0, 0, 0, 0],
                     [-0.125, -0.125, -0.3, -0.125, -0.125]])


def grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def conv_img_mat(img, conv_mat):
    result = []
    step_size = int(len(conv_mat)/2)
    for i in range(len(img)):
        row = []
        for j in range(len(img[0])):
            vec = np.zeros(3)
            for i_2 in range(len(conv_mat)):
                for j_2 in range(len(conv_mat)):
                    try:
                        vec = vec + conv_mat[i_2][j_2] * img[i -
                                                             step_size + i_2][j - step_size + j_2]
                    except:
                        pass
            vec = list(map(lambda x: int(x), vec))
            row.append(vec)
        result.append(row)
    return np.array(result).astype(np.uint8)


def edged_img(img, edge_arr, blur_arr, with_grayscale=True, with_blend=True, with_edge_detec=True):
    result = img
    if (len(img) < 10):
        print('some kind of image error')
        return np.array([[1]])
    if (with_grayscale):
        img = grayscale(img)
    if (with_blend):
        img = cv.filter2D(img, -1, blur_arr)
        img = np.array(img).astype(np.uint8)
    if (with_edge_detec):
        on_y = cv.filter2D(img, 0, edge_arr)
        on_x = cv.filter2D(img, 0, edge_arr.T)
        result = np.sqrt(np.square(on_y) + np.square(on_x))
    result = result*4
    # cv.imshow('h' , result.astype(np.uint8).T)
    # result *= 255.0 / result.max()
    return np.array(result).astype(np.uint8)


def frame_diff(prev, cur, next):
    diff1 = cv.absdiff(prev, cur)
    diff2 = cv.absdiff(cur, next)
    return cv.bitwise_and(diff1, diff2)


path = curdir + vid_name
video = cv.VideoCapture(path)
frame_rate = video.get(cv.CAP_PROP_FPS)
ret, prevframe = video.read()

prevframe = edged_img(prevframe, conv_arr2, blur_arr4, True, True)
ret, curframe = video.read()
curframe = edged_img(curframe, conv_arr2, blur_arr4, True, True)
ret, nextframe = video.read()
nextframe = edged_img(nextframe, conv_arr2, blur_arr4, True, True)

prevframe = cv.resize(prevframe, None, None, 0.5, 0.5,
                      interpolation=cv.INTER_AREA)
curframe = cv.resize(curframe, None, None, 0.5, 0.5,
                     interpolation=cv.INTER_AREA)
nextframe = cv.resize(nextframe, None, None, 0.5, 0.5,
                      interpolation=cv.INTER_AREA)


# GLOBAL PARAMS.
from_up = False
meteres = 500
frames_car_was_in = 0
see_car = False
car_out = False


def takeavrg(mat):
    return np.sum(mat)/(len(mat) * len(mat[0]))


def detect_car(mat):
    global see_car, from_up, frames_car_was_in, car_out
    if (not see_car):
        for i in range(int(len(mat)/16)):
           i = i * 4
           for j in range(int(len(mat[0])/4)):
               j = j * 4
               sub_mat = mat[i:i+15, j:j+15]
               if (np.sum(sub_mat)/225 > 4):
                   from_up = True
                   see_car = True
                   return
        for i in range(int(len(mat)*(5/28)), int(len(mat)/4)):
           i = i * 4
           for j in range(int(len(mat[0])/4)):
               j = j * 4
               sub_mat = mat[i:i+15, j:j+15]
               if (np.sum(sub_mat)/225 > 4):
                   from_up = False
                   see_car = True
                   return


def redetect_car(mat):
    global see_car, from_up, frames_car_was_in, car_out
    if (not from_up):
        for i in range(int(len(mat)/25)):
           i = i * 4
           for j in range(int(len(mat[0])/4)):
               j = j * 4
               sub_mat = mat[i:i+10, j:j+10]
               if (np.sum(sub_mat)/100 > 3):
                   car_out = True
                   see_car = False
                   return
    else:
        for i in range(int(len(mat)*(9/40)), int(len(mat)/4)):
           i = i * 4
           for j in range(int(len(mat[0])/4)):
               j = j * 4
               sub_mat = mat[i:i+10, j:j+10]
               if (np.sum(sub_mat)/100 > 3):
                   car_out = True
                   see_car = False
                   return

# def detect_car(mat):
#     global see_car, from_up,  car_out
#     sub_mat1 =


# def redetect_car(mat):
#     global see_car, from_up, car_out
def calculatespeed():
 if (frames_car_was_in != 0):
    time_ = (frames_car_was_in/frame_rate)
    speed = meteres/time_
    speed = (speed/1000) * 360
    print("speed - " + f'{speed}' + ' km/hr')
 else:
    print('there was no car')


while video.isOpened():
    framediff = np.array(frame_diff(
        prevframe, curframe, nextframe)).astype(np.uint8)
    # _,frame_th = cv.threshold(framediff,0,255,cv.THRESH_TRIANGLE)
    prevframe = curframe
    curframe = nextframe
    ret, nextframe = video.read()
    if not ret:
        break
    nextframe = edged_img(nextframe, conv_arr2, blur_arr4, True, True)
    nextframe = cv.resize(nextframe, None, None, 0.5,
                          0.5, interpolation=cv.INTER_AREA)
    # frame_th = cv.resize(frame_th,None,None,0.5,0.8,interpolation=cv.INTER_AREA)
    # framediff *= 255.0 / framediff.max()
    # framediff = framediff.astype(np.uint8)

    # cv.imshow('2',frame_th)
    # print('on right ' + str(from_up))

    if (see_car):
        redetect_car(framediff.T)
        frames_car_was_in = frames_car_was_in + 1
    if (not see_car):
        detect_car(framediff.T)

    # this breaks after the first car leaves the area,
    if (car_out):
        calculatespeed()
        frames_car_was_in = 0
        break

    cv.imshow('press q tu quit', framediff)
    if (cv.waitKey(20) & 0xFF == ord('q')):
        break


# newd = edged_img(tmp, conv_arr2, blur_arr4, True, True)
# newd = np.array(newd)
# newd = newd.astype(np.uint8)
# cv.imshow('edged', newd)
cv.waitKey(0)
