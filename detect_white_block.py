import cv2
import copy
import numpy as np
import math

def white_blocks(img_hsv, img_color_resized):
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([179, 40, 225])
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
    res_white = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_white)
    #res_white = cv2.medianBlur(res_white, 3)
    #kernel = np.ones((3,3), np.uint8) 
    #res_white = cv2.dilate(res_white, kernel, iterations=1)
    #res_white = cv2.erode(res_white, kernel, iterations=1)
    return res_white

def white_holes(res_white, img_color_resized):
    res_ = copy.deepcopy(res_white)
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_white, cv2.COLOR_BGR2GRAY), 1, 2)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > 5 and rect[1][0] <10 and rect[1][1] > 5 and rect[1][1] < 10:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color_resized,[box],0,(100,200,10),2)
            cv2.drawContours(res_,[box],0,(0,0,0),2)
    return res_
