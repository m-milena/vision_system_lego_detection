import cv2
import copy
import numpy as np
import math

def mask_white(img_hsv, img_clahe, img_color_resized):
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([179, 40, 225])
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
    res_white = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_white)
    res_ = copy.deepcopy(res_white)
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_white, cv2.COLOR_BGR2GRAY), 1, 2)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > 4 and rect[1][0] <10 and rect[1][1] > 4 and rect[1][1] < 10:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color_resized,[box],0,(80,205,80),thickness=cv2.FILLED)
            cv2.drawContours(res_white,[box],0,(0,0,0),thickness=cv2.FILLED)
    return res_white

def mask_white2(img_hsv, img_color_resized):
    lower_white = np.array([0, 0, 70])
    upper_white = np.array([80, 45, 255])
    #lower_white = np.array([0, 50, 0])
    #upper_white = np.array([50, 120, 155])
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
    res_white = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_white)
    return res_white

def detect_blocks(res_white, img_color_resized):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_white, cv2.COLOR_BGR2GRAY), 1, 2)
    white_block_cnt = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        #if (rect[1][0] > 15 and rect[1][1] > 15) and (rect[1][0] < 150 and rect[1][1] < 150):
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        d1 = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
        d2 = math.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
        cv2.drawContours(img_color_resized,[box],0,(200,200,0),2)
        white_block_cnt += 1


def white_holes(res_white, img_color_resized):
    #res_white = cv2.threshold(res_white, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_white, cv2.COLOR_BGR2GRAY), 1, 2)
    white_holes_cnt = 0
    to_detect_blocks = np.ones((res_white.shape[0], res_white.shape[1], 3), np.uint8)*255
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        rect2 = (rect[0], rect[1], 0)
        if rect[1][0] > 15 and rect[1][0] < 30  and rect[1][1] > 15 and rect[1][1] < 30 and math.fabs(rect[1][1]-rect[1][0])<8:
            box = cv2.boxPoints(rect2)
            box = np.int0(box)
            #M = cv2.moments(cnt)
            cX = int((box[2][0] + box[1][0])/2)  #int(M["m10"] / M["m00"])
            cY = int((box[0][1] + box[1][1])/2)#int(M["m01"] / M["m00"])
            cv2.circle(img_color_resized, (cX, cY), 2, (0,0,255))
            if res_white[cY, cX, 0] == 0:
                cv2.drawContours(img_color_resized,[box],0,(100,200,10),2)
                cv2.drawContours(to_detect_blocks,[box],0,(0,0,0),cv2.FILLED)
                cv2.drawContours(to_detect_blocks,[box],0,(0,0,0),10)
                white_holes_cnt += 1
    kernel = np.ones((3,3), np.uint8)
    to_detect_blocks = cv2.erode(to_detect_blocks, kernel, iterations=4)
    return white_holes_cnt, to_detect_blocks

def detect_white_blocks(block_mask, img_color):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(block_mask, cv2.COLOR_BGR2GRAY), 1, 2)
    white_blocks_counter = 0
    contours = contours[:-1]
    hierarchy = hierarchy[:-1]
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        min_x = min(rect[1][0], rect[1][1])
        max_x = max(rect[1][0], rect[1][1])
        if min_x > 30 and min_x <= 60:
            white_blocks_counter += 1
            cv2.drawContours(img_color,[box],0,(0,0,255),2)
        elif min_x > 60 and min_x < 100:
            white_blocks_counter += 2
            cv2.drawContours(img_color,[box],0,(0,0,255),2)
        elif min_x >=100 and min_x < 130:
            white_blocks_counter += 3
            cv2.drawContours(img_color,[box],0,(0,0,255),2)
    return white_blocks_counter