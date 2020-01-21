import cv2
import math
import numpy as np

def yellow_blocks(img_hsv, img_color_resized):
    # Detect yellow blocks
    lower_yellow = np.array([20, 60, 70])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2HSV)
    img_color_resized[:,:, 1] = 0 
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_HSV2BGR)
    res_yellow = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_yellow)
    kernel = np.ones((2,2), np.uint8) 
    res_yellow =  cv2.medianBlur(res_yellow, 3)
    res_yellow = cv2.dilate(res_yellow, kernel, iterations=1)
    res_yellow = cv2.erode(res_yellow, kernel, iterations=2)
    res_yellow =  cv2.medianBlur(res_yellow, 1)
    return res_yellow

def yellow_holes(res_yellow, img_color_resized):
    # Detect yellow - work
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_yellow, cv2.COLOR_BGR2GRAY), 1, 2)
    yellow_block_cnt = 0
    yellow_holes_cnt = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > 15 or rect[1][1] > 15:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color_resized,[box],0,(0,255,0),2)
            yellow_block_cnt += 1
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if (rect[1][0] > 5 and rect[1][0] < 10) or (rect[1][1] > 5 and rect[1][1] < 10):
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            d1 = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
            d2 = math.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
            d3 = math.sqrt((box[2][0] - box[3][0])**2 + (box[2][1] - box[3][1])**2)
            d4 = math.sqrt((box[3][0] - box[0][0])**2 + (box[3][1] - box[0][1])**2)
            if d1 > 2 and d2 > 2 and d3 > 2 and d4 > 2:
                cv2.drawContours(img_color_resized,[box],0,(0,0,0),2)
                yellow_holes_cnt += 1 
    print(yellow_holes_cnt)