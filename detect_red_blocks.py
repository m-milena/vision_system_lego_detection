import cv2
import numpy as np
import math

def red_blocks(img_hsv, img_color_resized):
    # Detect red blocks
    lower_red_first = np.array([0,50,100])
    upper_red_first = np.array([7,240,200])
    mask_red_first = cv2.inRange(img_hsv, lower_red_first, upper_red_first)
    lower_red_second = np.array([170,50,100])
    upper_red_second = np.array([179,240,200])
    mask_red_second = cv2.inRange(img_hsv, lower_red_second, upper_red_second)
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2HSV)
    img_color_resized[:,:, 1] = 0 
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_HSV2BGR)
    res_red_first = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_red_first)
    res_red_second = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_red_second)
    res_red = cv2.add(res_red_first, res_red_second)
    kernel = np.ones((1,1), np.uint8)
    res_red =  cv2.medianBlur(res_red, 1)
    res_red = cv2.dilate(res_red, kernel, iterations=1)
    #res_red = cv2.erode(res_red, kernel, iterations=1)
    #res_red =  cv2.medianBlur(res_red, 1)
    return res_red

def red_holes(res_red, img_color_resized):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY), 1, 2)
    red_block_cnt = 0
    red_holes_cnt = 0
    red_holes_cnt_1 = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > 15 or rect[1][1] > 15:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            d1 = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
            d2 = math.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
            if d1 > d2: # 7 elementow
                if d1 > 90 and d1 < 100:
                    red_holes_cnt += 7
                elif d1 > 65 and d1 < 75:
                    red_holes_cnt += 5 
                elif d1 > 15 and d1 < 30:
                    red_holes_cnt += 1
            else:
                if d2 > 90 and d2 < 100:
                    red_holes_cnt += 7
                elif d2 > 65 and d2 < 75:
                    red_holes_cnt += 5 
                elif d2 > 15 and d2 < 30:
                    red_holes_cnt += 1
            cv2.drawContours(img_color_resized,[box],0,(0,25,50),2)
            red_block_cnt += 1

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
                red_holes_cnt_1 += 1 
    print(red_holes_cnt)
    print(red_holes_cnt_1)