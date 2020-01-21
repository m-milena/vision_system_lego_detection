import cv2
import numpy as np
import math

def blue_blocks(img_hsv, img_color_resized):
    # Detect blue blocks
    #lower_blue = np.array([105,20,50])
    #upper_blue = np.array([115,255,255])
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([135,255,255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2HSV)
    img_color_resized[:,:, 1] = 0 
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_HSV2BGR)
    res_blue = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_blue)
    res_blue = cv2.medianBlur(res_blue, 1)
    kernel = np.ones((1,1), np.uint8) 
    res_blue = cv2.erode(res_blue, kernel, iterations=1)
    #res_blue = cv2.medianBlur(res_blue, 1)
    #res_blue = cv2.dilate(res_blue, kernel, iterations=1)
    return res_blue

def blue_holes(res_blue, img_color_resized):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY), 1, 2)
    blue_block_cnt = 0
    blue_holes_cnt = 0
    blue_holes_cnt_1 = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > 15 or rect[1][1] > 15:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            d1 = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
            d2 = math.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
            if d1 > d2: # 7 elementow
                if d1 > 90 and d1 < 100:
                    blue_holes_cnt += 7
                elif d1 > 65 and d1 < 75:
                    blue_holes_cnt += 5 
                elif d1 > 15 and d1 < 30:
                    blue_holes_cnt += 1
            else:
                if d2 > 90 and d2 < 100:
                    blue_holes_cnt += 7
                elif d2 > 65 and d2 < 75:
                    blue_holes_cnt += 5 
                elif d2 > 15 and d2 < 30:
                    blue_holes_cnt += 1
            cv2.drawContours(img_color_resized,[box],0,(200,10,0),2)
            blue_block_cnt += 1

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
                blue_holes_cnt_1 += 1 
    print(blue_holes_cnt)
    print(blue_holes_cnt_1)