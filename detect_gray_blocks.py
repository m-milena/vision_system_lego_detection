import cv2
import math
import numpy as np
import copy

def gray_blocks(img_hsv, img_color_resized):
    lower_gray = np.array([35, 15, 5])
    upper_gray = np.array([140, 180, 180])
    #lower_gray = np.array([40, 15, 10])
    # upper_gray = np.array([100, 180, 180])
    mask_gray = cv2.inRange(img_hsv, lower_gray, upper_gray)
    #img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2HSV)
    #img_color_resized[:,:, 1] = 0 
    #img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_HSV2BGR)
    res_gray = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_gray)
    kernel = np.ones((2,2), np.uint8) 
    res_gray = cv2.medianBlur(res_gray, 3)
    res_gray = cv2.dilate(res_gray, kernel, iterations=3)
    #res_gray= cv2.erode(res_gray, kernel, iterations=1)
    return res_gray

def gray_holes(res_gray, img_color_resized):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_gray, cv2.COLOR_BGR2GRAY), 1, 2)
    gray_block_cnt = 0
    gray_holes_cnt = 0
    gray_holes_cnt_1 = 0
    counter = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > 20 or rect[1][1] > 20:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            d1 = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
            d2 = math.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
            print(d1)
            print(d2)
            print('##############################')
            if d1 > d2: # 7 elementow
                if d1 > 90 and d1 < 100:
                    gray_holes_cnt += 7
                elif d1 > 65 and d1 < 75:
                    gray_holes_cnt += 5 
                elif d1 > 15 and d1 < 30:
                    gray_holes_cnt += 1
            else:
                if d2 > 90 and d2 < 100:
                    gray_holes_cnt += 7
                elif d2 > 65 and d2 < 75:
                    gray_holes_cnt += 5 
                elif d2 > 15 and d2 < 30:
                    gray_holes_cnt += 1
            if d1 > 10 and d2 > 10:
                cv2.drawContours(img_color_resized,[box],0,(200,10,100),2)
                gray_block_cnt += 1
                counter +=1
            #if counter == 1:
            #    break
            

    print(gray_holes_cnt)