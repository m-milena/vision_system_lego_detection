import cv2
import numpy as np
import math

def mask_red(img_hsv, img_color_resized):
    # Detect red blocks
    lower_red_first = np.array([0,50,30])
    upper_red_first = np.array([10,255,255])
    mask_red_first = cv2.inRange(img_hsv, lower_red_first, upper_red_first)
    lower_red_second = np.array([170,50,30])
    upper_red_second = np.array([179,255,255])
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

def detect_blocks(res_red, img_color_resized):
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
            cv2.drawContours(img_color_resized,[box],0,(0,25,50),2)
            red_block_cnt += 1

def mask_red_to_holes(img_hsv, img_color_resized):
    # Detect red blocks
    lower_red_first = np.array([0,50,0])
    upper_red_first = np.array([12,255,255])
    mask_red_first = cv2.inRange(img_hsv, lower_red_first, upper_red_first)
    lower_red_second = np.array([170,50,0])
    upper_red_second = np.array([179,255,255])
    mask_red_second = cv2.inRange(img_hsv, lower_red_second, upper_red_second)
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2HSV)
    img_color_resized[:,:, 1] = 0 
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_HSV2BGR)
    res_red_first = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_red_first)
    res_red_second = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_red_second)
    res_red = cv2.add(res_red_first, res_red_second)
    return res_red

red_size = {
    (50, 100): 1,
    (100, 180): 3,
    (180, 245): 5,
    (245, 320): 7,
    (320, 400): 9
}

def detect_holes(res_red, img, mask_white):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_red,cv2.COLOR_BGR2GRAY), 1, 1)
    red_holes_cnt = 0
    detected_block = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0]>35 and rect[1][1]>35 and (rect[1][1]>50 or rect[1][0] > 50):
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,255),2)
            min_x = min(rect[1][0], rect[1][1])
            max_x = max(rect[1][0], rect[1][1])
            if min_x <= 75:
                if max_x > 400:
                    max_x = max_x / 2
                    for key in red_size.keys():
                        if int(max_x) in range(key[0], key[1]):
                            red_holes_cnt += 2*red_size.get(key) + 2
                            detected_block += 2
                else:
                    for key in red_size.keys():
                        if int(max_x) in range(key[0], key[1]):
                            red_holes_cnt += red_size.get(key)
                            detected_block += 1
                cv2.drawContours(mask_white,[box],0,(255,255,255),thickness=cv2.FILLED)
            elif min_x > 75 and min_x < 110:
                if min_x == rect[1][0]:
                    rect = ((rect[0][0], rect[0][1]), (10, rect[1][1]), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_red,[box],0,(0,0,0),thickness=cv2.FILLED)
                elif min_x == rect[1][1]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0], 10), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_red,[box],0,(0,0,0),thickness=cv2.FILLED)
                kernel = np.ones((3,3), np.uint8)
                res_red =  cv2.medianBlur(res_red, 5)
                #res_red = cv2.dilate(res_red, kernel, iterations=1)
                res_red = cv2.erode(res_red, kernel, iterations=1)
                contours, hierarchy = cv2.findContours(cv2.cvtColor(res_red,cv2.COLOR_BGR2GRAY), 1, 1)
                detected_block = 0
                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    if rect[1][0]>30 and rect[1][1]>30 and (rect[1][1]>55 or rect[1][0] > 55):
                        detected_block = 1
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(mask_white,[box],0,(255,255,255),thickness=cv2.FILLED)
                        cv2.drawContours(img,[box],0,(0,255,255),2)
                        min_x = min(rect[1][0], rect[1][1])
                        max_x = max(rect[1][0], rect[1][1])
                        if min_x < 70:
                            if max_x > 400:
                                max_x = max_x / 2
                                for key in red_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        red_holes_cnt += 2*red_size.get(key) + 2
                                        detected_block += 2
                            else:
                                for key in red_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        red_holes_cnt += red_size.get(key)
                                        detected_block += 1
                break
            elif min_x > 110 and min_x < 160:
                if min_x == rect[1][0]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0]/3, rect[1][1]), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_red,[box],0,(0,0,0),thickness=5)
                elif min_x == rect[1][1]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/3), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_red,[box],0,(0,0,0),thickness=5)
                kernel = np.ones((3,3), np.uint8)
                res_red =  cv2.medianBlur(res_red, 5)
                #res_red = cv2.dilate(res_red, kernel, iterations=1)
                res_red = cv2.erode(res_red, kernel, iterations=1)
                contours, hierarchy = cv2.findContours(cv2.cvtColor(res_red,cv2.COLOR_BGR2GRAY), 1, 1)
                detected_block = 0
                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    if rect[1][0]>30 and rect[1][1]>30 and (rect[1][1]>55 or rect[1][0] > 55):
                        detected_block = 1
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(mask_white,[box],0,(255,255,255),thickness=cv2.FILLED)
                        cv2.drawContours(img,[box],0,(0,255,255),2)
                        min_x = min(rect[1][0], rect[1][1])
                        max_x = max(rect[1][0], rect[1][1])
                        if min_x < 70:
                            if max_x > 400:
                                max_x = max_x / 2
                                for key in red_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        red_holes_cnt += 2*red_size.get(key) + 2
                                        detected_block += 2
                            else:
                                for key in red_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        red_holes_cnt += red_size.get(key)
                                        detected_block += 1
                    break
            
    return red_holes_cnt, detected_block