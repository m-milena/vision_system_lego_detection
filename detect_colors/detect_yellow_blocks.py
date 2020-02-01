import cv2
import math
import numpy as np

def yellow_blocks(img_hsv, img_color_resized):
    # Detect yellow blocks
    lower_yellow = np.array([20, 65, 70])
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
               # cv2.drawContours(img_color_resized,[box],0,(0,0,0),2)
                yellow_holes_cnt += 1 

def mask_yellow_to_holes(img_hsv, img_color_resized):
    # Detect yellow blocks
    #lower_yellow = np.array([20, 65, 70])
    #upper_yellow = np.array([30, 255, 255])
    lower_yellow = np.array([20, 80, 125])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2HSV)
    img_color_resized[:,:, 1] = 0 
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_HSV2BGR)
    res_yellow = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_yellow)
    _, res_yellow = cv2.threshold(res_yellow, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2,2), np.uint8) 
    res_yellow =  cv2.medianBlur(res_yellow, 3)
    #res_yellow = cv2.dilate(res_yellow, kernel, iterations=1)
    #res_yellow = cv2.erode(res_yellow, kernel, iterations=2)
    #res_yellow =  cv2.medianBlur(res_yellow, 1)
    return res_yellow

yellow_size = {
    (50, 90): 1,
    (90, 190): 3,
    (190, 255): 5,
    (255, 320): 7,
    (320, 400): 9
}

def detect_holes(res_yellow, img, mask_white):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_yellow,cv2.COLOR_BGR2GRAY), 1, 1)
    yellow_holes_cnt = 0
    detected_block = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        rect2 = (rect[0], (rect[1][0]/1.3, rect[1][1]/1.3), rect[2])
        if rect[1][0]>30 and rect[1][1]>30 and (rect[1][1]>50 or rect[1][0] > 50):
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box2 = cv2.boxPoints(rect2)
            box2 = np.int0(box2)
            cv2.drawContours(img,[box],0,(0,255,255),2)
            min_x = min(rect[1][0], rect[1][1])
            max_x = max(rect[1][0], rect[1][1])
            if min_x <= 75:
                if max_x > 400:
                    max_x = max_x / 2
                    for key in yellow_size.keys():
                        if int(max_x) in range(key[0], key[1]):
                            yellow_holes_cnt += 2*yellow_size.get(key) + 2
                            detected_block += 2
                else:
                    for key in yellow_size.keys():
                        if int(max_x) in range(key[0], key[1]):
                            yellow_holes_cnt += yellow_size.get(key)
                            detected_block += 1
                cv2.drawContours(mask_white,[box2],0,(255,255,255),thickness=cv2.FILLED)
            elif min_x > 75 and min_x < 115:
                if min_x == rect[1][0]:
                    rect = ((rect[0][0], rect[0][1]), (15, rect[1][1]), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=cv2.FILLED)
                elif min_x == rect[1][1]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0], 15), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=cv2.FILLED)
                kernel = np.ones((3,3), np.uint8)
                res_yellow =  cv2.medianBlur(res_yellow, 5)
                res_yellow = cv2.erode(res_yellow, kernel, iterations=3)
                res_yellow = cv2.dilate(res_yellow, kernel, iterations=4)
                contours, hierarchy = cv2.findContours(cv2.cvtColor(res_yellow,cv2.COLOR_BGR2GRAY), 1, 1)
                detected_block = 0
                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    if rect[1][0]>30 and rect[1][1]>30 and (rect[1][1]>55 or rect[1][0] > 55):
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(img,[box],0,(0,155,255),2)
                        min_x = min(rect[1][0], rect[1][1])
                        max_x = max(rect[1][0], rect[1][1])
                        if min_x < 70:
                            if max_x > 400:
                                max_x = max_x / 2
                                for key in yellow_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        yellow_holes_cnt += 2*yellow_size.get(key) + 2
                                        detected_block += 2
                            else:
                                for key in yellow_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        yellow_holes_cnt += yellow_size.get(key)
                                        detected_block += 1
                            cv2.drawContours(mask_white,[box],0,(255,255,255),thickness=cv2.FILLED)
                break
            elif min_x > 115 and min_x < 160:
                if min_x == rect[1][0]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0]/3, rect[1][1]), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=5)
                elif min_x == rect[1][1]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/3), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=5)
                kernel = np.ones((3,3), np.uint8)
                res_yellow =  cv2.medianBlur(res_yellow, 5)
                res_yellow = cv2.erode(res_yellow, kernel, iterations=1)
                contours, hierarchy = cv2.findContours(cv2.cvtColor(res_yellow,cv2.COLOR_BGR2GRAY), 1, 1)
                detected_block = 0
                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    if rect[1][0]>30 and rect[1][1]>30 and (rect[1][1]>55 or rect[1][0] > 55):
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(mask_white,[box],0,(255,255,255),thickness=cv2.FILLED)
                        cv2.drawContours(img,[box],0,(0,155,255),2)
                        min_x = min(rect[1][0], rect[1][1])
                        max_x = max(rect[1][0], rect[1][1])
                        if min_x < 70:
                            if max_x > 400:
                                max_x = max_x / 2
                                for key in yellow_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        yellow_holes_cnt += 2*yellow_size.get(key) + 2
                                        detected_block += 2
                            else:
                                for key in yellow_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        yellow_holes_cnt += yellow_size.get(key)
                                        detected_block += 1
                break
            
    return yellow_holes_cnt, detected_block