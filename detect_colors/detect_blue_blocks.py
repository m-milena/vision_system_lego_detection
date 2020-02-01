import cv2
import numpy as np
import math

def mask_blue(img_hsv, img_color_resized):
    # Detect blue blocks
    #lower_blue = np.array([105,20,50])
    #upper_blue = np.array([115,255,255])
    lower_blue = np.array([105,20,10])
    upper_blue = np.array([135,255,255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2HSV)
    img_color_resized[:,:, 1] = 0 
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_HSV2BGR)
    res_blue = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_blue)
    #res_blue = cv2.medianBlur(res_blue, 1)
    #kernel = np.ones((1,1), np.uint8) 
    #res_blue = cv2.erode(res_blue, kernel, iterations=1)
    #res_blue = cv2.medianBlur(res_blue, 1)
    #res_blue = cv2.dilate(res_blue, kernel, iterations=1)
    return res_blue



def detect_blocks(res_blue, img_color_resized):
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
            cv2.drawContours(img_color_resized,[box],0,(200,10,0),2)
            blue_block_cnt += 1


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

def blue_holes2(res_blue, img_color_resized):
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

    circles = cv2.HoughCircles(cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY),cv2.HOUGH_GRADIENT,1,2,param1=100,param2=7,minRadius=2,maxRadius=7)
    circles = np.uint16(np.around(circles))
    circles_count = 0
    for i in circles[0,:]:
        if res_blue[i[1], i[0], 0] == 0:
            #cv2.circle(img_color_resized,(i[0],i[1]),i[2],(40,27,255),2)
            circles_count += 1

def mask_blue_to_holes(img_hsv, img_color_resized):
    lower_blue = np.array([105,20,10])
    upper_blue = np.array([135,255,255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2HSV)
    img_color_resized[:,:, 1] = 0 
    img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_HSV2BGR)
    res_blue = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_blue)
    #res_blue = cv2.medianBlur(res_blue, 1)
    kernel = np.ones((2,2), np.uint8) 
    #res_blue = cv2.erode(res_blue, kernel, iterations=3)
    #_, thresh = cv2.threshold(res_blue, 70, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((2,2), np.uint8) 
    #res_blue = cv2.erode(thresh, kernel, iterations=2)
    #res_blue = cv2.medianBlur(res_blue, 5)
    #kernel = np.ones((3,3), np.uint8) 
    #res_blue = cv2.dilate(res_blue, kernel, iterations=5)
    #kernel = np.ones((2,2), np.uint8) 
    #res_blue = cv2.erode(thresh, kernel, iterations=1)
    return res_blue

blue_size = {
    (50, 100): 1,
    (100, 180): 3,
    (180, 245): 5,
    (245, 320): 7,
    (320, 400): 9
}

def detect_holes(res_blue, img, mask_white):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_blue,cv2.COLOR_BGR2GRAY), 1, 1)
    blue_holes_cnt = 0
    detected_block = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > 35 and rect[1][1] > 35 and (rect[1][1]>50 or rect[1][0] > 50):
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,0),2)
            min_x = min(rect[1][0], rect[1][1])
            max_x = max(rect[1][0], rect[1][1])
            if min_x <= 75:
                if max_x > 400:
                    max_x = max_x / 2
                    for key in blue_size.keys():
                        if int(max_x) in range(key[0], key[1]):
                            blue_holes_cnt += 2*blue_size.get(key) + 2
                            detected_block += 2
                else:
                    for key in blue_size.keys():
                        if int(max_x) in range(key[0], key[1]):
                            blue_holes_cnt += blue_size.get(key)
                            detected_block += 1
                cv2.drawContours(mask_white,[box],0,(255,255,255),thickness=cv2.FILLED)
            elif min_x > 75 and min_x < 110:
                if min_x == rect[1][0]:
                    rect = ((rect[0][0], rect[0][1]), (10, rect[1][1]), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=cv2.FILLED)
                elif min_x == rect[1][1]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0], 10), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=cv2.FILLED)
                kernel = np.ones((3,3), np.uint8)
                res_blue =  cv2.medianBlur(res_blue, 5)
                res_blue = cv2.erode(res_blue, kernel, iterations=2)
                contours, hierarchy = cv2.findContours(cv2.cvtColor(res_blue,cv2.COLOR_BGR2GRAY), 1, 1)
                detected_block = 0
                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    if rect[1][0]>20 and rect[1][1]>20 and (rect[1][1]>55 or rect[1][0] > 55):
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(img,[box],0,(0,255,255),2)
                        cv2.drawContours(mask_white,[box],0,(255,255,255),thickness=cv2.FILLED)
                        min_x = min(rect[1][0], rect[1][1])
                        max_x = max(rect[1][0], rect[1][1])
                        if min_x < 70:
                            if max_x > 400:
                                max_x = max_x / 2
                                for key in blue_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        blue_holes_cnt += 2*blue_size.get(key) + 2
                                        detected_block += 2
                            else:
                                for key in blue_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        blue_holes_cnt += blue_size.get(key)
                                        detected_block += 1
                break
            elif min_x > 110 and min_x < 160:
                if min_x == rect[1][0]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0]/3, rect[1][1]), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=5)
                elif min_x == rect[1][1]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/3), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=5)
                kernel = np.ones((3,3), np.uint8)
                res_blue =  cv2.medianBlur(res_blue, 5)
                res_blue = cv2.erode(res_blue, kernel, iterations=1)
                contours, hierarchy = cv2.findContours(cv2.cvtColor(res_blue,cv2.COLOR_BGR2GRAY), 1, 1)
                detected_block = 0
                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    if rect[1][0]>35 and rect[1][1]>35:# and (rect[1][1]>55 or rect[1][0] > 55):
                        detected_block = 1
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(img,[box],0,(0,255,255),2)
                        cv2.drawContours(mask_white,[box],0,(255,255,255),thickness=cv2.FILLED)
                        min_x = min(rect[1][0], rect[1][1])
                        max_x = max(rect[1][0], rect[1][1])
                        if min_x < 70:
                            if max_x > 400:
                                max_x = max_x / 2
                                for key in blue_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        blue_holes_cnt += 2*blue_size.get(key) + 2
                                        detected_block += 2
                            else:
                                for key in blue_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        blue_holes_cnt += blue_size.get(key)
                                        detected_block += 1
                break

    return blue_holes_cnt, detected_block
