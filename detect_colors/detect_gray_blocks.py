import cv2
import math
import numpy as np
import copy

def mask_gray(img_hsv, img_color_resized):
    lower_gray = np.array([40, 10, 0])
    upper_gray = np.array([100, 255, 205])
    #lower_gray = np.array([0, 0, 100])
    #upper_gray = np.array([50, 85, 255])
    mask_gray = cv2.inRange(img_hsv, lower_gray, upper_gray)
    #img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2HSV)
    #img_color_resized[:,:, 1] = 0 
    #img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_HSV2BGR)
    res_gray = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_gray)
    kernel = np.ones((3,3), np.uint8) 
    res_gray = cv2.medianBlur(res_gray, 3)
    res_gray = cv2.dilate(res_gray, kernel, iterations=3)
    res_gray= cv2.erode(res_gray, kernel, iterations=2)
    res_gray = cv2.medianBlur(res_gray, 5)
    return res_gray

def detect_blocks(res_gray, img_color_resized):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_gray, cv2.COLOR_BGR2GRAY), 1, 2)
    gray_block_cnt = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > 15 and rect[1][1] > 15:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color_resized,[box],0,(200,10,100),2)
            gray_block_cnt += 1
    return contours

gray_size = {
    (20, 41): 1,
    (41, 63): 3,
    (63, 83): 5,
    (83, 105): 7,
    (105, 135): 9
}
def detect_blocks2(res_gray, img_color_resized, mask_white):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res_gray, cv2.COLOR_BGR2GRAY), 1, 2)
    gray_block_cnt = 0
    gray_holes_cnt = 0
    detected_block = 0
    to_mask_white = np.ones([res_gray.shape[0], res_gray.shape[1],3],dtype=np.uint8)*255
    img = copy.deepcopy(img_color_resized)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > rect[1][1]:
            rect2 = ((rect[0][0]-5, rect[0][1]-3), (rect[1][0], rect[1][1]-4), rect[2])
        else: 
            rect2 = ((rect[0][0]-3, rect[0][1]-5), (rect[1][0]-4, rect[1][1]), rect[2])
        if rect[1][0] > 13 and rect[1][1] > 13:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box2 = cv2.boxPoints(rect2)
            box2 = np.int0(box2)
            min_x = min(rect[1][0], rect[1][1])
            max_x = max(rect[1][0], rect[1][1])
            if min_x < 30:
                if max_x > 22:
                    for key in gray_size.keys():
                        if int(max_x) in range(key[0], key[1]):
                            gray_holes_cnt += gray_size.get(key)
                            detected_block += 1
                    cv2.drawContours(img_color_resized,[box],0,(255,255,255),thickness=2)
                    cv2.drawContours(to_mask_white,[box2],0,(0,0,0),thickness=cv2.FILLED)
            elif min_x >= 30 and min_x <= 43:
                if min_x == rect[1][0]:
                    rect = ((rect[0][0], rect[0][1]), (2, rect[1][1]), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=cv2.FILLED)
                elif min_x == rect[1][1]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0], 2), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=cv2.FILLED)
                kernel = np.ones((3,3), np.uint8)
                res_gray =  cv2.medianBlur(res_gray, 5)
                #res_red = cv2.dilate(res_red, kernel, iterations=1)
                res_gray = cv2.erode(res_gray, kernel, iterations=2)
                contours, hierarchy = cv2.findContours(cv2.cvtColor(res_gray,cv2.COLOR_BGR2GRAY), 1, 1)
                detected_block = 0
                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    
                    if rect[1][0]>5 and rect[1][1]>5:# and (rect[1][1]>19 or rect[1][0] > 19):
                        if rect[1][0] > rect[1][1]:
                            rect2 = ((rect[0][0]-5, rect[0][1]-5), (rect[1][0], rect[1][1]+3), rect[2])
                        else: 
                            rect2 = ((rect[0][0]-5, rect[0][1]-5), (rect[1][0]+3, rect[1][1]), rect[2])
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        box2 = cv2.boxPoints(rect2)
                        box2 = np.int0(box2)
                        min_x = min(rect[1][0], rect[1][1])
                        max_x = max(rect[1][0], rect[1][1])
                        if max_x > 135:
                            max_x = max_x / 2
                            for key in gray_size.keys():
                                if int(max_x) in range(key[0], key[1]):
                                    gray_holes_cnt += 2*gray_size.get(key) + 2
                                    detected_block += 2
                            cv2.drawContours(to_mask_white,[box2],0,(0,0,0),thickness=cv2.FILLED)
                            cv2.drawContours(img_color_resized,[box],0,(0,0,255),2)
                        elif max_x > 20:
                            for key in gray_size.keys():
                                if int(max_x) in range(key[0], key[1]):
                                    gray_holes_cnt += gray_size.get(key)
                                    detected_block += 1
                            cv2.drawContours(to_mask_white,[box2],0,(0,0,0),thickness=cv2.FILLED)
                            cv2.drawContours(img_color_resized,[box],0,(0,0,255),2)
                break
            else:
                if min_x == rect[1][0]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0]/3, rect[1][1]), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=2)
                elif min_x == rect[1][1]:
                    rect = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/3), rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=2)
                kernel = np.ones((3,3), np.uint8)
                res_gray =  cv2.medianBlur(res_gray, 5)
                #res_red = cv2.dilate(res_red, kernel, iterations=1)
                res_gray = cv2.erode(res_gray, kernel, iterations=2)
                contours, hierarchy = cv2.findContours(cv2.cvtColor(res_gray,cv2.COLOR_BGR2GRAY), 1, 1)
                detected_block = 0
                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    if rect[1][0] > rect[1][1]:
                        rect2 = ((rect[0][0]-5, rect[0][1]-5), (rect[1][0], rect[1][1]+3), rect[2])
                    else: 
                        rect2 = ((rect[0][0]-5, rect[0][1]-5), (rect[1][0]+3, rect[1][1]), rect[2])
                    if rect[1][0]>5 and rect[1][1]>5:# and (rect[1][1]>19 or rect[1][0] > 19):
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        box2 = cv2.boxPoints(rect2)
                        box2 = np.int0(box2)
                        min_x = min(rect[1][0], rect[1][1])
                        max_x = max(rect[1][0], rect[1][1])
                        if min_x < 20:
                            if max_x > 135:
                                max_x = max_x / 2
                                for key in gray_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        gray_holes_cnt += 2*gray_size.get(key) + 2
                                        detected_block += 2
                                cv2.drawContours(to_mask_white,[box2],0,(0,0,0),thickness=cv2.FILLED)
                                cv2.drawContours(img_color_resized,[box],0,(0,0,255),2)
                            elif max_x > 20:
                                for key in gray_size.keys():
                                    if int(max_x) in range(key[0], key[1]):
                                        gray_holes_cnt += gray_size.get(key)
                                        detected_block += 1
                                cv2.drawContours(to_mask_white,[box2],0,(0,0,0),thickness=cv2.FILLED)
                                cv2.drawContours(img_color_resized,[box],0,(0,0,255),2)
                break
    to_mask_white = cv2.resize(to_mask_white, None, fx=3, fy=3)
    mask_white = np.where(to_mask_white == [0,0,0], 255, mask_white)
    return gray_holes_cnt, detected_block, mask_white
        

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
            

def mask_gray_to_holes(img_hsv, img_color_resized):
    lower_gray = np.array([40, 10, 0])
    upper_gray = np.array([100, 245, 205])
    #lower_gray = np.array([0, 0, 100])
    #upper_gray = np.array([50, 85, 255])
    mask_gray = cv2.inRange(img_hsv, lower_gray, upper_gray)
    #img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2HSV)
    #img_color_resized[:,:, 1] = 0 
    #img_color_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_HSV2BGR)
    res_gray = cv2.bitwise_and(img_color_resized, img_color_resized, mask = mask_gray)
    kernel = np.ones((2,2), np.uint8) 
    res_gray = cv2.medianBlur(res_gray, 3)
    res_gray = cv2.dilate(res_gray, kernel, iterations=6)
    res_gray= cv2.erode(res_gray, kernel, iterations=3)
    res_gray = cv2.medianBlur(res_gray, 5)
    return res_gray