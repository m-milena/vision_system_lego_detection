import cv2
import math
import numpy as np

class RedLegoDetection():
    def __init__(self, img_original, img_hsv, img_to_draw):
        super().__init__()
        self.img_original = img_original
        self.img_hsv = img_hsv
        self.img_to_draw = img_to_draw
        self.blocks_sizes = {
                                (50, 100): 1,
                                (100, 180): 3,
                                (180, 245): 5,
                                (245, 320): 7,
                                (320, 400): 9
        }
        self.check_number_of_blocks = {
                                        (30, 75): 1,
                                        (75, 110): 2,
                                        (110, 160): 3,
                                        (160, 200): 4
        }

    def mask_basic_detection(self):
        lower_red_first = np.array([0,50,30])
        upper_red_first = np.array([10,255,255])
        mask_red_first = cv2.inRange(self.img_hsv, lower_red_first, upper_red_first)
        lower_red_second = np.array([170,50,30])
        upper_red_second = np.array([179,255,255])
        mask_red_second = cv2.inRange(self.img_hsv, lower_red_second, upper_red_second)
        img_to_mask = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2HSV)
        img_to_mask[:,:, 1] = 0 
        img_to_mask= cv2.cvtColor(img_to_mask, cv2.COLOR_HSV2BGR)
        res_red_first = cv2.bitwise_and(img_to_mask, img_to_mask, mask = mask_red_first)
        res_red_second = cv2.bitwise_and(img_to_mask, img_to_mask, mask = mask_red_second)
        res_red = cv2.add(res_red_first, res_red_second)
        kernel = np.ones((1,1), np.uint8)
        res_red =  cv2.medianBlur(res_red, 1)
        res_red = cv2.dilate(res_red, kernel, iterations=1)
        return res_red

    def mask_block_detection(self, img_hsv, img):
        lower_red_first = np.array([0,50,0])
        upper_red_first = np.array([12,255,255])
        mask_red_first = cv2.inRange(img_hsv, lower_red_first, upper_red_first)
        lower_red_second = np.array([170,50,0])
        upper_red_second = np.array([179,255,255])
        mask_red_second = cv2.inRange(img_hsv, lower_red_second, upper_red_second)
        img_to_mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_to_mask[:,:, 1] = 0 
        img_to_mask = cv2.cvtColor(img_to_mask, cv2.COLOR_HSV2BGR)
        res_red_first = cv2.bitwise_and(img_to_mask, img_to_mask, mask = mask_red_first)
        res_red_second = cv2.bitwise_and(img_to_mask, img_to_mask, mask = mask_red_second)
        res_red = cv2.add(res_red_first, res_red_second)
        return res_red

    def separeta_two_blocks(self, rect, res_red):
        if rect[1][0] < rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (10, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_red,[box],0,(0,0,0),thickness=cv2.FILLED)
        elif rect[1][0] >= rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0], 10), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_red,[box],0,(0,0,0),thickness=cv2.FILLED)
        
        kernel = np.ones((3,3), np.uint8)
        res_red =  cv2.medianBlur(res_red, 5)
        res_red = cv2.erode(res_red, kernel, iterations=1)
        return res_red


    def separeta_three_blocks(self, rect, res_red):
        if rect[1][0] < rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0]/3, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_red,[box],0,(0,0,0),thickness=5)
        elif rect[1][0] >= rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/3), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_red,[box],0,(0,0,0),thickness=5)
        kernel = np.ones((3,3), np.uint8)
        res_red =  cv2.medianBlur(res_red, 5)
        res_red = cv2.erode(res_red, kernel, iterations=1)
        return res_red

    def separate_four_blocks(self, rect, res_red):
        if rect[1][0] < rect[1][1]:
            rect2 = ((rect[0][0], rect[0][1]), (10, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect2)
            box = np.int0(box)
            cv2.drawContours(res_red,[box],0,(0,0,0),thickness=cv2.FILLED)
            rect3 = ((rect[0][0], rect[0][1]), (rect[1][0]/2, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect3)
            box = np.int0(box)
            cv2.drawContours(res_red,[box],0,(0,0,0),thickness=5)
        elif rect[1][0] >= rect[1][1]:
            rect2 = ((rect[0][0], rect[0][1]), (rect[1][0], 10), rect[2])
            box = cv2.boxPoints(rect2)
            box = np.int0(box)
            cv2.drawContours(res_red,[box],0,(0,0,0),thickness=cv2.FILLED)
            rect3 = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/2), rect[2])
            box = cv2.boxPoints(rect3)
            box = np.int0(box)
            cv2.drawContours(res_red,[box],0,(0,0,0),thickness=5)
        kernel = np.ones((3,3), np.uint8)
        res_red =  cv2.medianBlur(res_red, 5)
        res_red = cv2.erode(res_red, kernel, iterations=1)
        return res_red

    def separate_multiple_blocks(self, mask_red):
        min_block = 0

        while min_block != 1:
            min_block = 1
            contours, hierarchy = cv2.findContours(cv2.cvtColor(mask_red,cv2.COLOR_BGR2GRAY), 1, 1)
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                min_x = min(rect[1][0], rect[1][1])
                max_x = max(rect[1][0], rect[1][1])
                if min_x > 35 and max_x > 50:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    for key in self.check_number_of_blocks.keys():
                        if int(min_x) in range(key[0], key[1]):
                            current_block = self.check_number_of_blocks.get(key)
                            min_block = max(min_block, current_block)
                    if current_block == 2:
                        mask_red = self.separeta_two_blocks(rect, mask_red)
                    elif current_block == 3:
                        mask_red = self.separeta_three_blocks(rect, mask_red)
                    elif current_block == 4:
                        mask_red = self.separate_four_blocks(rect, mask_red)
        return mask_red

    def count_holes(self, mask_red, mask_white):
        mask_red = self.separate_multiple_blocks(mask_red)
        contours, hierarchy = cv2.findContours(cv2.cvtColor(mask_red,cv2.COLOR_BGR2GRAY), 1, 1)
        holes_cnt = 0
        blocks_cnt = 0
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            min_x = min(rect[1][0], rect[1][1])
            max_x = max(rect[1][0], rect[1][1])
            if min_x >= 30 and max_x >= 50:  
                box = cv2.boxPoints(rect)
                box = np.int0(box)     
                if max_x > 400:
                    max_x = max_x / 2
                    for key in self.blocks_sizes.keys():
                        if int(max_x) in range(key[0], key[1]):
                            holes_cnt += 2*self.blocks_sizes.get(key) + 2
                            blocks_cnt += 2
                else:
                    for key in self.blocks_sizes.keys():
                        if int(max_x) in range(key[0], key[1]):
                            holes_cnt += self.blocks_sizes.get(key)
                            blocks_cnt += 1
                cv2.drawContours(self.img_to_draw,[box],0,(0,0,0),2)
                cv2.drawContours(mask_white,[box],0,(255,255,255),thickness=cv2.FILLED)
        return holes_cnt, blocks_cnt, mask_white
