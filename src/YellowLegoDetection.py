import cv2
import math
import numpy as np

class YellowLegoDetection():
    def __init__(self, img_original, img_hsv, img_to_draw):
        super().__init__()
        self.img_original = img_original
        self.img_hsv = img_hsv
        self.img_to_draw = img_to_draw
        self.blocks_sizes = {
                                (50, 90): 1,
                                (90, 190): 3,
                                (190, 255): 5,
                                (255, 320): 7,
                                (320, 400): 9
        }
        self.check_number_of_blocks = {
                                        (30, 75): 1,
                                        (75, 115): 2,
                                        (115, 160): 3,
                                        (160, 200): 4
        }

    def mask_basic_detection(self):
        lower_yellow = np.array([20, 65, 70])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(self.img_hsv, lower_yellow, upper_yellow)
        img_to_mask = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2HSV)
        img_to_mask[:,:,1] = 0
        img_to_mask = cv2.cvtColor(img_to_mask, cv2.COLOR_HSV2BGR)
        res_yellow = cv2.bitwise_and(img_to_mask, img_to_mask, mask = mask_yellow)
        kernel = np.ones((2,2), np.uint8) 
        res_yellow =  cv2.medianBlur(res_yellow, 3)
        res_yellow = cv2.dilate(res_yellow, kernel, iterations=1)
        res_yellow = cv2.erode(res_yellow, kernel, iterations=2)
        res_yellow =  cv2.medianBlur(res_yellow, 1)
        return res_yellow

    def mask_block_detection(self, img_hsv, img):
        lower_yellow = np.array([20, 80, 125])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        img_to_mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_to_mask[:,:,1] = 0
        img_to_mask = cv2.cvtColor(img_to_mask, cv2.COLOR_HSV2BGR)
        res_yellow = cv2.bitwise_and(img_to_mask, img_to_mask, mask = mask_yellow)
        _, res_yellow = cv2.threshold(res_yellow, 100, 255, cv2.THRESH_BINARY)
        return res_yellow

    def separeta_two_blocks(self, rect, res_yellow):
        if rect[1][0] < rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (15, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=cv2.FILLED)
        elif rect[1][0] >= rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0], 15), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=cv2.FILLED)
        
        kernel = np.ones((3,3), np.uint8)
        res_yellow =  cv2.medianBlur(res_yellow, 5)
        res_yellow = cv2.erode(res_yellow, kernel, iterations=3)
        res_yellow = cv2.dilate(res_yellow, kernel, iterations=4)
        return res_yellow


    def separeta_three_blocks(self, rect, res_yellow):
        if rect[1][0] < rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0]/3, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=5)
        elif rect[1][0] >= rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/3), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=5)
        kernel = np.ones((3,3), np.uint8)
        res_yellow =  cv2.medianBlur(res_yellow, 5)
        res_yellow = cv2.erode(res_yellow, kernel, iterations=1)
        return res_yellow

    def separate_four_blocks(self, rect, res_yellow):
        if rect[1][0] < rect[1][1]:
            rect2 = ((rect[0][0], rect[0][1]), (15, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect2)
            box = np.int0(box)
            cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=cv2.FILLED)
            rect3 = ((rect[0][0], rect[0][1]), (rect[1][0]/2, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect3)
            box = np.int0(box)
            cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=5)
        elif rect[1][0] >= rect[1][1]:
            rect2 = ((rect[0][0], rect[0][1]), (rect[1][0], 15), rect[2])
            box = cv2.boxPoints(rect2)
            box = np.int0(box)
            cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=cv2.FILLED)
            rect3 = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/2), rect[2])
            box = cv2.boxPoints(rect3)
            box = np.int0(box)
            cv2.drawContours(res_yellow,[box],0,(0,0,0),thickness=5)
        kernel = np.ones((3,3), np.uint8)
        res_yellow =  cv2.medianBlur(res_yellow, 5)
        res_yellow = cv2.erode(res_yellow, kernel, iterations=1)
        return res_yellow

    def separate_multiple_blocks(self, mask_yellow):
        min_block = 0

        while min_block != 1:
            min_block = 1
            contours, hierarchy = cv2.findContours(cv2.cvtColor(mask_yellow,cv2.COLOR_BGR2GRAY), 1, 1)
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
                        mask_yellow = self.separeta_two_blocks(rect, mask_yellow)
                    elif current_block == 3:
                        mask_yellow = self.separeta_three_blocks(rect, mask_yellow)
                    elif current_block == 4:
                        mask_yellow = self.separate_four_blocks(rect, mask_yellow)
        return mask_yellow

    def count_holes(self, mask_yellow, mask_white):
        mask_yellow = self.separate_multiple_blocks(mask_yellow)
        contours, hierarchy = cv2.findContours(cv2.cvtColor(mask_yellow,cv2.COLOR_BGR2GRAY), 1, 1)
        holes_cnt = 0
        blocks_cnt = 0
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            rect2 = (rect[0], (rect[1][0]/1.3, rect[1][1]/1.3), rect[2])
            min_x = min(rect[1][0], rect[1][1])
            max_x = max(rect[1][0], rect[1][1])
            if min_x >= 30 and max_x >= 50:  
                box = cv2.boxPoints(rect)
                box = np.int0(box)  
                box2 = cv2.boxPoints(rect2)
                box2 = np.int0(box2)   
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
                cv2.drawContours(mask_white,[box2],0,(255,255,255),thickness=cv2.FILLED)
        return holes_cnt, blocks_cnt, mask_white
