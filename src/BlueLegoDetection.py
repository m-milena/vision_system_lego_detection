import cv2
import math
import numpy as np

class BlueLegoDetection():
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
        lower_blue = np.array([105,20,10])
        upper_blue = np.array([135, 255, 255])
        mask_blue = cv2.inRange(self.img_hsv, lower_blue, upper_blue)
        img_to_mask = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2HSV)
        img_to_mask[:,:,1] = 0
        img_to_mask = cv2.cvtColor(img_to_mask, cv2.COLOR_HSV2BGR)
        res_blue = cv2.bitwise_and(img_to_mask, img_to_mask, mask = mask_blue)
        return res_blue

    def mask_block_detection(self, img_hsv, img):
        lower_blue = np.array([105,20,10])
        upper_blue = np.array([135, 255, 255])
        mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
        img_to_mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_to_mask[:,:,1] = 0
        img_to_mask = cv2.cvtColor(img_to_mask, cv2.COLOR_HSV2BGR)
        res_blue = cv2.bitwise_and(img_to_mask, img_to_mask, mask = mask_blue)
        kernel = np.ones((2,2), np.uint8)
        res_blue = cv2.medianBlur(res_blue, 3)
        res_blue = cv2.dilate(res_blue, kernel, iterations=3)
        return res_blue

    def separeta_two_blocks(self, rect, res_blue):
        if rect[1][0] < rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (15, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=cv2.FILLED)
        elif rect[1][0] >= rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0], 15), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=cv2.FILLED)
        
        kernel = np.ones((3,3), np.uint8)
        res_blue =  cv2.medianBlur(res_blue, 5)
        res_blue = cv2.erode(res_blue, kernel, iterations=2)
        return res_blue


    def separeta_three_blocks(self, rect, res_blue):
        if rect[1][0] < rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0]/3, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=7)
        elif rect[1][0] >= rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/3), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=7)
        kernel = np.ones((3,3), np.uint8)
        res_blue =  cv2.medianBlur(res_blue, 5)
        res_blue = cv2.erode(res_blue, kernel, iterations=1)
        return res_blue

    def separate_four_blocks(self, rect, res_blue):
        if rect[1][0] < rect[1][1]:
            rect2 = ((rect[0][0], rect[0][1]), (15, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect2)
            box = np.int0(box)
            cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=cv2.FILLED)
            rect3 = ((rect[0][0], rect[0][1]), (rect[1][0]/2, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect3)
            box = np.int0(box)
            cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=7)
        elif rect[1][0] >= rect[1][1]:
            rect2 = ((rect[0][0], rect[0][1]), (rect[1][0], 15), rect[2])
            box = cv2.boxPoints(rect2)
            box = np.int0(box)
            cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=cv2.FILLED)
            rect3 = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/2), rect[2])
            box = cv2.boxPoints(rect3)
            box = np.int0(box)
            cv2.drawContours(res_blue,[box],0,(0,0,0),thickness=7)
        kernel = np.ones((3,3), np.uint8)
        res_blue =  cv2.medianBlur(res_blue, 5)
        res_blue = cv2.erode(res_blue, kernel, iterations=2)
        return res_blue

    def separate_multiple_blocks(self, mask_blue):
        min_block = 0
        while min_block != 1:
            min_block = 1
            contours, hierarchy = cv2.findContours(cv2.cvtColor(mask_blue,cv2.COLOR_BGR2GRAY), 1, 1)
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
                        mask_blue = self.separeta_two_blocks(rect, mask_blue)
                    elif current_block == 3:
                        mask_blue = self.separeta_three_blocks(rect, mask_blue)
                    elif current_block == 4:
                        mask_blue = self.separate_four_blocks(rect, mask_blue)
        return mask_blue
                    
    def count_holes(self, mask_blue, mask_white):
        mask_blue = self.separate_multiple_blocks(mask_blue)
        contours, hierarchy = cv2.findContours(cv2.cvtColor(mask_blue,cv2.COLOR_BGR2GRAY), 1, 1)
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
