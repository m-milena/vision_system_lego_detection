import cv2
import numpy as np

class GrayLegoDetection():
    def __init__(self, img_original, img_hsv, img_to_draw):
        super().__init__()
        self.img_original = img_original
        self.img_hsv = img_hsv
        self.img_to_draw = img_to_draw
        self.blocks_sizes = {
                                (20, 41): 1,
                                (41, 63): 3,
                                (63, 83): 5,
                                (83, 105): 7,
                                (105, 135): 9
        }
        self.check_number_of_blocks = {
                                        (8, 28): 1,
                                        (28, 43): 2,
                                        (43, 60): 3,
                                        (60, 78): 4
        }

    def mask_basic_detection(self):
        lower_gray = np.array([40, 10, 0])
        upper_gray = np.array([100, 255, 205])
        mask_gray = cv2.inRange(self.img_hsv, lower_gray, upper_gray)
        img_to_mask = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2HSV)
        img_to_mask[:,:,1] = 0
        img_to_mask = cv2.cvtColor(img_to_mask, cv2.COLOR_HSV2BGR)
        res_gray = cv2.bitwise_and(img_to_mask, img_to_mask, mask = mask_gray)
        kernel = np.ones((3,3), np.uint8) 
        res_gray = cv2.medianBlur(res_gray, 3)
        res_gray = cv2.dilate(res_gray, kernel, iterations=3)
        res_gray= cv2.erode(res_gray, kernel, iterations=2)
        res_gray = cv2.medianBlur(res_gray, 5)
        return res_gray

    def mask_block_detection(self, img_hsv, img):
        lower_gray = np.array([40, 10, 0])
        upper_gray = np.array([100, 245, 205])
        mask_gray = cv2.inRange(img_hsv, lower_gray, upper_gray)
        res_gray = cv2.bitwise_and(img, img, mask = mask_gray)
        kernel = np.ones((2,2), np.uint8) 
        res_gray = cv2.medianBlur(res_gray, 3)
        res_gray = cv2.dilate(res_gray, kernel, iterations=6)
        res_gray= cv2.erode(res_gray, kernel, iterations=3)
        res_gray = cv2.medianBlur(res_gray, 5)
        return res_gray

    def separeta_two_blocks(self, rect, res_gray):
        if rect[1][0] < rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (2, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=cv2.FILLED)
        elif rect[1][0] >= rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0], 2), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=cv2.FILLED)
        
        kernel = np.ones((3,3), np.uint8)
        res_gray =  cv2.medianBlur(res_gray, 5)
        res_gray = cv2.erode(res_gray, kernel, iterations=2)
        return res_gray


    def separeta_three_blocks(self, rect, res_gray):
        if rect[1][0] < rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0]/3, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=2)
        elif rect[1][0] >= rect[1][1]:
            rect = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/3), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=2)
        kernel = np.ones((3,3), np.uint8)
        res_gray =  cv2.medianBlur(res_gray, 5)
        res_gray = cv2.erode(res_gray, kernel, iterations=1)
        return res_gray

    def separate_four_blocks(self, rect, res_gray):
        if rect[1][0] < rect[1][1]:
            rect2 = ((rect[0][0], rect[0][1]), (2, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect2)
            box = np.int0(box)
            cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=cv2.FILLED)
            rect3 = ((rect[0][0], rect[0][1]), (rect[1][0]/2, rect[1][1]), rect[2])
            box = cv2.boxPoints(rect3)
            box = np.int0(box)
            cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=2)
        elif rect[1][0] >= rect[1][1]:
            rect2 = ((rect[0][0], rect[0][1]), (rect[1][0], 2), rect[2])
            box = cv2.boxPoints(rect2)
            box = np.int0(box)
            cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=cv2.FILLED)
            rect3 = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]/2), rect[2])
            box = cv2.boxPoints(rect3)
            box = np.int0(box)
            cv2.drawContours(res_gray,[box],0,(0,0,0),thickness=2)
        kernel = np.ones((3,3), np.uint8)
        res_gray =  cv2.medianBlur(res_gray, 5)
        res_gray = cv2.erode(res_gray, kernel, iterations=2)
        return res_gray

    def separate_multiple_blocks(self, mask_gray):
        min_block = 0

        while min_block != 1:
            min_block = 1
            contours, hierarchy = cv2.findContours(cv2.cvtColor(mask_gray,cv2.COLOR_BGR2GRAY), 1, 1)
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                min_x = min(rect[1][0], rect[1][1])
                max_x = max(rect[1][0], rect[1][1])
                if min_x >= 8 and max_x >= 20:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    for key in self.check_number_of_blocks.keys():
                        if int(min_x) in range(key[0], key[1]):
                            current_block = self.check_number_of_blocks.get(key)
                            min_block = max(min_block, current_block)
                    if current_block == 2:
                        mask_gray = self.separeta_two_blocks(rect, mask_gray)
                    elif current_block == 3:
                        mask_gray= self.separeta_three_blocks(rect, mask_gray)
                    elif current_block == 4:
                        mask_gray = self.separate_four_blocks(rect, mask_gray)
        return mask_gray


    def count_holes(self, mask_gray, mask_white):
        mask_gray = self.separate_multiple_blocks(mask_gray)
        contours, hierarchy = cv2.findContours(cv2.cvtColor(mask_gray,cv2.COLOR_BGR2GRAY), 1, 1)
        to_mask_white = np.ones([mask_gray.shape[0], mask_gray.shape[1],3],dtype=np.uint8)*255
        holes_cnt = 0
        blocks_cnt = 0
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            if rect[1][0] > rect[1][1]:
                rect2 = ((rect[0][0]-5, rect[0][1]-5), (rect[1][0], rect[1][1]-2), rect[2])
            else: 
                rect2 = ((rect[0][0]-5, rect[0][1]-5), (rect[1][0]-2, rect[1][1]), rect[2])
            min_x = min(rect[1][0], rect[1][1])
            max_x = max(rect[1][0], rect[1][1])
            if min_x >= 8 and max_x >= 20:  
                box = cv2.boxPoints(rect)
                box = np.int0(box) 
                box2 = cv2.boxPoints(rect2)
                box2 = np.int0(box2)    
                if max_x > 135:
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
                cv2.drawContours(to_mask_white,[box2],0,(0,0,0),thickness=cv2.FILLED)
        to_mask_white = cv2.resize(to_mask_white, None, fx=3, fy=3)
        mask_white = np.where(to_mask_white == [0,0,0], 255, mask_white)
        return holes_cnt, blocks_cnt, mask_white