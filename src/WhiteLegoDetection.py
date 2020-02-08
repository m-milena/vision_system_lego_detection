import cv2
import math
import numpy as np

class WhiteLegoDetection():
    def __init__(self, img_original, img_to_draw):
        super().__init__()
        self.img_original = img_original
        self.img_to_draw = img_to_draw
        self.check_number_of_blocks = {
                                        (30, 60): 1,
                                        (60, 100): 2,
                                        (100, 130): 3,
                                        (130, 160): 4
        }

    def mask_basic_detection(self, img_hsv):
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([179, 40, 225])
        mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
        res_white = cv2.bitwise_and(self.img_original, self.img_original, mask = mask_white)
        contours, hierarchy = cv2.findContours(cv2.cvtColor(res_white, cv2.COLOR_BGR2GRAY), 1, 2)
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            if rect[1][0] > 4 and rect[1][0] <10 and rect[1][1] > 4 and rect[1][1] < 10:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(res_white,[box],0,(0,0,0),thickness=cv2.FILLED)
        return res_white

    def mask_block_detection(self, img_hsv, img):
        lower_white = np.array([0, 0, 70])
        upper_white = np.array([80, 45, 255])
        mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
        res_white = cv2.bitwise_and(img, img, mask = mask_white)
        return res_white

    def count_blocks(self, block_mask):
        contours, hierarchy = cv2.findContours(cv2.cvtColor(block_mask, cv2.COLOR_BGR2GRAY), 1, 2)
        white_blocks_counter = 0
        contours = contours[:-1]
        hierarchy = hierarchy[:-1]
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            min_x = min(rect[1][0], rect[1][1])
            max_x = max(rect[1][0], rect[1][1])
            if min_x >= 30 and min_x <= 60:
                white_blocks_counter += 1
            elif min_x > 60 and min_x < 100:
                white_blocks_counter += 2
            elif min_x >=100 and min_x < 130:
                white_blocks_counter += 3
            elif min_x >= 130 and min_x <= 160:
                white_blocks_counter += 4
        return white_blocks_counter

    def count_holes(self, mask_white):
        contours, hierarchy = cv2.findContours(cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY), 1, 2)
        white_holes_cnt = 0
        to_detect_blocks = np.ones((mask_white.shape[0], mask_white.shape[1], 3), np.uint8)*255
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            rect2 = (rect[0], rect[1], 0)
            if rect[1][0] > 15 and rect[1][0] < 30  and rect[1][1] > 15 and rect[1][1] < 30 and math.fabs(rect[1][1]-rect[1][0])<8:
                box = cv2.boxPoints(rect2)
                box = np.int0(box)
                cX = int((box[2][0] + box[1][0])/2)
                cY = int((box[0][1] + box[1][1])/2)
                if mask_white[cY, cX, 0] == 0:
                    cv2.drawContours(to_detect_blocks,[box],0,(0,0,0),cv2.FILLED)
                    cv2.drawContours(to_detect_blocks,[box],0,(0,0,0),10)
                    white_holes_cnt += 1
        kernel = np.ones((3,3), np.uint8)
        to_detect_blocks = cv2.erode(to_detect_blocks, kernel, iterations=4)
        block_cnt = self.count_blocks(to_detect_blocks)
        return white_holes_cnt, block_cnt, to_detect_blocks