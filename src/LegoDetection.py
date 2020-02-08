import cv2
import numpy as np
from src.BlueLegoDetection import BlueLegoDetection
from src.YellowLegoDetection import YellowLegoDetection
from src.RedLegoDetection import RedLegoDetection
from src.GrayLegoDetection import GrayLegoDetection
from src.WhiteLegoDetection import WhiteLegoDetection

class LegoDetection():
    def __init__(self, img, scale):
        self.img_original = cv2.resize(img, None, fx=scale, fy=scale)
        self.img_to_draw = self.img_original
        self.img_hsv = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2HSV)
        self.img_clahe = self.clahe_processing()
        self.img_clahe_hsv = cv2.cvtColor(self.img_clahe, cv2.COLOR_BGR2HSV)
        self.blue = BlueLegoDetection(self.img_original, self.img_clahe_hsv, self.img_to_draw)
        self.red = RedLegoDetection(self.img_original, self.img_clahe_hsv, self.img_to_draw)
        self.yellow = YellowLegoDetection(self.img_original, self.img_hsv, self.img_to_draw)
        self.gray = GrayLegoDetection(self.img_original, self.img_clahe_hsv, self.img_to_draw)
        self.white = WhiteLegoDetection(self.img_original, self.img_to_draw)


    def clahe_processing(self):
        img_lab = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit = 10.0, tileGridSize=(25,25))
        l = clahe.apply(l)
        img_clahe = cv2.merge((l, a, b))
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
        return img_clahe

    def preprocess_white(self, res_blue, res_gray, res_yellow, res_red, img_color):
        img_2 = np.where(res_blue != [0, 0, 0], 255, res_blue)
        img_2 = cv2.bitwise_not(img_2)
        img_color_2 = cv2.bitwise_and(img_color, img_2)

        img_2 = np.where(res_yellow != [0, 0, 0], 255, res_yellow)
        img_2 = cv2.bitwise_not(img_2)
        img_color_2 = cv2.bitwise_and(img_color_2, img_2)

        img_2 = np.where(res_red != [0, 0, 0], 255, res_red)
        img_2 = cv2.bitwise_not(img_2)
        img_color_2 = cv2.bitwise_and(img_color_2, img_2)

        kernel = np.ones((2,2), np.uint8) 
        res_gray = cv2.erode(res_gray, kernel, iterations=2)
        img_2 = np.where(res_gray != [0, 0, 0], 255, res_gray)
        img_2 = cv2.bitwise_not(img_2)
        img_color_2 = cv2.bitwise_and(img_color_2, img_2)

        return img_color_2

    def preprocess_groups_detection(self):
        mask_blue = self.blue.mask_basic_detection()
        mask_red = self.red.mask_basic_detection()
        mask_yellow = self.yellow.mask_basic_detection()
        mask_gray = self.gray.mask_basic_detection()

        # Merge all colors
        color_mask = cv2.add(mask_yellow, mask_red)
        color_mask = cv2.add(color_mask, mask_blue)
        color_mask = cv2.add(color_mask, mask_gray)

        # Preprocess to detect white blocks
        img_to_white = np.where(color_mask != [0,0,0], 255, color_mask)
        img_to_white = cv2.bitwise_not(img_to_white)
        img_to_white = cv2.bitwise_and(self.img_clahe, img_to_white)

        img = self.img_original
        img_to_white_2 = self.preprocess_white(mask_blue, mask_gray, mask_yellow, mask_red, img)
        mask_white = self.white.mask_basic_detection(cv2.cvtColor(img_to_white_2, cv2.COLOR_BGR2HSV))
        kernel = np.ones((3,3), np.uint8) 
        mask_white = cv2.dilate(mask_white, kernel, iterations=1)
        mask_white = cv2.erode(mask_white, kernel, iterations=5)
        return mask_white
    
    def detect_groups(self):
        res = self.preprocess_groups_detection()
        rectangles_array = []
        points_array = []
        contours, hierarchy = cv2.findContours(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 1, 2)
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            if rect[1][0] > 20 and rect[1][1] > 20:
                min_x = min(rect[1][0], rect[1][1])
                max_x = max(rect[1][0], rect[1][1])
                if max_x < self.img_to_draw.shape[0] and min_x < self.img_to_draw.shape[1]:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    points_array.append(box)
                    rectangles_array.append(rect)
                    cv2.drawContours(self.img_to_draw, [box], 0, (255,0,0), 2)
        return rectangles_array, points_array

    def transform_group(self, r, p, img, k = 3):
        pts1 = np.float32([[p[0][0], p[0][1]], [p[1][0],p[1][1]], [p[3][0],p[3][1]], [p[2][0],p[2][1]]])
        pts2 = np.float32([[0,0], [k*r[1][1], 0], [0, k*r[1][0]], [k*r[1][1], k*r[1][0]]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        box = cv2.warpPerspective(img, M, (k*int(r[1][1]), k*int(r[1][0])))
        return box

    def count_holes(self):
        rects, points = self.detect_groups()
        group_clahe = []
        group_normal = []
        group_little = []
        for r,p in zip(rects, points):
            group_clahe.append(self.transform_group(r, p, self.img_clahe, k=3))
            group_normal.append(self.transform_group(r, p, self.img_original, k=3))
            group_little.append(self.transform_group(r, p,self.img_clahe, k=1))

        blocks_info_array = []
        for i in range(0, len(group_clahe)):
            mask_blue = self.blue.mask_block_detection(cv2.cvtColor(group_clahe[i], cv2.COLOR_BGR2HSV), group_clahe[i])
            mask_yellow = self.yellow.mask_block_detection(cv2.cvtColor(group_normal[i], cv2.COLOR_BGR2HSV), group_normal[i])
            mask_red = self.red.mask_block_detection(cv2.cvtColor(group_normal[i], cv2.COLOR_BGR2HSV), group_normal[i])
            mask_gray = self.gray.mask_block_detection(cv2.cvtColor(group_little[i], cv2.COLOR_BGR2HSV), group_little[i])
            mask_white = self.white.mask_block_detection(cv2.cvtColor(group_clahe[i], cv2.COLOR_BGR2HSV), group_clahe[i])
            mask_white = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY)
            _, mask_white = cv2.threshold(mask_white, 105, 255, cv2.THRESH_BINARY)
            mask_white = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
            blue_holes, blue_blocks, mask_white = self.blue.count_holes(mask_blue, mask_white)
            yellow_holes, yellow_blocks, mask_white = self.yellow.count_holes(mask_yellow, mask_white)
            red_holes, red_blocks, mask_white = self.red.count_holes(mask_red, mask_white)
            gray_holes, gray_blocks, mask_white = self.gray.count_holes(mask_gray, mask_white)
            white_holes, white_blocks, b = self.white.count_holes(mask_white)

            info = {'red': (red_blocks, red_holes), 'blue': (blue_blocks, blue_holes), \
            'white': (white_blocks, white_holes), 'grey': (gray_blocks, gray_holes), 'yellow': (yellow_blocks, yellow_holes)}
            blocks_info_array.append(info)

        return blocks_info_array




    
