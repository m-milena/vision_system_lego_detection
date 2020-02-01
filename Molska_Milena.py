import sys
import os
import cv2
import copy
import json
import math
import numpy as np
import detect_colors.detect_yellow_blocks as detect_yellow_blocks
import detect_colors.detect_red_blocks as detect_red_blocks
import detect_colors.detect_blue_blocks as detect_blue_blocks
import detect_colors.detect_white_block as detect_white_block
import detect_colors.detect_gray_blocks as detect_gray_blocks

def process_white(res_blue, res_gray, res_yellow, res_red, img_color):
    #res_blue = cv2.medianBlur(res_blue, 5)
    #kernel = np.ones((2,2), np.uint8) 
    #res_blue = cv2.dilate(res_blue, kernel, iterations=2)
    img_2 = np.where(res_blue != [0, 0, 0], 255, res_blue)
    img_2 = cv2.bitwise_not(img_2)
    img_color_2 = cv2.bitwise_and(img_color, img_2)

    #res_yellow = cv2.medianBlur(res_yellow, 3)
    #kernel = np.ones((3,3), np.uint8) 
    #res_yellow = cv2.dilate(res_yellow, kernel, iterations=1)
    img_2 = np.where(res_yellow != [0, 0, 0], 255, res_yellow)
    img_2 = cv2.bitwise_not(img_2)
    img_color_2 = cv2.bitwise_and(img_color_2, img_2)

    #res_red = cv2.medianBlur(res_red, 5)
    #kernel = np.ones((3,3), np.uint8) 
    #res_red = cv2.dilate(res_red, kernel, iterations=2)
    img_2 = np.where(res_red != [0, 0, 0], 255, res_red)
    img_2 = cv2.bitwise_not(img_2)
    img_color_2 = cv2.bitwise_and(img_color_2, img_2)

    #res_gray = cv2.medianBlur(res_gray, 3)
    #kernel = np.ones((3,3), np.uint8) 
    #res_gray = cv2.erode(res_gray, kernel, iterations=4)
    #res_gray = cv2.dilate(res_gray, kernel, iterations=1)
    kernel = np.ones((2,2), np.uint8) 
    res_gray = cv2.erode(res_gray, kernel, iterations=2)
    img_2 = np.where(res_gray != [0, 0, 0], 255, res_gray)
    img_2 = cv2.bitwise_not(img_2)
    img_color_2 = cv2.bitwise_and(img_color_2, img_2)

    return img_color_2, img_2

def detect_groups(res, img_color):
    rectangles_array = []
    points_array = []
    contours, hierarchy = cv2.findContours(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 1, 2)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > 20 and rect[1][1] > 20:
            if rect[1][0]>rect[1][1]:
                if rect[1][0] < img_color.shape[0] and rect[1][1] < img_color.shape[1]:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img_color,[box],0,(210,210,20),2)
            elif rect[1][0]<rect[1][1]:
                if rect[1][1] < img_color.shape[0] and rect[1][0] < img_color.shape[1]:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img_color,[box],0,(210,210,20),2)
            points_array.append(box)
            rectangles_array.append(rect)
    return rectangles_array, points_array

def split_group(rects, points, img):
    r = rects
    p = points
    pts1 = np.float32([[p[0][0], p[0][1]], [p[1][0],p[1][1]], [p[3][0],p[3][1]], [p[2][0],p[2][1]]])
    pts2 = np.float32([[0,0], [3*r[1][1], 0], [0, 3*r[1][0]], [3*r[1][1], 3*r[1][0]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    box = cv2.warpPerspective(img, M, (3*int(r[1][1]), 3*int(r[1][0])))
    return box

def split_group_normal_size(rects, points, img):
    r = rects
    p = points
    pts1 = np.float32([[p[0][0], p[0][1]], [p[1][0],p[1][1]], [p[3][0],p[3][1]], [p[2][0],p[2][1]]])
    pts2 = np.float32([[0,0], [r[1][1], 0], [0, r[1][0]], [r[1][1], r[1][0]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    box = cv2.warpPerspective(img, M, (int(r[1][1]), int(r[1][0])))
    return box

def preprocess_group_detection(img_hsv_original, img_hsv_clahe, img, img_clahe):
    # Detect every color to get mask
    mask_yellow = detect_yellow_blocks.yellow_blocks(img_hsv_original, img)
    mask_red = detect_red_blocks.mask_red(img_hsv_clahe, img)
    mask_blue = detect_blue_blocks.mask_blue(img_hsv_clahe, img)
    mask_gray = detect_gray_blocks.mask_gray(img_hsv_clahe, img)
    # Merge all color masks
    color_mask = cv2.add(mask_yellow, mask_red)
    color_mask = cv2.add(color_mask, mask_blue)
    color_mask = cv2.add(color_mask, mask_gray)

    # Preprocess to detect white
    img_2 = np.where(color_mask != [0, 0, 0], 255, color_mask)
    img_2 = cv2.bitwise_not(img_2)
    img_to_white = cv2.bitwise_and(img_clahe, img_2)
    x,y = process_white(mask_blue, mask_gray, mask_yellow, mask_red, img)
    # Mask white and detect holes
    mask_white = detect_white_block.mask_white(cv2.cvtColor(x, cv2.COLOR_BGR2HSV), img_hsv_clahe, img)
    # Detect red, blue, yellow blocks
    detect_red_blocks.detect_blocks(mask_red, img)
    detect_yellow_blocks.yellow_holes(mask_yellow, img)
    detect_blue_blocks.detect_blocks(mask_blue, img)
    detect_gray_blocks.detect_blocks(mask_gray, img)
    # Preprocess groups detection
    kernel = np.ones((3,3), np.uint8) 
    mask_white = cv2.dilate(mask_white, kernel, iterations=1)
    mask_white = cv2.erode(mask_white, kernel, iterations=5)
    return mask_white

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)


def function_for_detection(img):
    img_original = copy.deepcopy(img)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    #clahe
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(25,25))
    l = clahe.apply(l)
    img_clahe = cv2.merge((l,a,b))
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)

    img_hsv_original = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2HSV)

    # Preprocess block detection
    masked_blocks = preprocess_group_detection(img_hsv_original, img_hsv_clahe, img, img_clahe)

    # Split groups on different images
    rects, points = detect_groups(masked_blocks, img)
    rects = rects[:-1]
    points = points[:-1]
    groups = []
    groups2 = []
    groups3 = []

    blocks_info_array = []
    for r,p in zip(rects, points):
        groups.append(split_group(r, p, img_clahe))
        groups2.append(split_group(r, p, img_original))
        groups3.append(split_group_normal_size(r,p,img_clahe))

    for i in range(0, len(groups)):
        group = groups[i]
        group2 = groups2[i]
        group3 = groups3[i]
        # Detect colours on groups
        mask_blue = detect_blue_blocks.mask_blue_to_holes(cv2.cvtColor(group, cv2.COLOR_BGR2HSV), group)
        mask_yellow = detect_yellow_blocks.mask_yellow_to_holes(cv2.cvtColor(group2, cv2.COLOR_BGR2HSV), group2)
        mask_red = detect_red_blocks.mask_red_to_holes(cv2.cvtColor(group2, cv2.COLOR_BGR2HSV), group2)
        #mask_gray = detect_gray_blocks.mask_gray_to_holes(cv2.cvtColor(group3, cv2.COLOR_BGR2HSV), group3)
        mask_gray = detect_gray_blocks.mask_gray_to_holes(cv2.cvtColor(group3, cv2.COLOR_BGR2HSV), group3)
        mask_white = detect_white_block.mask_white2(cv2.cvtColor(group, cv2.COLOR_BGR2HSV), group)
        #mask_white = adjust_gamma(mask_white, 0.5)
        mask_white = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY)
        _, mask_white = cv2.threshold(mask_white, 105, 255, cv2.THRESH_BINARY)
        mask_white = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
        # DONE: yellow, red, gray, blue
        blue_holes, blue_blocks = detect_blue_blocks.detect_holes(mask_blue, group, mask_white)
        yellow_holes, yellow_blocks = detect_yellow_blocks.detect_holes(mask_yellow, group2, mask_white)
        red_holes, red_blocks = detect_red_blocks.detect_holes(mask_red, group2, mask_white)
        gray_holes, gray_blocks, mask_white= detect_gray_blocks.detect_blocks2(mask_gray, group3, mask_white)
        white_holes, white_blocks_detection = detect_white_block.white_holes(mask_white, group)
        white_blocks = detect_white_block.detect_white_blocks(white_blocks_detection, group)
        #Print results
        info = {'red': (red_blocks, red_holes), 'blue': (blue_blocks, blue_holes), \
            'white': (white_blocks, white_holes), 'grey': (gray_blocks, gray_holes), 'yellow': (yellow_blocks, yellow_holes)}
        blocks_info_array.append(info)

        #key = ord('a')
        #result = np.hstack((group, mask_blue))
        #while key != ord('q'):
        #    cv2.imshow('r', result)
        #    key = cv2.waitKey(30)
        #cv2.destroyAllWindows()
    
    return blocks_info_array


def main():
    if len(sys.argv) != 4:
        print('Wrong number of arguments')
    else: 
        imgs_path = sys.argv[1]
        json_path = sys.argv[2]
        output_path = sys.argv[3]

        with open(json_path) as json_file:
            json_data = json.load(json_file)

        imgs_names = [f for f in os.listdir(imgs_path) \
                if os.path.isfile(os.path.join(imgs_path, f))]
        
        data_to_write = {}

       # imgs_names = [imgs_names[9]]
        for img in imgs_names:
            print(img)
            current_img = cv2.imread(imgs_path+'/'+img, cv2.IMREAD_COLOR)
            current_img = cv2.resize(current_img, None, fx=0.15, fy=0.15)
            info_array = function_for_detection(current_img)
            data_to_current_img = json_data.get(img[:-4])
            found_holes = []
            for i in range(0, len(data_to_current_img)):
                for j in range(0, len(info_array)):
                    found = 0
                    holes = 0
                    if int(data_to_current_img[i].get('red')) == int(info_array[j].get('red')[0]):
                        found += 1
                        holes += info_array[j].get('red')[1]
                    if int(data_to_current_img[i].get('blue')) == int(info_array[j].get('blue')[0]):
                        found += 1
                        holes += info_array[j].get('blue')[1]
                    if int(data_to_current_img[i].get('grey')) == int(info_array[j].get('grey')[0]):
                        found += 1
                        holes += info_array[j].get('grey')[1]
                    if int(data_to_current_img[i].get('yellow')) == int(info_array[j].get('yellow')[0]):
                        found += 1
                        holes+= info_array[j].get('yellow')[1]
                    if int(data_to_current_img[i].get('white')) == int(info_array[j].get('white')[0]):
                        found += 1
                        holes += info_array[j].get('white')[1]
                    if found == 5:
                        found_holes.append(holes)
                        data_to_current_img[i] = 0
                        info_array.pop(j)
                        break
                if len(found_holes) <= i:
                    found_holes.append(0)
            for i in range(0, len(data_to_current_img)):
                if data_to_current_img[i]:
                    error_table = []
                    holes_table = []
                    for j in range(0, len(info_array)):
                        calculate_error = 0
                        holes_2 = 0
                        if int(data_to_current_img[i].get('white')):
                            calculate_error = calculate_error + math.fabs(int(data_to_current_img[i].get('red')) - int(info_array[j].get('red')[0]))\
                            + math.fabs(int(data_to_current_img[i].get('blue')) - int(info_array[j].get('blue')[0]))\
                            + math.fabs(int(data_to_current_img[i].get('grey')) - int(info_array[j].get('grey')[0]))\
                            + math.fabs(int(data_to_current_img[i].get('yellow')) - int(info_array[j].get('yellow')[0]))\
                            + math.fabs(int(data_to_current_img[i].get('white')) - int(info_array[j].get('white')[0]))
                        else: 
                            calculate_error = calculate_error + math.fabs(int(data_to_current_img[i].get('red')) - int(info_array[j].get('red')[0]))\
                            + math.fabs(int(data_to_current_img[i].get('blue')) - int(info_array[j].get('blue')[0]))\
                            + math.fabs(int(data_to_current_img[i].get('grey')) - int(info_array[j].get('grey')[0]))\
                            + math.fabs(int(data_to_current_img[i].get('yellow')) - int(info_array[j].get('yellow')[0]))

                        if int(data_to_current_img[i].get('red')) and int(info_array[j].get('red')[0]):
                            holes_2 += info_array[j].get('red')[1]
                        if int(data_to_current_img[i].get('blue')) and int(info_array[j].get('blue')[0]):
                            holes_2 += info_array[j].get('blue')[1]
                        if int(data_to_current_img[i].get('grey')) and int(info_array[j].get('grey')[0]):
                            holes_2 += info_array[j].get('grey')[1]
                        if int(data_to_current_img[i].get('yellow')) and int(info_array[j].get('yellow')[0]):
                            holes_2 += info_array[j].get('yellow')[1]
                        if int(data_to_current_img[i].get('white')) and int(info_array[j].get('white')[0]):
                            diff = math.fabs(int(data_to_current_img[i].get('white'))-int(info_array[j].get('white')[0]))
                            holes_2  += info_array[j].get('white')[1] - diff
                        error_table.append(calculate_error)
                        holes_table.append(holes_2)
                    index = min(range(len(error_table)), key=error_table.__getitem__)
                    found_holes[i] = holes_table[index]
                    info_array.pop(index)

            data_to_write[img[:-4]] = found_holes
            with open(output_path, 'w+') as json_file:
                json_string = json.dumps(data_to_write)
                json_file.write(json_string)

        print(data_to_write)

if __name__ == "__main__":
    main()