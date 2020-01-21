import cv2
import copy
import numpy as np
import detect_yellow_blocks
import detect_red_blocks
import detect_blue_blocks
import detect_gray_blocks
import detect_white_block

def preprocess_gray(res_blue, res_white, img_color):
    res_blue = cv2.medianBlur(res_blue, 5)
    kernel = np.ones((3,3), np.uint8) 
    res_blue = cv2.dilate(res_blue, kernel, iterations=2)
    img_2 = np.where(res_blue != [0, 0, 0], 255, res_blue)
    img_2 = cv2.bitwise_not(img_2)
    img_color_2 = cv2.bitwise_and(img_color, img_2)
    res_white = cv2.medianBlur(res_white, 5)
    kernel = np.ones((3,3), np.uint8) 
    res_white = cv2.dilate(res_white, kernel, iterations=1)
    img_2 = np.where(res_white != [0, 0, 0], 255, res_white)
    img_2 = cv2.bitwise_not(img_2)
    img_color_2 = cv2.bitwise_and(img_color_2, img_2)
    return img_color_2, img_2

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

def detect_groups(img_color):
    contours, hierarchy = cv2.findContours(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY), 1, 2)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > 20 and rect[1][1] > 20:
            if rect[1][0]>rect[1][1]:
                if rect[1][0] < img_color.shape[0] and rect[1][1] < img_color.shape[1]:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img_color,[box],0,(20,20,100),2)
            elif rect[1][0]<rect[1][1]:
                if rect[1][1] < img_color.shape[0] and rect[1][0] < img_color.shape[1]:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img_color,[box],0,(20,20,100),2)
    
    
def main():
    img_color = cv2.imread("./imgs/img_001.jpg", cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_color= cv2.resize(img_color, None, fx=0.15, fy=0.15)
    img_gray = cv2.resize(img_gray, None, fx=0.15, fy=0.15)
    img_color_original = copy.deepcopy(img_color)

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    res_yellow = detect_yellow_blocks.yellow_blocks(img_hsv, img_color)
    res_red = detect_red_blocks.red_blocks(img_hsv, img_color)
    res_blue = detect_blue_blocks.blue_blocks(img_hsv, img_color)
    res_white = detect_white_block.white_blocks(img_hsv, img_color)
    img_color_2, img_2 = preprocess_gray(res_blue, res_white, img_color)
    res_gray = detect_gray_blocks.gray_blocks(cv2.cvtColor(img_color_2, cv2.COLOR_BGR2HSV), img_2)
    r, d = process_white(res_blue, res_gray, res_yellow, res_red, img_color)
    res_white = detect_white_block.white_blocks(cv2.cvtColor(r, cv2.COLOR_BGR2HSV), img_color)

    edges = cv2.Canny(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY),50,30,L2gradient= True)
    kernel = np.ones((2,2), np.uint8) 
    edges = cv2.dilate(edges, kernel, iterations=1)
    result_d = cv2.bitwise_not(np.where(edges==[0,0,0], 255, edges))

    detect_yellow_blocks.yellow_holes(res_yellow, img_color)
    detect_red_blocks.red_holes(res_red, img_color)
    detect_blue_blocks.blue_holes(res_blue, img_color)
    detect_gray_blocks.gray_holes(res_gray, img_color) 
    res = detect_white_block.white_holes(res_white, img_color)
    kernel = np.ones((3,3), np.uint8) 
    res = cv2.dilate(res, kernel, iterations=1)
    res = cv2.erode(res, kernel, iterations=5)
    detect_groups(res)

    img_result = np.hstack((img_color, res))
    key = ord('a')
    while key != ord('q'):
        cv2.imshow('Result window', img_result)

        key = cv2.waitKey(30)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()