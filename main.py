import os
import sys
import cv2
import json
import math
import numpy as np
from src.LegoDetection import LegoDetection

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

        for img in imgs_names:
            print('Current processed image: ', img)
            current_img = cv2.imread(imgs_path+'/'+img, cv2.IMREAD_COLOR)
            legodetection = LegoDetection(current_img, 0.15)
            info_array = legodetection.count_holes()
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
                    if error_table == []:
                        break
                    index = min(range(len(error_table)), key=error_table.__getitem__)
                    found_holes[i] = holes_table[index]
                    info_array.pop(index)

            data_to_write[img[:-4]] = found_holes
            with open(output_path, 'w+') as json_file:
                json_string = json.dumps(data_to_write)
                json_file.write(json_string)
            
        print(data_to_write)

if __name__ == '__main__':
    main()