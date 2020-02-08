# vision_system_lego_detection
This repository contains project on Vision Systems Subject - detecting and counting holes in group of lego blocks

## Description

This script count holes in Lego blocks, based on groups of blocks in colors descripted in file [public.json](https://github.com/m-milena/vision_system_lego_detection/blob/master/files/public.json). To run program you have to specify as arguments:

  - path to imgs folder,
  - path to file with image description (public.json file),
  - path to not created json file to save results (in example result.json file).

Example of command:
```sh
python main.py [path]\imgs\ [path]\files\public.json  [path]\files\result.json
```

To images in \imgs folder, there are only 2 mistakes. Method used to solve this problem works only for this dataset. 
