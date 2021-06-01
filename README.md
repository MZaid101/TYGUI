# TYGUI

Train Yolov4 for your custom object detection using TYGUI on windows. 
## Requirements:
1. CUDA and cuDNN installed.
2. OpenCV built with CUDA
3. [Darknet](https://github.com/AlexeyAB/darknet) Configured for windows.
4. sklearn
5. Dataset Directory that contains all your images with labels

## Usage 
'''
import tkinter as tk
from tygui import TrainYolo
def main():
	root = tk.Tk()
	TrainYolo(root)
	root.mainloop()

if __name__ == "__main__":
	main()
'''

## GUI
![image](https://user-images.githubusercontent.com/53510596/120350802-bd731780-c318-11eb-8f0a-b563c07d65fc.png)

## Steps:
1. First Enter The Data Split Percentage and Select the Directory where you have your labelled data.
2. Then Enter Class Names which should be comma separated. > [class1, class2, class3, ..., classN]
3. Then Download [yolov4-custom.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-custom.cfg)
4. And enter batch, subdivisions and width/height as per your requirements.
5. Download Pre-Trained [Weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137)
6. Load pre-trained weights and darknet.exe
7. Start Training.
