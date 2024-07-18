# Waterside Pollution Dataset
As title.

## Something to say
***All data is collected by Chihjung Chien.***

* [Origin](https://github.com/Jung217/Harbor_Pollution_Dataset/tree/main/Origin) : origin image & hand-made masks
* [via-2.0.12](https://github.com/Jung217/Harbor_Pollution_Dataset/tree/main/via-2.0.12) : a tool uses to draw masks
* [yoloV9Od.ipynb](https://github.com/Jung217/Harbor_Pollution_Dataset/blob/main/yoloV9Od.ipynb) : a yolov9 example using this dataset training model
* [pollute.v2i.yolov9](https://github.com/Jung217/Harbor_Pollution_Dataset/tree/main/pollute.v2i.yolov9) : dataset split on [roboflow](https://roboflow.com/)

## results
* test
<img src="pic/test.jpg" width=50% height=50%>

* new to model
<img src="pic/test (1).png" width=50% height=50%>

## P.s.
My laptop graphics card was too suck to train model.
```ps
PS C:\Users\alex2\Desktop\NTUST\Harbor_Pollution_Dataset> python .\testCUDA.py
PyTorch 版本: 2.0.1+cu118
CUDA 可用: True
CUDA 版本: 11.8
GPU 數量: 1
CUDA:0 (NVIDIA GeForce GTX 1650, 4095 MB)
```
