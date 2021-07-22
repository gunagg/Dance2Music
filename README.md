# DanceToMusic
​
## [Paper link](https://arxiv.org/abs/2107.06252).  
This repo requires OpenPose for running on GPU and OpenVINO for running on CPU.  
To install OpenPose follow the instructions given [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source). Make sure to enable the `BUILD_PYTHON` flag while installing.    
To install OpenVINO follow the instructions given [here](https://docs.openvinotoolkit.org/latest/installation_guides.html).  
## Live OpenPose 
Run this on a GPU machine -   
```
python Dance2Music_openpose.py
```
​
## Run on an existing video using OpenPose 
Run this on a GPU machine -   
```
python Dance2Music_openpose.py --input video_fname
```
​
## Live OpenVINO - 
Can be run on a CPU -   
```
python Dance2Music_openvino.py
```
​
## Run on an existing video using OpenVINO - 
Can be run on a CPU -   
```
python Dance2Music_openvino.py --input video_fname
```
