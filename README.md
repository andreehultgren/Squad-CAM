# Squad-CAM
A project based on further investigating the Grad-CAM explanation method.

![](results.jpg?raw=true)

## Table of contents
- [Squad-CAM](#squad-cam)
  - [Table of contents](#table-of-contents)
  - [General info](#general-info)
  - [Technologies](#technologies)
  - [Setup](#setup)

## General info
This project compares Squad-CAM with other explanation methods like Grad-CAM and guided backprop. The target is DCNN models like Resnet, VGG16 or AlexNet. This project uses VGG16 as a base for experiments. This requires all images to be of size 224x224. We opted not to resize images due to distortions.
	
## Technologies
Project is created with:
* Tensorflow version: 2.1.0
* Keras version: 2.3.1
## Setup
To run this project, set up a conda enviroment and run 
```cmd
$ pip install -r requirements.txt
```
Analyze a set of photos by defining the source and target folders 
```cmd
$ python Squad-CAM -source images/*.png -target output
```
