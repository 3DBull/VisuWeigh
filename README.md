# The  VisuWeigh Project

Project to develop a data pipeline and machine learning models to estimate cattle weight based on visual information. 

![image](deployment/assets/demo.gif)

## Quickstart 

For am interactive preview of the project in action visit [VisuWay.tech](https://visuway.tech).
You'll find a web app where you can try weighing cows of your own! You can also use the 
interactive data page to see how some of the top models perform. 

## Contents

The project is divided into five development modules. Each of the modules have their own explanations 
of their development and how they are used. Use the links below to learn about a specific module. 

1. [Collect](docs/development.md#collect) 
2. [Clean](docs/development.md#clean)
3. [Train](docs/development.md#train)
4. [Evaluate](docs/development.md#evaluate)
5. [Deploy](docs/development.md#deploy)


## Overview

### Introduction 
 The cattle industry relies heavily on weight information for sales and marketing as well as animal health. The traditional method of weighing cattle is with a mechanical scale that cattle walk onto. These scales are expensive to install and maintain and therefore many smaller cattle producers do not have these scales installed in their facilities. 
 An expert cattle buyer can estimate the weight of a cow by visually looking at the cow. If this visual estimation could be performed with computer vision, cattle producers can quickly and automatically track the weight of their animals without having to transport them to a scale or consult an expert. 
 
### Related Work
Companies like [agroninja](https://agroninja.com/) have developed an app for cattle weighing from images. agronija's app uses 
an image of the cow and prompt the user to input points onto the image in key locations on the cow. These apps are 
cumbersome to use and impractical to use at scale. It works to get a 4H cow to pose for a picture, but if a producer, or market
wants to obtain regular weights on a large scale farm a larger scale solution is needed. One such solution is the 
[ClickR](https://clicrweight.com/) solution. They have 

Other research has show that estimating the weight of cattle from images can be effective. 
M. Gjergji et al. [[1]](#references-and-related-works) were able to achieve MAE of 23.19kg with convolutional neural 
networks and Weber et al. [[3]](#references-and-related-works) achieved an MAE of 13.44kg using an active contour model. 
Given such promising research, it is only a matter of time before this technology becomes adopted into the industry. 
The main gap that seems to be holding the technology back is the lack of data. This project seeks to minimize that gap. 

### Objective
The purpose of this project is to provide a reliable method of cattle weighing that reduces the stress on animals, 
reduces the cost and time of weighing, and increases the frequency at which livestock can be weighed. The focus of this 
project is on achieving results on real-world data. Weighing cows in a natural setting, walking, running or turning, from any
angle, is what provides value to production oriented system. 
This is accomplished by using ordinary 2D images of cattle to estimate weight accurately in a natural environment. 
Deep learning models are employed for regression on image data. 

### Methods
The key to a good prediction CNN is good data. The focus of this project is to retrieve and clean large amounts of data in an automated fashon to train a CNN on predicting cattle weight.

### System Architecture
 The VisuWeigh project is designed to handle data, models, and evaluations automatically. For more information, read the [System Architecture documentation.](https://github.com/3DBull/VisuWeigh/blob/main/docs/system_architecture.md)

### To Do:
- [x] Create Raw Collection Pipline 
- [x] Create Data Cleaning Pipline
- [x] Train on CNN Model
- [x] Compare Transformer and CNN Models
- [x] Compare Single and Multi-Angle Prediction
- [x] Scale up Data with Automated Collection
- [x] Build Server GUI
- [ ] Integrate mongo DB to accompany system scaling
- [ ] Automate Cleaning and Training
- [ ] Automate Evaluation and Deployment


### Models in Action

![image](https://user-images.githubusercontent.com/28244647/156103607-91e49917-0ef2-49a9-a644-7e792b2a2cdb.png)

These models vary in performance with each image. Overall the Xception model achieves the highest performance on new data. On the first image where the image is distorted, none of the models perform well. This is a good sign that the models are fitting the data in the expected manner. 


### References and Related Works 
1. M. Gjergji et al., "Deep Learning Techniques for Beef Cattle Body Weight Prediction," 2020 International Joint Conference on Neural Networks (IJCNN), 2020, pp. 1-8, doi: 10.1109/IJCNN48605.2020.9207624.

2. Vanessa Aparecida Moraes Weber et al. Cattle weight estimation using active contour models and regression trees Bagging, Computers and Electronics in Agriculture, Volume 179, 2020, 105804, ISSN 0168-1699, https://doi.org/10.1016/j.compag.2020.105804. (https://www.sciencedirect.com/science/article/pii/S016816992031783X)

3. Rudenko, Oleg et al. “Cattle breed identification and live weight evaluation on the basis of machine learning and computer vision.” CMIS (2020).

4. Ozkaya, Serkan & Bozkurt, Yalcin. (2009). The accuracy of prediction of body weight from body measurements in beef cattle. Archiv fur Tierzucht. 52. 10.5194/aab-52-371-2009.

5. R. A. Gomes, G. R. Monteiro, G. J. F. Assis, K. C. Busato, M. M. Ladeira, M. L. Chizzotti, Technical note: Estimating body weight and body composition of beef cattle trough digital image analysis, Journal of Animal Science, Volume 94, Issue 12, December 2016, Pages 5414–5422, https://doi.org/10.2527/jas.2016-0797