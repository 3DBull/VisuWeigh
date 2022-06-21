# The  VisuWeigh Project

Project to develop a data pipeline and machine learning models to estimate cattle weight based on visual information. 

## Quickstart 

### Web App
For am interactive preview of the project in action visit [VisuWay.tech](https://visuway.tech).

### Running the Code Locally

The project is divided into five modules. Each of the modules have their own
set of instructions. 

1. [Collect](#collect) 
2. [Clean](#clean)
3. [Train](#train)
4. [Evaluate](#evaluate)
5. [Deploy](#deploy)

---

## Introduction 
 The cattle industry relies heavily on weight information for sales and marketing as well as animal health. The traditional method of weighing cattle is with a mechanical scale that cattle walk onto. These scales are expensive to install and maintain and therefore many smaller cattle producers do not have these scales installed in their facilities. 
 An expert cattle buyer can estimate the weight of a cow by visually looking at the cow. If this visual estimation could be performed with computer vision, cattle producers can quickly and automatically track the weight of their animals without having to transport them to a scale or consult an expert. 
 
## Objective
The purpose of this project is to provide a reliable method of cattle weighing that reduces the stress on animals, reduces the cost and time of weighing, and increases the frequency at which livestock can be weighed. This will be accomplished by using common 2D images of cattle to estimate weight accurately in a natural environment. Deep learning models will be employed for regression on image data. 

## Methods
 The key to a good prediction CNN is good data. The focus of this project is to retrieve and clean large amounts of data in an automated fashon to train a CNN on predicting cattle weight. 

## System Architecture
 The VisuWeigh system is designed to handle data, models, and evaluations automatically. For more information, read the [System Architecture documentation.](https://github.com/3DBull/VisuWeigh/blob/main/docs/system_architecture.md) 

## Development 
...

## To Do:
- [x] Create Raw Collection Pipline 
- [x] Create Data Cleaning Pipline
- [x] Train on CNN Model
- [x] Compare Transformer and CNN Models
- [x] Compare Single and Multi-Angle Prediction
- [x] Scale up Data with Automated Collection
- [x] Build Server GUI
- [ ] Automate Cleaning and Training
- [ ] Automate Evaluation and Deployment


## Models in Action
![pdf] (https://drive.google.com/open?id=1oU7_ls9v-QRjiqVqDR_KMKvcpPorGvRY)
![image](https://user-images.githubusercontent.com/28244647/156103607-91e49917-0ef2-49a9-a644-7e792b2a2cdb.png)

These models vary in performance with each image. Overall the Xception model achieves the highest performance on new data. On the first image where the image is distorted, none of the models perform well. This is a good sign that the models are fitting the data in the expected manner. 
