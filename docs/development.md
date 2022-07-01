# Development

The development follows an agile approach. Iterative development is performed for each module as well as for the entire 
pipeline. A level of effectiveness is desired before formalizing and documenting the code. 
Development roughly followed the following procedure: 

1. List the desired outcomes 
2. Experiment with the data and code in Jupyter notebooks 
3. Create prototype solutions that achieve the desired results
4. Revise the desired outcomes and the processes for achieving them
5. Reiterate the process

The most basic functional product was pushed to the public server as early as possible. After seeing results and revising
the desired outcomes, the process was reiterated to revise functionality, improve models, and solidify code into a more
robust system. 

### Contents
The project is divided into five major modules. The development of these modules will be discussed in more detail in the 
following sections. 

1. [Collect](#1.-collect) 
2. [Clean](#clean)
3. [Train](#3.-train)
4. [Evaluate](#evaluate)
5. [Deploy](#deploy)


## 1. Collect
After finding very limited data availability for this project, two methods were considered for data collection. 
1. Collect data from a farm with a scale and camera apparatus
2. Scrape relevant data from online auction sites  

While the first approach offered the opportunity for higher quality data, it would be expensive and time consuming to 
set up. In addition, many set ups would have to be put in place on many farms to get a variety of animal information. 
The second option was chosen even though the data is lower quality due to the availability, variety, and quantity of data. 

### Implementation of Web Scraper
Web scraping was started with “scrapy”, a python library that is common for scrawling websites and scraping data. 
Although, promising at first, scrapy proved too limiting in capability for the complexity of the JavaScript auction sites. 
I moved to a more powerful library called “selenium” which could handle the intricacies of digging into the site. 
This was effective in simulating human input to pass the verification as well as load and parse the JavaScript window.

### Building the Pipline
Once the collection was started, there were many challenges that arose in getting accurate information. 
The main challenge was from the timing of the information. ON the auction stream the cows are run through in batches, 
as they are allowed to enter, the information is placed on the website by someone on site. This manual entry of information 
means the data is often delayed with respect to the images. Sometimes the information would change several times before 
the correct information would be displayed. Sometimes the information for a cow or group of cows would not be displayed 
at all. A few strategies were attempted to avoid bad data from these inconsistencies. 
The first method was to record information only after a switch in weight, then stop recording after a certain length of 
time. This worked with a measure of accuracy, but bad data was still being collected. 
The second attempt included a cow detection for each collection point using the yolov3 neural network. Surprisingly, 
this made the data only slightly cleaner. 
On the third attempt, the cleaning of data was taken offline and correlated larger chunks of data at a time. 
This allowed for all the raw data to be collected for future utilization. Processing the network on every datapoint was 
extremely time-consuming. Some effort was extended to implementing multiprocessing for the offline job. The full
prediction of any cow data was added to the dataset in this stage. This prevented the need to run the predictor
on the dataset again at a later time. 
Once the processing time was decreased with utilizing multiple cores, the prediction process was placed back online in 
the collection script. Three workers were implemented for predicting and saving the data while the main script grabbed 
the data every second and placed it in a queue for the workers. With this optimization, I was even able to run two 
instances of the collection program at the same time pulling from two different auction locations simultaneously.  
Unfortunately, with no filtering on the data collection end, very high volumes of data must be collected and stored. 
The first side effect of collected a large amount of data was file size. All my structured data was stored in one file, 
which caused slower read/write access. The scraper program was then modified to create a new file for every unique day 
and auction location. After this optimization, the bottleneck was the hard drive. Using an external HDD for data storage 
created a read/write bottleneck especially when collecting from two auctions simultaneously. The queues would back up 
with data to be written. This is ok as long as the memory/paging file is a sufficient size, but not acceptable as the 
project scales. A hardware change to M.2 SSD to support the data volume needs.  

The auctions run in different locations on different days of the week. I was able to automate the process of collecting 
from these different locations at different times using the Windows Task Scheduler. After performing the multiprocessing 
optimizations, I was able to schedule more than one auction at a time.

## Clean
One of the most important aspects of machine learning is data cleaning. Careful consideration must be made for what makes
clean data and how the data can be cleaned automatically without loosing, or worse, corrupting portions of the data. 

### Fill Missing Data
On the first iteration of data collection, not all the information was collected. One example is the timestamp.
This vital piece of information was left out of the first two weeks of data. Rather than letting this data go to waste, 
a solution was developed into the cleaning procedure to fill any missing timestamps with the 'modified' property of the
corresponding image file. Another piece of data that is filled to update older data versions is the lot number. 
This identifies images that are of the same cow. To retroactively fill this data, the number is given according to the 
uniqueness of other data. 

### Fixing Numerical Data
Numerical data included commas in the thousands place, units that had to be removed from the string, and ranges that
had to be discarded. 

### Correcting the Data Timing
Much of the data comes from live data in which there is a severe lack of quality. 
One of the biggest challenges in cleaning this data is the timing of information. When the auction is live, each lot has
information that is taken from a queue and placed in the viewing screen manually by a human that is present when the cattle
are in view of the camera. Often times the cows are allowed into view before the data is switched. 
Sometimes the cows are let in after the data is switched. Occasionally, a lot's information did not get put in the 
queue and thus stays incorrect for the duration of the viewing for the cattle lot. Sometimes these lots will have the
information entered manually during the viewing. This means the information is significantly delayed compared to the 
image view.

![img_4.png](img_4.png)

The following criteria is used to define the most accurate information for a group of cows in the arena:
 1.	A group of points in a dataset is accepted as a valid lot if it contains at least 4 consecutive images with predicted cows. And there are at least two consecutive points with no cows between them.
 2.	The correct weight associated with the entire lot is the new weight after the last weight change during the time of the valid lot.
 3.	If the valid lot has no weight change for the duration of cows present in the image, and there is a weight change present before the valid lot (while there are no detected cows in image), then valid weight will be the weight at the start of the valid lot.
 4.	If there are multiple valid lots separated by only one image with no cows detected, combine the valid lots.

Based on the criteria, I was able to make an algorithm that adjusted the timing of the data to match the timing of the 
images. The following diagram shows the updated weight information (in red) In correct timing with the number of cows 
predicted in the image (yellow). This can be compared with the raw weight information (purple).

![img_6.png](img_6.png)

### Extracting Useful Data
It is difficult to obtain a weight estimate with multiple cows in an image. Also, to use single cows cropped out of a 
multi-cow image would only add variance to the data since the average weight of the group would be applied to the 
selected cow. To simplify the task, only the single cows lots were chosen for training on. This reduced the volume of 
usable data significantly, but it was a necessary cost for data quality. 

### Distribution of Validated Data
After getting a valid set of data, we need to look at the distribution to see where our data and therefore 
our model will be biased. The following shows a histogram of weight bins indicating the distribution across the 
validated data. 

![img_7.png](img_7.png)

The large amount of 0 weight data is single-cow data that was not able to be validated by the timing adjustment algorithm.
We get rid of this data as well as the limited data on the tails of the distribution to reduce bias in the model. 
The final training distribution is as follows:

![img_8.png](img_8.png)

## 3. Train
Training is implemented for automated model testing. The training script trains any model placed in the built models folder of the database.
Parameters for training are set in a .json configuration file. This allows for the fast iterative approach that is necessary for machine learning.

Regression is performed on the models with a single weight value in pounds as the output. The pound is used since this is the 
commonly used unit in the cattle industry.

For the loss function a commonly used mean squared error is used: $$ MSE = \frac {\sum \limits _{i=0} ^{N}  (y - \hat{y})^2} {N} $$ 

During training, the model parameters are embedded into  the name of the model. With this method, each version of the
model can easily be identified in tensorboard. 
For the evaluation process, these embeddings can be decoded and compared. 


### The Data
Training begins with the cleaned dataset and utilizes it for three main operations. 
1. Training
2. Validation 
3. Evaluation


Splitting the data into these three sets is an extremely important step in the training process. It is necessary that we use part of the dataset for 
validation that has not been seen by the network during training. In addition to this split, we perform another 
split to reserve a portion of the data for evaluation of our models once the training has completed. We want to evaluate how the 
model performs on new cows that go through the auction. Since the training and testing data is selected at random from the datset, 
and the data has multiple images of each cow, it is possible that some pictures of the same cow are in the training set as well 
as the testing set. To achieve a more reliable evaluation of our models, we will use data that has been recently added 
to the dataset. This way the entire evaluation set will be new cows.

### Scaling
The training process was developed as more images were being collected. As more data was collected, the process is repeated 
with the new data. An experiment was performed to view the trend in increased performance versus the amount of data collected.
The details and results of that experiment can be found in the [scaling notebook.](../notebooks/4_scaling.ipynb)
The scaling process was performed on the Xception network. Subsequent iterations of this experiment should be performed 
to compare the effects of data amount on a CNN versus a Transformer. 

### Models
Most of the models used are common models initialized with pre-trained weights. The transfer learning approach is used to 
adapt these pre-trained models to the cattle weight dataset. The idea is that some knowledge obtained by those models 
in other domains can be utilized in this domain.
Multiple iterations of the training process were performed for the following models:
1. Xception 
2. Inception
3. InceptionRes
4. Resnet152
5. A custom CNN
6. Vgg19

Other models attempted in earlier iterations but did not have promising results:
1. A Custom CNN 
2. Resnet50
3. A Vision Transformer 

For more information on the training process and results, see the training [notebooks](../notebooks).

### Visualization
To better understand the training of the models and to visualize the attention of the models, the [keract](https://pypi.org/project/keract/)
library was used to look at the activations of the layers. Below are some examples taken from the training of the Xception network.



It is not easy to derive meaning or effectiveness from these activations. Some research [[3]](../README.md#references-and-related-works) 
has shown the chest width measurement to be the trait most related to the body weight. With "correct" training of the model, 
it would be expected that there would be more activation in the layers around the chest area. While this expectation is occasionally met, 
the more commonly observed trend is segmentation of the animal in early activation layers with a focus on the back in the
later layers. 

![img_1.png](img_1.png)
![img_3.png](img_3.png)

One notable observation is that in several of the images the heatmap has the cow’s shadow highlighted. 
It seems like it might be using the shadow of the cow for added perspective on the width of the animal! 

![img.png](img.png)


## Evaluate

### Metrics
The two main evaluation components that are evaluated are:

 1. Mean absolute error as an error metric: 
           
    $$ MAE =  \frac {\sum \limits _{i=0} ^{N} |y_i - \hat{y_i}|} {N} $$

 2. Mean absolute accuracy percentage as an accuracy metric:

    $$ MAAP = (1 -  {\sum \limits _{i=0} ^{N} \frac {|y_i - \hat{y_i}|} {\hat{y_i}}}) * 100 \% $$

MAE was chosen to give a fair representation by how many pounds the model misrepresented the image. This metric is often
used in machine learning and also has meaningful information for someone in the cattle industry.  
The accuracy metric gives a comparative value for anyone in any domain to easily recognize. The normalization in the
accuracy metric also provides us with an understanding of the relativity with which a model predicts. In essence, it is 
more sensitive toward lighter animals of which the accuracy is of greater importance. This is especially true as it relates 
to the health of the animal. 

Other metrics were added for extra insights: 
1. Mean Error  
    This metric is included to observe the bias of the model. There is a greater amount of training data in the 500-700lb 
range. If a model is trained poorly, it could be "guessing" low on higher weight cows. This metric will catch that bias.  


2. Error STD

Helps us understand the distribution of the error. 
    
3. Minimum Error
4. Maximum Error
5. Maximum Absolute Error

These last three metrics are included to show us the extremes of the error for reliability purposes. 

### Results
The following table shows the evaluation results of models trained on 65,000 images.  

![img_9.png](img_9.png)

## Deploy
The [streamlit](https://streamlit.io/) library was chosen for a deployment app due to the intuitive
nature of the interface and compatibility with machine learning/data visualization.
Not only is the interface convenient for showcasing the project, but it's also useful for 
testing models and visualising data locally for iterative development. 

The web app was built in two with the following two main components: 

1. The Prediction Page was built to showcase the model and to retrieve extra data from users.
2. The Data Page was built to allow users to interact with the models and data.
