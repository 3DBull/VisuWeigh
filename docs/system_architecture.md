# System Architecture

---

The VisuWeigh system is designed to automatically collect data, clean data, train models, and evaluate models. 
If a model meets the criteria for production, it will automatically be pushed to the server where it can meet the 
demands of the cattle weighing user! 
All project tasks are monitored and scheduled with airflow. If there is a task failure, Airflow will send an email 
notification alert. 


![img.png](architecture.png)

---

## Data Collection

Data collection is performed from five cloud sources on a fixed schedule. 
The amount of raw data collected varies with each collection instance. 
Multiple instances of the collector can be run in parallel using multiple cores per instance. Multiple instances allow for collecting from multiple sources simultaneously. Multiple cores allow higher volumes of data to be processed without filling up large buffers. High processing power is required for the collector since it runs the YOLOV3 network on collected images for cow object detection. 

### Inputs
The data collector takes in a single argument which indicates the source of the data. 

### Outputs 
The collector outputs a json file that stores the records of each collected point. 
The json file contains the following record identifiers:</br>
[ lot, IMG_ID, Type, Shrink, Age, Avg_Weight, Tot_Weight, Hauled, Weaned, Feed, Health, Timestamp, prediction, auction] 


## Data Cleaning 
Various filters and algorithms are applied for the ETL(extract-transform-load) process. You can see this process in detail in the [data cleaning notebook.]()
The data cleaning process is executed on a fixed weekly schedule. 

### Input
The ETL task collects all the raw data files found in the database folder. [See Database Structure](#dbstruct) 

### Output
The output is a single json file with the cleaned database stored in it. The json data includes the columns: [timestamp, weight, auction, path] The file is stored in the training folder along with the cropped images prepared for the training network. 

## Machine Learning
Models are experimented with and developed. Once a model is ready to be trained on the data it is placed in the `cattle_data/models/input/` folder. Any model that is placed in this folder will be included in the regular training and evaluation process. Models are trained when the amount of data in the training folder changes by a significant amount (5000 points) since the last training. 

### Input
The input to the ML part of the system is the `inputs` folder in the database. Each model should be a `model_name.py` python file that returns a compiled keras model.

### Ouput
The output from the ML process is a folder that contains all the trained models: `cattle_data/models/output/` in the database.

## Evaluation
Once training is completed, the evaluation process is initiated. Each model that exists in the `models/output/` folder will be evaluated using the following metrics. 

1. Mean absolute error as an error metric: 

    $$ MAE =  \sum \limits _{i=0} ^{N} \frac {|y_i - \hat{y_i}|} {y} $$


2. Mean absoluted accuracy as an accuracy metric:

    $$ MAA = (1 -  \frac{MAE} {N}) * 100 \% $$

Each model is loaded from the database and evaluated with the above metrics on the data that exists in the `evaluation` folder in the database. 
After evaluating, the evaluation task tests the best model against the existing model in the `models/serve` directory. 
If the best model from evaluation has a lower $MAE$ and a higher $MAA$, the model will be pushed into the `serve` directory
to be used for deployment in the server. The previous serving model is moved into the `models/archive/` directory.

### Input 
The input to the evaluation process are the trained models stored in the database at `cattle_data/models/output/`. 

### Output
The output from the evaluation process is a dataframe with the scema: 


    Results.csv 
        root
            |-- name: (string) - The file name of the model
            |-- mean_abs_accuracy: (float) - The mean absolute accuracy for the model
            |-- mean_abs_error: (float) - The mean absolute error for the model
            |-- error_mean: (float) - The mean error for the model predictions
            |-- error_std: (float) - The standard deviation of the prediction error
            |-- error_min: (float) - The minimum prediction error
            |-- error_max: (float) - The maximum prediction error

The dataframe is stored as a csv at `cattle_data/evaluation/results.csv` in the database.

## Serving the Model
The model with the best evaluation metrics is compared with the current server model. 
If the model has a lower MAE and higher MAA than the currently serving model, the best model from evaluation 
is automatically pushed into the server. 

### Input
The server uses the model that is stored in the `models/serve` directory. 

### Output
The server provides an interface for users to measure the weight of cows from an uploaded image.
Check out the web app at [VisuWay.tech/weigh/](https://VisuWay.tech/weigh/)

## Database Structure <dbstruct name="dbstruct"/>

A file system architecture was chosen for the database to keep the data broadly accessible for any method of extraction. 

The following outline shows the database structure for the entire database:

    root/
        cattle_data/
            raw/
                {auction}_{date}_{extension}.json
                ...

               img/
                   {auction}/
                       im_{image#}.png
                       ...
        
            training/
               img/
                   {singles}
               training.json
            
            evaluation/
               img/
                   im_{image#}.png
               easy/
                   easy_eval_{date}.json
                   ...
               hard/
                   hard_eval_{date}.json
                   ...
               results.csv
            
            
            models/
               input/
                   {model_name}.py
                   ...
               output/
                   {model_name}_{epoch}_{val_loss} #keras model
                   ...
               serv/
                   {model_name}_{epoch}_{val_loss} #keras model

               training_logs/
                   {model_name}_{parameter_name}_{date}/
                       train/
                       validation/
                       parameters.json
                   ...



