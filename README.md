# Disaster_Response

1. [Libraries](#libraries)
2. [Motivation](#motivation)
3. [General Descriptions](#generaldescriptions)
4. [File Descriptions](#filedescriptions)
5. [How To](#howto)
6. [Summary](#summary)

## Libraries <a name="libraries"></a>
+ pandas
+ numpy
+ sqlalchemy
+ sqlite3

+ re
+ sklearn

Using Python 3.9*

## Motivation <a name="motivation"></a>
The goal of this project is to develop an automatic data pipeline and a classification algorithm that clusters a message via API and Webapp. 
Since this is a lot of new functionality, I expect a strong learning curve from this project.

## General Descriptions <a name="generaldescriptions"></a>
The project exist of three major functionalites:

### 1. *ETL Pipeline*: 'process_data.py' file is a script that:
+ Loads the categories and messages datasets as csvs
+ Merges those two datasets into one bigger on
+ Process cleaning steps for machine learning readey data 
+ Stores this final dataset into a SQLite database

### 2. *ML Pipeline*: 'train_classifer.py' file is a scripte to start the ML pipeline that includes:
+ Loading the data from the SQLLite database
+ Split the loaded data into train and test sets
+ Uses a text processing and machine learning pipeline
+ Traines and imprives the resulting model via GridsearchCV
+ Outputs results on the test set and export it as a pickle for performance improvements

### 3. *Flask Web App*: a web application that returns the user for a givin input disaster message the category associated by the ML model.

## File Descriptions <a name="filedescriptions"></a>
The Project is structed as following:
+ README.md: read me file
+ ETL Pipeline Preparation.ipynb: Extract, contains the code for Transform, Load Pipline
+ ML Pipeline Preparation.ipynb: contains the code for Machine Learning training and classification model
   - workspace 
        - \app
            + run.py: file to run the application
        - \data
            + categories.csv: CSV file for categories datas
            + messages.csv: CSV file for messages data
            + DisasterResponse.db: Database file for the disaster response data
            + process_data.py: ETL process python script
        - \models
            + train_classifier.py: Classification python code 

## How To <a name="howto"></a>

 ### Data Python Scripts 
 1. Prepare the data with the following command:
        - Run the ETL Pipeline for data processing: 'python data/process_data.py data/dmessages.csv data/categories.csv data/DisasterResponse.db'
        - Run the ML Model for data anylsis and training: 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'

 ### Web App
To start the app run the following codes:
2. To start the App run the python script in the app's directory (/app) 'python run.py'
3. Go to the app screen {link: TBD}

## Summary <a name="summary"></a>
