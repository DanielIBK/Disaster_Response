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


## How To <a name="howto"></a>

    ### Python Scripts 

    ### Web App

## Summary <a name="summary"></a>
