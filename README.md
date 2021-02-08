# Disaster Response Pipeline Project

## Table of contents

1. [Project Description](#project-description)
2. [Example2](#example2)
3. [Third Example](#third-example)
4. [Fourth Example](#fourth-examplehttpwwwfourthexamplecom)


## Project Description

This project is part of UDACITY Data Science Nanodegree Program. The dataset of this project contains aballed messasges and it is provided by Figure Eight company.It this project the main goal is to build NLP machine learning model which will be used for the future message classification.

The project was divided into 3 separate sections:
 - Data preparation step. ETL pipeline was used to prepare data for the next step.
 - Build ML model for message classification
 - Run a web app which can show model results in real time



## Example2
## Third Example
## Fourth Example (http://www.fourthexample.com) 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
