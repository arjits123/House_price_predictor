import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def convert_total_sq(x):
    y = x.split('-')
    
    if len(y) == 2:
        
        return (float(y[0]) + float(y[1]))/2
    try:
        return float(x)
    except:
        return None

def outlier_remove(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):

        mean = np.mean(subdf.Price_per_sq_feet) # calculated mean

        sd = np.std(subdf.Price_per_sq_feet) # calculated standard deviation

        #outlier detection and removal, 1 std idhar udhar ke rakhe hain data
        general_df = subdf[(subdf.Price_per_sq_feet > (mean-sd)) & (subdf.Price_per_sq_feet <= (mean + sd))]
    
        df_output = pd.concat([df_output, general_df], ignore_index=True)
        
    return df_output

def TrainTestSplit(predictor,target):
    try:
        X_train, X_test, y_train, y_test = train_test_split(predictor,target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e,sys)

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        pass

def model_training(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            #list of all the models
            model = list(models.values())[i]

            #train the model
            model.fit(X_train,y_train)
            #make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #evaluate the model
            training_score = r2_score(y_train, y_train_pred)
            testing_score = r2_score(y_test, y_test_pred)
            
            #report[keys] = value
            report[list(models.keys())[i]] = testing_score

            return report
    
    except Exception as e:
        raise CustomException(e,sys)
