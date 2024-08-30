import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass

from data_transformation import DataTransformation
from model_development import ModelTrainer

import pandas as pd # type: ignore
import numpy as np # type: ignore

#Create the Data ingestion config class
@dataclass
class DataIngestionConfig:
    """This class contains configuration for data ingestion."""
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    logging.info('Data ingestion configuration completed')

# Class to ingest the data
class DataIngestion:
    """This class is used for ingesting the data from the source."""
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    # initiate the data ingestion

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion component')

        try:
            df = pd.read_csv('notebook/data/Bengaluru_House_Data.csv')
            logging.info('Imported the dataset from the local drive')

            #making the artifacts directory
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            #Saving the raw csv file
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Data ingestion completed')
            return self.data_ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    
    # Data Ingestion
    data_ingestion = DataIngestion()
    raw_data_path = data_ingestion.initiate_data_ingestion()

    #Data Transformation
    data_transformation = DataTransformation()
    cleaned_df = data_transformation.clean_data(data = raw_data_path)
    final_df = data_transformation.remove_out(new_df = cleaned_df)
    X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(final_data = final_df)

    #Model Development 
    model_development = ModelTrainer()
    print(model_development.initiate_model_trainer(X_train = X_train, X_test = X_test, y_train=y_train, y_test=y_test))
    





