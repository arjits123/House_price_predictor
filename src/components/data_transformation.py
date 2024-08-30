import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass
from utils import convert_total_sq, outlier_remove, TrainTestSplit, save_obj

# ML libraries
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.compose import make_column_transformer # type: ignore
from sklearn.pipeline import make_pipeline # type: ignore
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder # type: ignore

#Data transformation configuration class
@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation."""
    # Handel missing values, outliers, scale the variables and encode (OneEncoder, StandardScaler, MinMaxscaler)
    feature_engineering_obj_path : str = os.path.join('artifacts', 'feature_engineering.pkl')  
    cleaned_data_obj_path : str = os.path.join('artifacts', 'cleaned_data.csv')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def clean_data(self, data):
        # reading the data
        df = pd.read_csv(data)

        df = df.drop(columns=['area_type', 'availability', 'society', 'balcony'])

        # Handling null values
        df['location'] = df['location'].fillna('Whitefield')
        df['bath'] = df['bath'].fillna(df['bath'].median())
        df['size'] = df['size'].fillna('2 BHK')
        
        # Converting size (object) into BHK (int)
        df['BHK'] = df['size'].str.split(' ').str.get(0).astype(int)
        
        # cleaing total_sqft
        df['total_sqft'] = df['total_sqft'].apply(convert_total_sq)
        
        # handling location column
        df['location'] = df['location'].apply(lambda x : x.strip())
        location_count = df['location'].value_counts()
        location_less_10 = location_count[location_count <= 10]
        df['location'] = df['location'].apply(lambda x : 'other' if x in location_less_10 else x )
        
        df = df.drop(columns=['size'])
        logging.info('Data cleaning completed')
        
        return df
    
    def remove_out(self,new_df):
    
        new_df['Price_per_sq_feet'] = new_df['price']*100000/new_df['total_sqft']
        
        #remove outliers for total_sq_feet
        new_df = new_df[((new_df['total_sqft']/new_df['BHK']) >=300)]

        #removing other outliers from utils
        new_df = outlier_remove(new_df)
        
        #removing outliers for BHK
        Q1 = new_df["BHK"].quantile(0.25)
        Q3 = new_df["BHK"].quantile(0.75)
        IQR = Q3 - Q1
        lower_threshold = Q1 - 1.5 * IQR
        upper_threshold = Q3 + 1.5 * IQR
        new_df["is_outlier"] = (new_df["BHK"] < lower_threshold) | (new_df["BHK"] > upper_threshold)
        new_df = new_df[~new_df['is_outlier']]

        new_df = new_df.drop(columns=['Price_per_sq_feet','is_outlier'])

        logging.info('Removal of outliers completed')

        #saving cleaned csv file
        new_df.to_csv(self.transformation_config.cleaned_data_obj_path, index=False, header=True)

        return new_df

    def get_data_transformation_object(self):
        try:

            transformer = make_column_transformer((OneHotEncoder(sparse_output=False), ['location']), remainder='passthrough')
            scaler = MinMaxScaler()

            preprocessor_pipeline = make_pipeline(transformer, scaler)
            return preprocessor_pipeline
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, final_data):
        try:
            df = final_data
            
            #split the dataset into X and y variables
            X = df.drop(columns = ['price'], axis = 1)
            y = df['price']
            logging.info('splitting data set into X and y completed')
            
            #train test split
            X_train, X_test, y_train, y_test = TrainTestSplit(
                predictor = X,
                target = y
            )
            logging.info('train test split completed')

            # obtaining preprocessor object
            preprocessor_obj = self.get_data_transformation_object()
            X_train = preprocessor_obj.fit_transform(X_train)
            X_test = preprocessor_obj.transform(X_test)
            logging.info('Data transformation/ feature engineering completed')

            save_obj(
                file_path = self.transformation_config.feature_engineering_obj_path,
                obj = preprocessor_obj
            )
            logging.info('feature_engieering object file saved')

            return(
                X_train,
                X_test,
                y_train,
                y_test
            )
        
        except Exception as e:
            raise CustomException(e,sys)




