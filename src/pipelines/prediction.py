import sys
import pandas as pd # type: ignore
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import load_obj

"""
Below class is responsible for mapping all the inputs we are giving in HTML to the backend with these below values
"""
class CustomData:
    def __init__(self, location:str, BHK:int, bath:int, total_sqft:int):
        self.location = location
        self.BHK = BHK
        self.bath = bath
        self.total_sqft = total_sqft
        
    def get_data_as_df(self):
        try:
            #Create a dictionary to create a df
            custom_data_dictionary = {
                'location' : [self.location],
                'BHK': [self.BHK],
                'bath': [self.bath],
                'total_sqft': [self.total_sqft]
            }
            df = pd.DataFrame(custom_data_dictionary)
            return df
        except Exception as e:
            raise CustomException(e,sys)


# Create the prediction pipeline

class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model_trainer.pkl'
            feature_engineering_path = 'artifacts/feature_engineering.pkl'
            model = load_obj(model_path)
            feature_engineering = load_obj(feature_engineering_path)
            tranformed_data = feature_engineering.transform(features)
            prediction = model.predict(tranformed_data)
            return prediction
        except:
            pass

