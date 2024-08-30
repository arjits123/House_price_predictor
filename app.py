# Importing the necessary libraries
from flask import Flask, render_template, request # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import sys
from sklearn.preprocessing import StandardScaler # type: ignore
from src.pipelines.prediction import CustomData, PredictionPipeline

#Creating flask application
application = Flask(__name__)
app = application

# load the dataset
house = pd.read_csv('artifacts/cleaned_data.csv')

# Calling the main page
@app.route('/')
def index():
    location = sorted(house['location'].unique())
    return render_template('index.html', locations = location)

@app.route('/', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Get the data
        data = CustomData(
            location = request.form.get('location'),
            BHK = request.form.get('BHK'),
            bath=request.form.get('bath'),
            total_sqft=request.form.get('total_sqft')
        )
        
        # Create the data frame
        prediction_df = data.get_data_as_df()
        print(prediction_df)

        #Initialise the pipeline
        prediction_pipeline = PredictionPipeline()
        results = prediction_pipeline.predict(prediction_df)
        print(results)
        return render_template('index.html', result = np.round(results[0],5))
    
if __name__ =='__main__':
    app.run(debug = True, port = 5001)