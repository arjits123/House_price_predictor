import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import save_obj, model_training

#Importing machine learning libraries


from sklearn.linear_model import LinearRegression, Lasso, Ridge # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import r2_score # type: ignore


@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer."""
    model_trainer_file_path : str = os.path.join('artifacts', 'model_trainer.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Random_forest": RandomForestRegressor()
            }

            # Model training with evaluation
            model_report: dict = model_training(X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Best model score
            best_model_score = max(sorted(model_report.values()))

            #Best model name - gives the key of dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # best model - gives the value of dictionary
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            save_obj(
                file_path = self.model_trainer_config.model_trainer_file_path,
                obj = best_model
            )

            #Prediction of the best model
            ypred = best_model.predict(X_test)
            score = r2_score(y_test, ypred)
            return best_model_name, score

        except Exception as e:
            raise CustomException(e, sys)
