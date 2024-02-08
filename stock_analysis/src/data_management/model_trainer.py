import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception_handler import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path  = os.path.join("model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiae_model_trainer(self, train_array , test_array):
        try:
            logging.info("Split training and testinput data")
            print("")
        except Exception as e:
            raise CustomException(e,sys)


