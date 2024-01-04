import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import math
import joblib
import os

class Trainer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def train_stress_score(self, X_train, X_test, y_train, y_test):

        # XGBoost Classifier for Stress Score
        xgb_model = XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train-1)

        y_pred_xgb = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test-1, y_pred_xgb)
        print(f"Stress Score Prediction Accuracy: {accuracy:.2f}")

        # Save the model
        model_path = os.path.join(self.output_dir, 'stress_score_model.pkl')
        joblib.dump(xgb_model, model_path)

    def train_moral(self, X_train, X_test, y_train, y_test):
        
        # XGBoost Classifier for Moral
        xgb_model = XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train)

        y_pred_xgb = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_xgb)
        print(f"Moral Prediction Accuracy: {accuracy:.2f}")

        # Save the model
        model_path = os.path.join(self.output_dir, 'moral_model.pkl')
        joblib.dump(xgb_model, model_path)

    def train_completion_time(self, X_train, X_test, y_train, y_test):
        
        # XGBoost Regressor for Completion Time
        xgb_model = XGBRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        print(f"Completion Time Prediction RMSE: {rmse}")

        # Save the model
        model_path = os.path.join(self.output_dir, 'completion_time_model.pkl')
        joblib.dump(xgb_model, model_path)
