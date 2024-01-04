import pandas as pd
import joblib
import os

class Inference:
    def __init__(self, data, models_dir):
        self.data = data
        print(f"Data shape: {self.data.shape}")

        # Load the trained models
        self.stress_score_model = joblib.load(os.path.join(models_dir, 'stress_score_model.pkl'))
        self.moral_model = joblib.load(os.path.join(models_dir, 'moral_model.pkl'))
        self.completion_time_model = joblib.load(os.path.join(models_dir, 'completion_time_model.pkl'))

    def data_preprocessing(self, name):
        # Label Encoding for Categorical Features
        df = self.data[self.data['Employee Name'] == name].copy()

        categorical_columns = ['Gender', 'Married', 'Role', 'Position', 'Moral', 'Project_Difficulty']
        for col in categorical_columns:
            df[col] = pd.factorize(df[col])[0]

        # Removing unnecessary columns
        cols = ['Employee Name', 'Joining_Year','ID','Project_Description']
        df = df.drop(columns=cols)

        # Date Processing
        try:
          df['Project_Start_Date'] = pd.to_datetime(df['Project_Start_Date'], format='%d/%m/%y')
          df['Project_Deadline'] = pd.to_datetime(df['Project_Deadline'], format='%d/%m/%y')
        except:
          df['Project_Start_Date'] = pd.to_datetime(df['Project_Start_Date'], format='%d/%m/%Y',errors='coerce')
          df['Project_Deadline'] = pd.to_datetime(df['Project_Deadline'], format='%d/%m/%Y',errors='coerce')

        df['Time_Allotted'] = (df['Project_Deadline'] - df['Project_Start_Date']).dt.days

        # Dropping date columns
        cols = ['Project_Start_Date','Project_Deadline']
        df = df.drop(columns=cols)

        # Salary Conversion
        df['Salary'] = df['Salary'].str.replace(',', '').astype(int)

        # Standard Scaler for Numerical Features
        numerical_features = ['Age', 'Salary','Mean Monthly Hours', 'Absences',
                       'Ongoing_Project_Count','Projects_Within_Deadline','Projects_Completed','Completion_Time']

        categorical_features = ['Gender', 'Current_Employ_Rating','Married', 'Role','Position', 'Moral', 'Project_Difficulty','Manager_ID']

        features = numerical_features + categorical_features

        scaler = joblib.load('scaler.pkl')
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        # Removing highly correlated features
        highly_correlated_features = ['Years in the company']
        df = df.drop(highly_correlated_features, axis=1)

        return df

    def stress_score_preprocessing(self, name):
        # Call data preprocessing method
        df = self.data_preprocessing(name)

        # Prepare the target and features for stress score prediction
        target_variable = 'Stress & Burnout Score'
        X = df.drop(columns=[target_variable])

        return X

    def moral_data_preprocessing(self, name):
        # Call general preprocessing
        df = self.data_preprocessing(name)

        # Preprocessing steps specific to the 'Moral' predictio
        target_variable = 'Moral'
        X = df.drop(columns=[target_variable])

        return X

    def completion_time_preprocessor(self, name):

        df = self.data[self.data['Employee Name'] == name].copy()

        cols = ['Employee Name', 'Joining_Year','ID']
        df = df.drop(columns=cols)

        try:
          df['Project_Start_Date'] = pd.to_datetime(df['Project_Start_Date'], format='%d/%m/%y')
          df['Project_Deadline'] = pd.to_datetime(df['Project_Deadline'], format='%d/%m/%y')
        except:
          df['Project_Start_Date'] = pd.to_datetime(df['Project_Start_Date'], format='%d/%m/%Y',errors='coerce')
          df['Project_Deadline'] = pd.to_datetime(df['Project_Deadline'], format='%d/%m/%Y',errors='coerce')

        df['Time_Allotted'] = (df['Project_Deadline'] - df['Project_Start_Date']).dt.days
        df['Time_Allotted'] = df['Time_Allotted'].abs()

        cols = ['Project_Start_Date','Project_Deadline']
        df = df.drop(columns=cols)

        categorical_columns = ['Gender', 'Married', 'Role', 'Position', 'Moral', 'Project_Difficulty']
        for col in categorical_columns:
            df[col] = pd.factorize(df[col])[0]

        df['Salary'] = df['Salary'].str.replace(',', '').astype(int)

        numerical_features = ['Age', 'Salary','Mean Monthly Hours', 'Absences',
                       'Ongoing_Project_Count','Projects_Within_Deadline','Projects_Completed']

        categorical_features = ['Gender', 'Current_Employ_Rating', 'Role','Position', 'Moral', 'Stress & Burnout Score','Project_Difficulty','Manager_ID']

        text_features = ['Project_Description']

        target_variable = ['Completion_Time']

        scaler = joblib.load('scaler.pkl')
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

        features = numerical_features + categorical_features + text_features
        target = df['Completion_Time']
        df = df.drop(columns=['Completion_Time','Project_Description'])

        highly_correlated_features = ['Years in the company']
        df = df.drop(highly_correlated_features, axis=1)

        cols = ['Age','Salary','Mean Monthly Hours','Absences','Ongoing_Project_Count','Projects_Within_Deadline','Current_Employ_Rating', 'Role','Position','Moral','Stress & Burnout Score','Project_Difficulty']
        X = df

        return X

    def predict_stress_score(self, name):
        # Prepare the data in the format expected by the stress score model
        # This involves selecting the appropriate columns, etc.
        X_stress = self.stress_score_preprocessing(name)
        predictions = self.stress_score_model.predict(X_stress)
        return predictions

    def predict_moral(self, name):
        # Prepare the data for the moral model
        X_moral = self.moral_data_preprocessing(name) # Modify as per the model's requirements
        predictions = self.moral_model.predict(X_moral)
        return predictions

    def predict_completion_time(self, name):
        # Prepare the data for the completion time model
        X_completion = self.completion_time_preprocessor(name) # Modify as per the model's requirements
        predictions = self.completion_time_model.predict(X_completion)
        return predictions

