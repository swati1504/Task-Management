import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

class DataPreProcessor:
    def __init__(self, data):
        self.data = data
        print(f"Data shape: {self.data.shape}")

    def data_preprocessing(self):
        # Label Encoding for Categorical Features
        df = self.data.copy()
        df.drop_duplicates(inplace=True)
        df.fillna(df.median(), inplace=True)

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

        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        joblib.dump(scaler, 'scaler.pkl')
        # Removing highly correlated features
        correlation_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix')
        plt.show()

        correlation_threshold = 0.8

        highly_correlated_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    colname = correlation_matrix.columns[i]
                    highly_correlated_features.add(colname)

        df = df.drop(highly_correlated_features, axis=1)

        return df

    def stress_score_preprocessing(self):
        # Call data preprocessing method
        df = self.data_preprocessing()
        # Prepare the target and features for stress score prediction
        target_variable = 'Stress & Burnout Score'
        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        return X_train, X_test, y_train, y_test

    def moral_data_preprocessing(self):
        # Call general preprocessing
        df = self.data_preprocessing()

        # Preprocessing steps specific to the 'Moral' predictio
        target_variable = 'Moral'
        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        return X_train, X_test, y_train, y_test

    def completion_time_preprocessor(self):
        df = self.data.copy()

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

        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

        features = numerical_features + categorical_features + text_features
        target = df['Completion_Time']
        df = df.drop(columns=['Completion_Time','Project_Description'])

        correlation_matrix = df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix')
        plt.show()

        correlation_threshold = 0.8

        highly_correlated_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    colname = correlation_matrix.columns[i]
                    highly_correlated_features.add(colname)

        df = df.drop(highly_correlated_features, axis=1)
        print(f"highly_correlated_features = {highly_correlated_features}")
        cols = ['Age','Salary','Mean Monthly Hours','Absences','Ongoing_Project_Count','Projects_Within_Deadline','Current_Employ_Rating', 'Role','Position','Moral','Stress & Burnout Score','Project_Difficulty']
        X = df
        y = target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        return X_train, X_test, y_train, y_test
