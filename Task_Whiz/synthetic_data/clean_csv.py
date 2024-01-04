import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Sample data
data = pd.read_csv("synthetic_data.csv")
df = pd.DataFrame(data)

# Create a dictionary to store unique IDs for each name
name_id_map = {}
unique_id = 0

# Assign unique IDs to each name
for index, row in df.iterrows():
    name = row['Employee Name']
    if name not in name_id_map:
        unique_id += 1
        name_id_map[name] = unique_id
    df.loc[index, 'ID'] = name_id_map[name]

df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Print the updated dataframe
print(df["Employee Name"].nunique())

df["Project_Difficulty"] = df["Project_Difficulty"].replace(
    ["Very High", "Extremly High"], "High", regex=True
)

df.to_csv('clean_synthetic_data.csv', index=False) # Replace "output.csv" with your desired filename

import pandas as pd

# Sample data
data = pd.read_csv("synthetic_data.csv")
df = pd.DataFrame(data)

# Create a dictionary to store unique IDs for each name
name_id_map = {}
unique_id = 0

# Assign unique IDs to each name
for index, row in df.iterrows():
    name = row['Employee Name']
    if name not in name_id_map:
        unique_id += 1
        name_id_map[name] = unique_id
    df.loc[index, 'ID'] = name_id_map[name]

df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Print the updated dataframe
print(df["Employee Name"].nunique())

df["Project_Difficulty"] = df["Project_Difficulty"].replace(
    ["Very High", "Extremly High"], "High", regex=True
)


def date_to_days(date):
    date_list = date.split('/')
    if(len(date_list[2])) == 2:
        date_value = (int(date_list[1])-1)*30+(int(date_list[2])-21)*365 + int(date_list[0])
    else:
        date_value = (int(date_list[1])-1)*30+(int(date_list[2])-2021)*365 + int(date_list[0])
    return date_value
# Function to calculate random completion time based on difficulty
def calculate_completion_time(row):
    if (date_to_days(row['Project_Start_Date']) > date_to_days(row['Project_Deadline'])):
        temp = row['Project_Start_Date'];
        row['Project_Start_Date'] = row['Project_Deadline']
        row['Project_Deadline'] = temp
    if row['Project_Difficulty'] == 'Medium':
        random_value = np.random.randint(-2, 5)
    elif row['Project_Difficulty'] == 'Hard':
        random_value = np.random.randint(5, 10)
    else:  # Assume 'Easy'
        random_value = np.random.randint(-5, 0)
    
    completion_time = date_to_days(row['Project_Deadline']) - date_to_days(row['Project_Start_Date']) + random_value
    return completion_time

# Apply the function to create the 'Completion_Time' column
df['Completion_Time'] = df.apply(calculate_completion_time, axis=1)



df.to_csv('clean_synthetic_data.csv', index=False) # Replace "output.csv" with your desired filename

df[["Project_Description", "Project_Difficulty", "Role"]].to_csv('clean_synthetic_data_classifier.csv', index=False) 
