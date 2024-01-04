import pandas as pd

# Creating an empty DataFrame with specified columns
columns = [
    'Employee Name', 'ID', 'Gender', 'Age', 'Married', 'Role', 'Salary', 'Position', 
    'Absences', 'Projects_Completed', 'Mean Monthly Hours', 'Years in the company', 
    'Joining_Date', 'Current_Employ_Rating', 'Moral', 'Stress & Burnout Score', 
    'Ongoing_Project_Count', 'Within_Deadline', 'Project_Description', 
    'Project_Difficulty', 'Project_Deadline', 'Text_Feedback_Each_Manager'
]

df = pd.DataFrame(columns=columns)

# Saving the DataFrame as a CSV file
df.to_csv('employee_data.csv', index=False)

print("Empty DataFrame created and saved as 'employee_data.csv'.")
