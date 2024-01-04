import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-alpha", device_map="auto", load_in_4bit=True
)

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Step 1: Read the CSV File
def read_employee_data(file_path):
    return pd.read_csv(file_path)

# Step 2: Search for Employee Data
def get_employee_data(df, employee_name):
    return df[df['Employee Name'] == employee_name].to_dict('records')[0]

# Step 3: Prepare Context for the Language Model
def prepare_context(employee_data):
    context = (
        f"Employee Review:\n"
        f"Name: {employee_data['Employee Name']}\n"
        f"ID: {employee_data['ID']}\n"
        f"Gender: {employee_data['Gender']}\n"
        f"Age: {employee_data['Age']}\n"
        f"Marital Status: {employee_data['Married']}\n"
        f"Role: {employee_data['Role']}\n"
        f"Salary: {employee_data['Salary']}\n"
        f"Position: {employee_data['Position']}\n"
        f"Absences: {employee_data['Absences']}\n"
        f"Projects Completed: {employee_data['Projects_Completed']}\n"
        f"Mean Monthly Hours: {employee_data['Mean Monthly Hours']}\n"
        f"Years in the Company: {employee_data['Years in the company']}\n"
        f"Joining Year: {employee_data['Joining_Year']}\n"
        f"Current Employment Rating: {employee_data['Current_Employ_Rating']}\n"
        f"Moral: {employee_data['Moral']}\n"
        f"Stress & Burnout Score: {employee_data['Stress & Burnout Score']}\n"
        f"Ongoing Project Count: {employee_data['Ongoing_Project_Count']}\n"
        f"Projects Within Deadline: {employee_data['Projects_Within_Deadline']}\n"
        f"Project Start Date: {employee_data['Project_Start_Date']}\n"
        f"Project Description: {employee_data['Project_Description']}\n"
        f"Project Difficulty: {employee_data['Project_Difficulty']}\n"
        f"Project Deadline: {employee_data['Project_Deadline']}\n"
        f"Manager ID: {employee_data['Manager_ID']}\n"
        f"Completion Time: {employee_data['Completion_Time']}\n"
    )
    return context

# Set a random seed for reproducibility
set_seed(0)

# Example Usage
file_path = "/content/drive/MyDrive/Task_Whiz/clean_data.csv"  # Replace with your file path
employee_name = "Janvi Arya"  # Replace with the name you're searching for

# Read and search for the employee data
df = read_employee_data(file_path)
employee_data = get_employee_data(df, employee_name)

# Prepare context for the language model
context = prepare_context(employee_data)
print(context)

# Modify few-shot prompting
messages = [
    {"role": "system", "content": "You are an advanced AI trained to provide insightful reviews of employee performance based on given data."},
    {"role": "user", "content": '''Employee Review:
Name: Ram Prasad
ID: 1
Gender: Male
Age: 34
Marital Status: Married
Role: Backend Developer
Salary: 2,60,000
Position: Senior
Absences: 2
Projects Completed: 42
Mean Monthly Hours: 143
Years in the Company: 5
Joining Year: 2015
Current Employment Rating: 9
Moral: Very High
Stress & Burnout Score: 3
Ongoing Project Count: 3
Projects Within Deadline: 22
Project Start Date: 5/2/2024
Project Description: Design an efficient algorithm for data processing and visualization
Project Difficulty: High
Project Deadline: 15/3/2024
Manager ID: 4
Completion Time: 37'''},
    {"role": "assistant", "content": '''Ram Prasad, a Senior Backend Developer, has been a notable asset to our team for five years. His track record of completing 42 projects, including 22 within the deadline, highlights his efficiency and dedication. Specializing in high-difficulty tasks, such as designing algorithms for data processing, he demonstrates both technical expertise and innovative problem-solving skills.

Despite a high workload, as indicated by his 143 mean monthly hours, Ram maintains a 'Very High' morale and manages a balance between work and well-being, evidenced by his resilience in high-stress environments. His long tenure since 2015 and an impressive employment rating of 9 reflect his consistent performance and value to the team.

While managing stress more effectively could be a focus area, Ram's overall contribution, marked by a strong work ethic and a positive attitude, is highly commendable. His growth and success are integral to our team's progress.'''},
    {"role": "user", "content": "Employee Review:\nName: Sara Jenkins\nRole: Project Manager\nYears in the company: 3\nProjects Completed: 30\nProject Success Rate: 90%\nMean Monthly Hours: 130\nJoining Year: 2017\nCurrent Employment Rating: 8.5\nStress Level: Moderate"},
    {"role": "assistant", "content": "Sara Jenkins, a Project Manager, has shown remarkable leadership over three years. Her management of 30 projects, with a 90% success rate, showcases her project coordination skills. Balancing 130 monthly hours, she demonstrates a work-life balance, maintaining moderate stress. Her tenure since 2017 and a rating of 8.5 reflect her strong alignment with our goals. Sara's potential for further growth in strategic planning is exciting."},
    {"role": "user", "content": "Employee Review:\nName: Alex Martinez\nRole: IT Specialist\nYears in the company: 4\nProjects Completed: 50\nProject On-Time Completion: 80%\nMean Monthly Hours: 120\nJoining Year: 2016\nCurrent Employment Rating: 8\nStress Level: Low"},
    {"role": "assistant", "content": "Alex Martinez, an IT Specialist for four years, has successfully completed 50 projects, 80% on time, showing his technical prowess and timeliness. Working 120 hours monthly, he balances efficiency and personal time. Since joining in 2016 with a rating of 8, Alex has consistently contributed to our tech advancements, while maintaining low stress levels. His approach to project management and time efficiency is highly effective."},
    {"role": "user", "content": "Employee Review:\nName: Emily Chen\nRole: Marketing Analyst\nYears in the company: 2\nProjects Completed: 20\nProject Success Rate: 85%\nMean Monthly Hours: 135\nJoining Year: 2018\nCurrent Employment Rating: 7.5\nStress Level: High"},
    {"role": "assistant", "content": "Emily Chen, a Marketing Analyst for two years, has completed 20 projects with an 85% success rate, demonstrating her analytical skills and creativity. Working 135 hours monthly, she shows dedication, though her high stress level suggests a need for better work-life balance. Since 2018, with a rating of 7.5, Emily has been a growing asset. Her ability to manage stress while maintaining productivity will be key to her future success."},
    {"role": "user", "content": "Employee Review:\nName: Michael Brown\nRole: Sales Executive\nYears in the company: 6\nProjects Completed: 60\nProject Success Rate: 95%\nMean Monthly Hours: 150\nJoining Year: 2014\nCurrent Employment Rating: 9.2\nStress Level: Medium"},
    {"role": "assistant", "content": "Michael Brown, a Sales Executive for six years, has impressively completed 60 projects with a 95% success rate. His dedication is evident from 150 monthly hours, balancing substantial workload with medium stress levels. Since joining in 2014 with a high rating of 9.2, Michael has been a key player in driving sales growth. His ability to handle stress and maintain high performance is commendable and crucial for his role."},
    {"role": "user", "content": context}
]
model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
input_length = model_inputs.shape[1]

# Generate the text review
generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=200)
review = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
print(review)
