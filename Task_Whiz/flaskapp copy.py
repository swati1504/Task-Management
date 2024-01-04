from openapi import OpenAPI
from taskallocator import TaskAllocator
from inference import Inference
from flask import Flask, request, jsonify
from pyngrok import ngrok
import pandas as pd
from chatbot import ChatBot

app = Flask(__name__)

port = 8070

public_url = ngrok.connect(port).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

app.config["BASE_URL"] = public_url

# Load the data
online_data = pd.read_csv("/Users/dhruvnagill/Coding/Task_Whiz/data/online/data.csv")
data_path = "/Users/dhruvnagill/Coding/Task_Whiz/clean_data.csv"
models_dir = '/Users/dhruvnagill/Coding/Task_Whiz/models'
inference = Inference(online_data, models_dir)
openapi = OpenAPI("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")
task_allocator = TaskAllocator()
chatbot = ChatBot("/Users/dhruvnagill/Coding/Task_Whiz/data/online/data.csv", "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")


@app.route('/test', methods=['GET'])
def test():
    return jsonify("Working")

@app.route('/decompose_allocate_task', methods=['POST'])
def decompose_allocate_task():
    content = request.json
    task_description = content['task']
    # Decompose the task into subtasks
    subtasks = openapi.decompose_task(task_description)

    # Allocate subtasks to employees
    allocation_results = task_allocator.allocate_subtasks(subtasks, online_data)

    # Transform allocation_results into the desired format
    allocations = []
    for task_desc, details in allocation_results.items():
        allocation = {
            "name": details["name"],  # Assuming this is the subtask name
            "Description": details["description"],
            "allotedUser": details["employee_allocated"]
        }
        allocations.append(allocation)

    # Format the response
    response = {
        'allocations': allocations
    }
    
    return jsonify(response)

@app.route('/get_employee_review', methods=['POST'])
def get_employee_review():
    content = request.json
    employee_name = content['name']

    # Generate employee review
    review = openapi.generate_employee_review(online_data, employee_name)

    # Format the response
    response = {
        'employee_name': employee_name,
        'review': review
    }

    return jsonify(response)

@app.route('/predict_stress_score', methods=['POST'])
def predict_stress_score():
    content = request.json
    employee_name = content['name']

    # Assuming a function to get the stress score prediction
    stress_score = inference.predict_stress_score(employee_name)

    return jsonify(int(stress_score[0]))

@app.route('/predict_moral_score', methods=['POST'])
def predict_moral_score():
    content = request.json
    employee_name = content['name']

    # Assuming a function to get the moral score prediction
    moral_score = inference.predict_moral(employee_name)
    mapping = {
        4:"Very High",
        3:"High",
        2:"Moderate",
        1:"Low",
        0:"Very Low"
    }
    return jsonify(mapping[moral_score[0]])

@app.route('/predict_completion_time', methods=['POST'])
def predict_completion_time():
    content = request.json
    employee_name = content['name']

    # Assuming a function to get the completion time prediction
    completion_time = inference.predict_completion_time(employee_name)

    return jsonify(int(completion_time[0]))

@app.route('/generate_email', methods=['POST'])
def generate_email_route():
    content = request.json
    input_text = content['email_points']
    response = openapi.generate_email(input_text)
    return jsonify({'email': response})

@app.route('/summarise', methods=['POST'])
def summarise_route():
    content = request.json
    input_text = content['text']
    response = openapi.summarise(input_text)
    return jsonify({'summary': response})

@app.route('/improve', methods=['POST'])
def improve_route():
    content = request.json
    original_text = content['original_text']
    response = openapi.improve(original_text)
    return jsonify({'improved_text': response})

@app.route('/chat', methods=['POST'])
def chat_route():
    content = request.json
    original_text = content['query']
    response = chatbot.chat(original_text)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(port = "8070")
