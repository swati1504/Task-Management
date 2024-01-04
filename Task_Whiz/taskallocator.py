import torch
from transformers import BertModel, BertTokenizer
from torch import nn

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

class TaskAllocator:
    def __init__(self):
        # Define the path to the saved model checkpoint
        model_checkpoint_path = '/Users/dhruvnagill/Coding/Task_Whiz/bert_classifier.pth'


        self.device = torch.device("cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BERTClassifier('bert-base-uncased', 4)

        # Load the state_dict from the saved checkpoint
        state_dict = torch.load(model_checkpoint_path, map_location=self.device)

        # Load the state_dict into the model
        self.model.load_state_dict(state_dict)

        # Make sure the model is in evaluation mode
        self.model.eval()

        # Define your role names here
        self.role_names = ["Frontend Developer", "Backend Developer", "R&D", "HR"]

    def predict_role(self, text, max_length=128):
        encoding = self.tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            role_index = torch.argmax(outputs, dim=1).item()

        return self.role_names[role_index]

    def allocate_task(self, role, task_description, data):
        # Filter data based on the specified role
        self.data = data
        role_data = self.data[self.data['Role'] == role]

        # Check if there are any people in the specified role
        if role_data.empty:
            return f"No available person found for the role: {role}"

        # Filter people with zero ongoing projects
        zero_projects_data = role_data[role_data['Ongoing_Project_Count'] == 0]

        # Check if there are people with zero ongoing projects
        if not zero_projects_data.empty:
            # Assign the task to the person with the least stress score among those with zero ongoing projects
            selected_person = zero_projects_data.loc[zero_projects_data['Stress & Burnout Score'].idxmin()]
        else:
            # If no one has zero ongoing projects, assign the task to the person with the overall least stress score
            selected_person = role_data.loc[role_data['Stress & Burnout Score'].idxmin()]

        # Assign the task to the selected person
        return selected_person['Employee Name']


    def allocate_subtasks(self, subtasks_dict, data):
        allocation_results = {}
        for task_description, task_type in subtasks_dict.items():

            # Allocate the task
            allocation = self.allocate_task(task_type, task_description, data)

            # Update the allocation results to include the type, description, and employee
            allocation_results[task_description] = {
                "name": task_type,
                "description": task_description,
                "employee_allocated": allocation
            }

        return allocation_results
