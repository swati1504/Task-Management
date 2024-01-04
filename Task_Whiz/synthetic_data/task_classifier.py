from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Define labels for difficulty and role
difficulty_labels = ["High", "Medium", "Low"]
role_labels = ["Fronted", "Backend", "R&D", "HR"]

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(difficulty_labels) + len(role_labels))

# Function to preprocess data
def preprocess_data(data):
    project_descriptions = data["Project_Description"].tolist()

    encoded_data = tokenizer(project_descriptions, padding="max_length", truncation=True)

    input_ids = encoded_data["input_ids"]
    attention_mask = encoded_data["attention_mask"]

    difficulty_labels = data["Project_Difficulty"]
    role_labels = data["Role"]

    return input_ids, attention_mask, difficulty_labels, role_labels

# Load your dataset into a pandas DataFrame
df = pd.read_csv("clean_synthetic_data.csv")

# Preprocess data
input_ids, attention_mask, difficulty_labels, role_labels = preprocess_data(df)

# Split data into train and test sets
X_train, X_test, y_train_difficulty, y_test_difficulty, y_train_role, y_test_role = train_test_split(
    input_ids, difficulty_labels, role_labels, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_difficulty_tensor = torch.tensor(y_train_difficulty.values)
y_train_role_tensor = torch.tensor(y_train_role.values)

X_test_tensor = torch.tensor(X_test)
y_test_difficulty_tensor = torch.tensor(y_test_difficulty.values)
y_test_role_tensor = torch.tensor(y_test_role.values)

# Create PyTorch datasets
train_dataset_difficulty = TensorDataset(X_train_tensor, y_train_difficulty_tensor)
train_dataset_role = TensorDataset(X_train_tensor, y_train_role_tensor)

eval_dataset_difficulty = TensorDataset(X_test_tensor, y_test_difficulty_tensor)
eval_dataset_role = TensorDataset(X_test_tensor, y_test_role_tensor)

# Create data loaders
train_dataloader_difficulty = DataLoader(train_dataset_difficulty, batch_size=8, shuffle=True)
train_dataloader_role = DataLoader(train_dataset_role, batch_size=8, shuffle=True)

eval_dataloader_difficulty = DataLoader(eval_dataset_difficulty, batch_size=8, shuffle=False)
eval_dataloader_role = DataLoader(eval_dataset_role, batch_size=8, shuffle=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="model_output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=1000,
    eval_steps=500,
)

# Trainer for difficulty
trainer_difficulty = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader_difficulty,
    eval_dataset=eval_dataloader_difficulty,
)

# Train the model for difficulty
trainer_difficulty.train()

# Evaluate the model for difficulty
y_pred_difficulty = trainer_difficulty.predict(eval_dataloader_difficulty.dataset)
predictions_difficulty = y_pred_difficulty.predictions.argmax(-1)

# Calculate accuracy and classification report for difficulty
accuracy_difficulty = accuracy_score(y_test_difficulty_tensor.numpy(), predictions_difficulty)
classification_report_difficulty = classification_report(y_test_difficulty_tensor.numpy(), predictions_difficulty, labels=difficulty_labels)

# Trainer for role
trainer_role = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader_role,
    eval_dataset=eval_dataloader_role,
)

# Train the model for role
trainer_role.train()

# Evaluate the model for role
y_pred_role = trainer_role.predict(eval_dataloader_role.dataset)
predictions_role = y_pred_role.predictions.argmax(-1)

# Calculate accuracy and classification report for role
accuracy_role = accuracy_score(y_test_role_tensor.numpy(), predictions_role)
classification_report_role = classification_report(y_test_role_tensor.numpy(), predictions_role, labels=role_labels)

# Print results for difficulty
print("Accuracy for Project Difficulty:", accuracy_difficulty)
print("Classification Report for Project Difficulty:")
print(classification_report_difficulty)

# Print results for role
print("Accuracy for Role:")
print(accuracy_role)
print("Classification Report for Role:")
print(classification_report_role)

# Save the trained model for future use
model.save_pretrained("saved_model")
