import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from scripts.data.CustomDataset import CustomDataset  # Assuming you have saved the CustomDataset class in a file named custom_dataset.py

# Define a function to create the CNN model with variable architecture
def create_model(trial):
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 5)
    num_filters = [trial.suggest_int(f'num_filters_layer_{i}', 8, 64) for i in range(num_conv_layers)]

        
    return CNN(image_size, num_filters, num_conv_layers)

# Define the objective function for Optuna
def objective(trial):

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize the image to 32x32
        transforms.ToTensor(),         # Convert to tensor
    ])

    
# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
