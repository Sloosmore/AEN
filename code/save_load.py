import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os

def create_path_if_not_exist(path):
    """
    Ensures that the directory for the provided path exists.
    If the directory does not exist, it is created.
    
    Parameters:
    path (str): The directory path to be checked and created if necessary.
    """
    # Extract the directory path from the full path
    directory = os.path.dirname(path)
    
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

# Assuming you have your TSN_model class defined as before

def save_model(model, optimizer, epoch, metrics, filepath):
    """
    Save the model state, optimizer state, and other training info.
    
    Args:
    model (nn.Module): The model to save
    optimizer (torch.optim.Optimizer): The optimizer used in training
    epoch (int): The current epoch number
    loss (float): The current loss value
    filepath (str): Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, model, optimizer=None, device='cpu'):
    """
    Load a saved model and optionally optimizer state.
    
    Args:
    filepath (str): Path to the saved checkpoint
    model (nn.Module): An instance of the model architecture
    optimizer (torch.optim.Optimizer, optional): The optimizer to load state into
    device (str): The device to load the model onto
    
    Returns:
    model (nn.Module): The loaded model
    optimizer (torch.optim.Optimizer): The loaded optimizer (if provided)
    epoch (int): The epoch at which the model was saved
    loss (float): The loss at which the model was saved
    """
    try:
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']
        
        print(f"Model loaded from {filepath}")
        print(f"Resuming from epoch {epoch}")
        return model, optimizer, epoch, metrics
    except FileNotFoundError:
        print(f"No checkpoint found at {filepath}. Starting from scratch.")
        return model, optimizer, 0, None
    except KeyError as e:
        print(f"Checkpoint file is missing key: {e}. The checkpoint may be corrupted.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise