import torch
import time


def compute_probabilities_full(cap:str, thresh:str, model, tokenizer, device):
    """
    Function to compute probabilities using the given model and tokenizer.

    Args:
    cap (str): The caption or input text.
    thresh (str): The threshold or prompt text.
    model (torch.nn.Module): The pre-trained model.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the model.
    device (torch.device): The device to run the computation on (e.g., 'cuda' or 'cpu').

    Returns:
    torch.Tensor: The computed probabilities.
    """
    
    # Tokenize the inputs and move them to the specified device
    token_ts = tokenizer(cap, padding=True, truncation=True, return_tensors='pt').to(device)
    tokenized_prompts = tokenizer(thresh, padding=True, truncation=True, return_tensors='pt').to(device)

    # Extract input_ids and attention_mask for both inputs
    input_ids_chunk = token_ts['input_ids']
    attention_mask_chunk = token_ts['attention_mask']
    input_ids_prompt = tokenized_prompts['input_ids']
    attention_mask_prompt = tokenized_prompts['attention_mask']

    # Set the model to evaluation mode and disable gradient computation
    model.eval()
    with torch.inference_mode():
        start_time = time.time()
        # Forward pass through the model
        model_out = model(input_ids_chunk, attention_mask_chunk, input_ids_prompt, attention_mask_prompt)
        
        # Apply the sigmoid function to get probabilities
        probabilities = torch.sigmoid(model_out)
        end_time = time.time()
        time_passed = end_time - start_time
        print(f"time for inference: {time_passed:.3f}")
    
    return probabilities


def compute_probabilities_parallel(caption:str, thresh:torch.Tensor, encoder, model, tokenizer, device):
    
    thresh = thresh.to(device)
    token_ts = tokenizer(caption, padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.inference_mode():
        cap_vec = encoder(**token_ts).unsqueeze(dim=0)
        # shape (1, x)
        cap_mat = cap_vec.expand(thresh.shape[0], -1)
        # shape (# of thres, x)
        vec_input = torch.cat((cap_mat, thresh, torch.abs(cap_mat-thresh)), dim=1).to(device)
        # shape (# of thes, cap, thresh, |cap-thres|)
        logits = model(vec_input)
        
        return torch.sigmoid(logits)
        
        
        
        
    
    
