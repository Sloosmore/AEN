import torch
import time
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def clear_memory(device):
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()

def train_loop(model, tokenizer, train_data, optim, loss_fn, model_metrics, device, scaler=None, metric_compute_interval=50):
    """
    Train the model for one epoch.

    Parameters:
    model (torch.nn.Module): The model to train.
    train_data (DataLoader): DataLoader for training data.
    optim (torch.optim.Optimizer): Optimizer for the model.
    loss_fn (callable): Loss function.
    model_metrics (List of callables): Metric functions (e.g., F1 score).
    device (torch.device): Device to use for computation.

    Returns:
    float, float: Average loss and F1 score over the training data.
    """
    model.train()
    
    tot_loss = torch.tensor(0.0).to(device)
    tot_metrics = [torch.tensor(0.0).to(device) for _ in model_metrics]
    
    progress_bar = tqdm(train_data, desc="Training", leave=False)

    for i_batch, batch in enumerate(progress_bar):
        
        captions, prompts, target = batch
        with torch.no_grad():
            token_caption = tokenizer(captions, padding=True, truncation=True, return_tensors='pt').to(device)
            tokenized_prompts = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(device)      
        # Assuming 'target' is also a tensor, move it to the device
        target = target.to(device)

        loss = None
        pred = None

        if scaler is not None:
            # Use mixed precision for CUDA
            with autocast():
                pred = model(token_caption, tokenized_prompts)
                loss = loss_fn(pred, target)
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"NaN or Inf detected in loss at batch {i_batch}")
                break

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            
        else:
            # Use full precision for CPU and MPS
            pred = model(token_caption, tokenized_prompts)
            loss = loss_fn(pred, target).to(device)
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"NaN or Inf detected in loss at batch {i_batch}")
                break
            
            loss.backward()
            optim.step()

        optim.zero_grad(set_to_none=True)

        tot_loss += loss.item()


        clear_memory(device)
        
        # if i_batch % 50 == 0:
        #     print(f"Batch num: {i_batch} | Loss: {loss:.3f} | F1: {f1:.3f}")

        if (i_batch + 1) % metric_compute_interval == 0:
            with torch.inference_mode():
                metric_scores = [metric(pred.detach(), target) for metric in model_metrics]
                tot_metrics = [tot + score.item() for tot, score in zip(tot_metrics, metric_scores)]
    
    tot_loss /= len(train_data)
    tot_metrics = [tot / (len(train_data) // metric_compute_interval) for tot in tot_metrics]
    
    return tot_loss, tot_metrics

def test_loop(model, tokenizer, test_data, loss_fn, model_metrics, device):
    with torch.inference_mode():
        model.eval()
        test_tot_loss = torch.tensor(0.0).to(device)
        test_tot_metrics = [torch.tensor(0.0).to(device) for _ in model_metrics]
        for i_batch, batch in enumerate(test_data):
            
            captions, prompts, target = batch
            
            token_caption = tokenizer(captions, padding=True, truncation=True, return_tensors='pt').to(device)
            tokenized_prompts = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(device)      
            target = target.to(device)
            
            # no need to asign threshold on test as it will not change
            test_pred = model(token_caption, tokenized_prompts)
            test_tot_loss += loss_fn(test_pred, target)
            metric_scores = [metric(test_pred, target) for metric in model_metrics]
            test_tot_metrics = [tot + score.item() for tot, score in zip(test_tot_metrics, metric_scores)]

            
            clear_memory(device)
            
        
        test_tot_loss /= len(test_data)
        test_tot_metrics = [tot / len(test_data) for tot in test_tot_metrics]

        
    return test_tot_loss, test_tot_metrics            

def train_model(model, tokenizer, optim, loss_fn, train_data, test_data, metrics, epochs:int, device:str='cpu', freeze_epochs:any = None, start_epochs=0):
    start_time = time.time()
    train_loss_ar, train_metrics_ar, test_loss_ar, test_metrics_ar = [], [], [], []
    
    scaler = GradScaler() if device == 'cuda' else None
    metric_names = [f.__class__.__name__ for f in metrics]


    for i in range(epochs):
        if isinstance(freeze_epochs, int) and i+1 == freeze_epochs:
            model.set_encoder_trainable(True)
            print(f"Epoch {i+1}: Encoder unfrozen")
            
        
        tot_loss, tot_metrics = train_loop(model=model, tokenizer=tokenizer, train_data=train_data, loss_fn=loss_fn, model_metrics=metrics, optim=optim, device=device, scaler=scaler)
        test_loss, test_metrics = test_loop(model=model, tokenizer=tokenizer, test_data=test_data, loss_fn=loss_fn, model_metrics=metrics, device=device)

        train_metrics_str = ', '.join(f'{name}: {m:.3f}' for name, m in zip(metric_names, tot_metrics))
        test_metrics_str = ', '.join(f'{name}: {m:.3f}' for name, m in zip(metric_names, test_metrics))
        print(f"Epoch: {i+1+start_epochs} | Train Loss: {tot_loss:.3f} |{train_metrics_str} | Test Loss: {test_loss:.3f} | {test_metrics_str}")
        
        
        train_loss_ar.append(tot_loss.cpu().item())
        train_metrics_ar.append([m.cpu().item() for m in tot_metrics])
        test_loss_ar.append(test_loss.cpu().item())
        test_metrics_ar.append([m.cpu().item() for m in test_metrics])
    
    end_time = time.time()
    
    
    return {
        "train_loss_ar": train_loss_ar,
        "train_metrics_ar": tot_metrics,
        "test_loss_ar": test_loss_ar,
        "test_metrics_ar": test_metrics,
        'time_elapsed': end_time - start_time
    }
    

# Example usage:
# probabilities = compute_probabilities("input caption", "threshold text", your_model, your_tokenizer, torch.device('cuda'))
# print(probabilities)

    
if __name__ == "__main__":
    print('script is being run directly')