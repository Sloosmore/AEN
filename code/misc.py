from datetime import datetime

def get_ordinal_suffix(day):
    if 11 <= day <= 13:
        return 'th'
    else:
        return {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

def current_date_with_ordinal():
    now = datetime.now()
    month = now.strftime("%B")
    day = now.day
    ordinal_suffix = get_ordinal_suffix(day)
    formatted_date = f"{month} {day}{ordinal_suffix}"
    return formatted_date


def check_grad_devices(model, device):
    for param in model.parameters():
        if param.grad is not None and param.grad.device != device:
            print(f"Gradient of {param} is on {param.grad.device}, should be on {device}")
            param.grad = param.grad.to(device)
            
def check_model_devices(model, device):
    for param in model.parameters():
        if param.device != device:
            print(f"Parameter {param} is on {param.device}, should be on {device}")
            param.data = param.data.to(device)


def safe_round(value, decimals=3):
    """
    Safely round a value whether it's a tensor or a regular number.
    
    Args:
    value: The value to round. Can be a tensor or a regular number.
    decimals: The number of decimal places to round to. Default is 3.
    
    Returns:
    The rounded value as a Python float.
    """
    try:
        # Try to use .item() method (for tensors)
        return round(value.item(), decimals)
    except AttributeError:
        # If .item() method is not available, it's likely a regular number
        return round(value, decimals)

if __name__ == "__main__":
    print(current_date_with_ordinal())

