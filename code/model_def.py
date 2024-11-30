import torch
from torch import nn
from transformers import AutoModel
from model_vectorize import create_batch_aware_vectorized_kde


class TSN_model(nn.Module):
    def __init__(self, 
                 encoder_model: str,
                 end_pred_sequential: nn.Sequential,
                 device: str = 'cpu',
                 encoder_tune: bool = False,
                 same_encoder: bool = True,
                 kde_config: dict = None
                 ) -> None:
        super().__init__()
        """
        Initializes the TSN_model class.

        Args:
            encoder_model (str): Path to the pre-trained model for encoding.
            end_pred_sequential (nn.Sequential): Sequential layers for final prediction.
            device (str): The device of the model.
            encoder_tune (bool): Whether to fine-tune the encoder.
            same_encoder (bool): Whether to use the same encoder for captions and prompts.
            kde_config (dict): Configuration for KDE. Should contain:
                - 'apply_to': 'caption' or 'threshold'
                - 'bandwidth': float or None
                - 'pdf_type': 'gaussian', 'epanechnikov', or 'triangular'
        """
        self.device = device
        if same_encoder:
            self.caption_encoder = self.prompt_encoder = AutoModel.from_pretrained(encoder_model)
        else:
            self.caption_encoder = AutoModel.from_pretrained(encoder_model)
            self.prompt_encoder = AutoModel.from_pretrained(encoder_model)
                
        self.pred_layer = end_pred_sequential.to(device)
        self.encoder_tune = encoder_tune
        self.set_encoder_trainable(encoder_tune)

        self.kde_config = kde_config or {'apply_to': 'caption', 'bandwidth': None, 'pdf_type': 'gaussian'}
        self.kde_func = None  # Will be initialized in forward pass

    def set_encoder_trainable(self, trainable: bool):
        """Set the requires_grad parameter for the encoder(s) in the TSN_model."""
        self.encoder_tune = trainable
        if self.caption_encoder is self.prompt_encoder:
            self.caption_encoder.requires_grad_(trainable).to(self.device)
        else:
            self.caption_encoder.requires_grad_(trainable).to(self.device)
            self.prompt_encoder.requires_grad_(trainable).to(self.device)
        
        print(f"Encoder(s) trainable status set to: {trainable}")

    def get_encoder_trainable(self) -> bool:
        """Get the current trainable status of the encoder(s)."""
        return self.encoder_tune

    def mean_pooling(self, model_output, attention_mask):
        """Applies mean pooling to the model output."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        return torch.sum(model_output * input_mask_expanded, -2) / torch.clamp(input_mask_expanded.sum(-2), min=1e-9)
       
    def forward(self, caption_dict, threshold_dict):
        """
        Forward pass of the model.

        Args:
            caption_dict: Dictionary containing caption tokens and attention mask.
            threshold_dict: Dictionary containing threshold tokens and attention mask.

        Returns:
            torch.Tensor: The output logits.
        """
        # Encode captions and thresholds
        caption_output = self.caption_encoder(**caption_dict)
        threshold_output = self.prompt_encoder(**threshold_dict)

        caption_embeddings = self.mean_pooling(caption_output[0], caption_dict['attention_mask'])
        threshold_embeddings = self.mean_pooling(threshold_output[0], threshold_dict['attention_mask'])

        # Determine which embeddings to apply KDE to
        if self.kde_config['apply_to'] == 'caption':
            kde_data = caption_output[0]
            kde_mask = caption_dict['attention_mask']
            target_embeddings = threshold_embeddings
        else:  # apply to threshold
            kde_data = threshold_output[0]
            kde_mask = threshold_dict['attention_mask']
            target_embeddings = caption_embeddings

        # Create or update KDE function
        self.kde_func = create_batch_aware_vectorized_kde(
            kde_data, 
            kde_mask, 
            bandwidth=self.kde_config['bandwidth'],
            pdf_type=self.kde_config['pdf_type']
        )

        target_embeddings_expanded = target_embeddings.unsqueeze(1)
        
        kde_values = self.kde_func(target_embeddings_expanded)

        kde_values = torch.clamp(kde_values.squeeze(1), min=-1e6, max=1e6)

        # Debugging: Final prediction
        #pred = self.pred_layer(kde_values).squeeze()

        
        return self.pred_layer(kde_values).squeeze()
        


if __name__ == '__main__':
    MiniLM_L6 = {'path': 'sentence-transformers/all-MiniLM-L6-v2', 'size': 384}
    
    # Create a simple end_pred_sequential
    input_size = 384  # Size of the KDE output (same as embedding dimension)
    end_pred_sequential = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )
    
    kde_config = {
        'apply_to': 'caption',
        'bandwidth': None,
        'pdf_type': 'gaussian'
    }
    
    model_01 = TSN_model(
        encoder_model=MiniLM_L6['path'],
        end_pred_sequential=end_pred_sequential,
        kde_config=kde_config
    )

    print("TSN_model with KDE initialized successfully.")
    
