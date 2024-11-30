import torch
import math
import numpy as np
from typing import Union

class BatchAwareVectorizedKDE:
    def __init__(self, data: torch.Tensor, attention_mask: torch.Tensor, bandwidth: Union[float, str, torch.Tensor] = None, pdf_type: str = 'gaussian'):
        self.device = data.device
        self.data = data
        self.attention_mask = attention_mask
        self.batch_size, self.num_tokens, self.embedding_dim = data.shape
        
        # Apply attention mask to data
        self.masked_data = data * attention_mask.unsqueeze(-1)
        valid_tokens_per_batch = torch.clamp(attention_mask.sum(dim=1), min=1)
        

        # Compute bandwidth using Scott's rule if not provided
        if not isinstance(bandwidth, (torch.Tensor, float, int, list)):
            # Calculate the number of valid tokens per batch
            
            # Calculate the standard deviation of valid tokens

            masked_sum = (self.masked_data * attention_mask.unsqueeze(-1)).sum(dim=1)
            masked_mean = masked_sum / valid_tokens_per_batch.unsqueeze(-1)

            # Calculate the variance of valid tokens
            squared_diff = ((self.masked_data - masked_mean.unsqueeze(1)) * attention_mask.unsqueeze(-1)) ** 2
            masked_var = squared_diff.sum(dim=1) / (valid_tokens_per_batch.unsqueeze(-1) - 1)

            # Calculate the standard deviation
            masked_std = torch.sqrt(masked_var + 1e-8)
            
            # Calculate bandwidth using Scott's rule
            if bandwidth== 'scott':
                # Scott's rule
                scott_factor = valid_tokens_per_batch.unsqueeze(-1) ** (-1 / 5)
                self.bandwidth = torch.clamp(scott_factor * masked_std, min=1e-5, max=1000)
            elif bandwidth == 'silverman':
                # Silverman's rule
                silverman_factor = (valid_tokens_per_batch.unsqueeze(-1) * (self.embedding_dim + 2) / 4) ** (-1 / (self.embedding_dim + 4))
                self.bandwidth = torch.clamp(silverman_factor * masked_std, min=1e-5, max=1000)
            else:
                raise ValueError(f"Unsupported bandwidth method: {bandwidth}")
        else:
            self.bandwidth = torch.tensor(bandwidth, device=self.device)
        
        # Ensure bandwidth has the correct shape (batch_size, embedding_dim)
        if self.bandwidth.dim() == 0:
            self.bandwidth = self.bandwidth.expand(self.batch_size, self.embedding_dim)
        elif self.bandwidth.dim() == 1:
            self.bandwidth = self.bandwidth.unsqueeze(0).expand(self.batch_size, -1)
        
        assert self.bandwidth.shape == (self.batch_size, self.embedding_dim), f"Bandwidth shape should be ({self.batch_size}, {self.embedding_dim}), got {self.bandwidth.shape}"

        
        # Set the PDF function based on the specified type
        supported_pdfs = ['gaussian', 'epanechnikov', "triangular"]  # Add other supported PDFs here
        assert pdf_type in supported_pdfs, f"Unsupported PDF type. Choose from {supported_pdfs}"
        self.pdf_type = pdf_type
        if pdf_type == 'gaussian':
            self.pdf_func = self._gaussian_pdf
            self.norm_factor = 1 / (self.bandwidth * math.sqrt(2 * math.pi))
        elif pdf_type == 'epanechnikov':
            self.pdf_func = self._epanechnikov_pdf
            self.norm_factor = 3 / (4 * self.bandwidth)
        elif pdf_type == 'triangular':
            self.pdf_func = self._triangular_pdf
            self.norm_factor = 1 / self.bandwidth
        else:
            raise ValueError("Unsupported PDF type. Choose 'gaussian', 'epanechnikov', or 'triangular'.")

    def _gaussian_pdf(self, x):
        """Gaussian (normal) probability density function."""
        return torch.exp(-0.5 * x**2)

    def _epanechnikov_pdf(self, x):
        """Epanechnikov probability density function."""
        return torch.max(1 - x**2, torch.zeros_like(x))

    def _triangular_pdf(self, x):
        """Triangular probability density function."""
        return torch.max(1 - torch.abs(x), torch.zeros_like(x))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the KDE for given points x across all sentences.
        
        Args:
        x (torch.Tensor): 3D tensor of shape (batch_size, num_points, embedding_dim) containing the points to evaluate
        
        NOTE: If you didn't want to mean pool you could use number of points and do some evaluation across each tokens.
        This extra part is not used in training.
        
        Returns:
        torch.Tensor: KDE values for the input points, shape (batch_size, num_points, embedding_dim)
        """
        # Reshape masked_data to (batch_size, 1, num_tokens, embedding_dim) for efficient computation
        data = self.masked_data.unsqueeze(1)  # Shape: (batch_size, 1, num_tokens, embedding_dim)
        
        # Reshape bandwidth to (batch_size, 1, 1, embedding_dim) for broadcasting
        bandwidth = self.bandwidth.view(self.batch_size, 1, 1, self.embedding_dim)
        
        # Compute the scaled difference between x and data points
        """so for each batch, we take every token and subtract the embedding_dim of x from it. 
        So on a per point in the batch basis we subtract the embedding dim of x num of token times"""
        scaled_diff = (x.unsqueeze(2) - data) / bandwidth
        
        # Apply the chosen PDF function to the scaled differences
        kernel = self.pdf_func(scaled_diff)
        
        # Compute KDE values by summing over all valid data points
        # Then we devide by the sum of the attnetion mask to apply the needed normalization.
        kde_values = (kernel * self.attention_mask.view(self.batch_size, 1, -1, 1)).sum(dim=2) / self.attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        # we have already applied everything else except the norm so we send it here
        kde_values = kde_values * self.norm_factor.unsqueeze(1)
        
        # Return KDE values with shape (batch_size, num_points, embedding_dim)
        return torch.clamp(kde_values, min=1e-10, max=1e10)

    
    
def create_batch_aware_vectorized_kde(data: torch.Tensor, attention_mask: torch.Tensor, bandwidth: float = None, pdf_type: str = 'gaussian') -> BatchAwareVectorizedKDE:
    """
    Create a batch-aware vectorized KDE function for the given data.
    
    Args:
    data (torch.Tensor): 3D tensor of shape (batch_size, num_tokens, embedding_dim) containing the token embeddings
    attention_mask (torch.Tensor): 2D tensor of shape (batch_size, num_tokens) containing the attention mask
    bandwidth (float): The bandwidth of the kernel. If None, Scott's rule is used.
    pdf_type (str): The type of probability density function to use.
    
    Returns:
    BatchAwareVectorizedKDE: A callable object that computes the KDE for given points across all sentences and embedding dimensions
    """
    return BatchAwareVectorizedKDE(data, attention_mask, bandwidth, pdf_type)

# Helper function to apply KDE to a range of values
def apply_batch_kde_to_range(kde_func: BatchAwareVectorizedKDE, start: float, end: float, num_points: int) -> torch.Tensor:
    """
    Apply the batch-aware KDE function to a range of values for all embedding dimensions.
    
    Args:
    kde_func (BatchAwareVectorizedKDE): The KDE function to apply
    start (float): Start of the range
    end (float): End of the range
    num_points (int): Number of points to evaluate in the range
    
    Returns:
    torch.Tensor: KDE values for the specified range of points, shape (batch_size, num_points, embedding_dim)
    """
    x = torch.linspace(start, end, num_points, device=kde_func.device)
    x = x.unsqueeze(0).unsqueeze(-1).expand(kde_func.batch_size, -1, kde_func.embedding_dim)
    return kde_func(x)

def plot_batch_kde(kde_func, start, end, num_points, title, dims_to_plot=4):
    x = torch.linspace(start, end, num_points, device=kde_func.data.device)
    x = x.unsqueeze(0).unsqueeze(-1).expand(kde_func.batch_size, -1, kde_func.embedding_dim)
    kde_values = kde_func(x)  # Shape: (batch_size, num_points, embedding_dim)
    x = x[0, :, 0].cpu().numpy()
    kde_values = kde_values.cpu().numpy()
    
    dims_to_plot = min(dims_to_plot, kde_values.shape[2])
    rows = math.ceil(dims_to_plot / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
    fig.suptitle(title, fontsize=16)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, kde_values.shape[0]))
    
    for d in range(dims_to_plot):
        row = d // 2
        col = d % 2
        ax = axes[row, col] if rows > 1 else axes[col]
        for i in range(kde_values.shape[0]):  # For each sentence
            ax.plot(x, kde_values[i, :, d], color=colors[i], label=f'Sentence {i+1}' if d == 0 else "")
        ax.set_ylabel(f'Density (Dim {d+1})')
        ax.set_xlabel('Value')
        if d == 0:
            ax.legend()
    
    # Remove any unused subplots
    for d in range(dims_to_plot, rows * 2):
        row = d // 2
        col = d % 2
        fig.delaxes(axes[row, col] if rows > 1 else axes[col])
    
    plt.tight_layout()
    plt.show()

# Example usage and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer, AutoModel

    def encode_sentences(sentences, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**encoded_input)
        
        # Use the last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Get attention mask
        attention_mask = encoded_input['attention_mask']
        
        return last_hidden_state, attention_mask


    # Example sentences
    # sentences = [
    # """
    # In natural language processing (NLP), a word embedding is a representation of a word. The embedding is used in text analysis. Typically, the representation is a real-valued vector that encodes the meaning of the word in such a way that the words that are closer in the vector space are expected to be similar in meaning.[1] Word embeddings can be obtained using language modeling and feature learning techniques, where words or phrases from the vocabulary are mapped to vectors of real numbers.

    # Methods to generate this mapping include neural networks,[2] dimensionality reduction on the word co-occurrence matrix,[3][4][5] probabilistic models,[6] explainable knowledge base method,[7] and explicit representation in terms of the context in which words appear.[8]

    # Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing[9] and sentiment analysis.[10]

    # Development and history of the approach
    # In distributional semantics, a quantitative methodological approach to understanding meaning in observed language, word embeddings or semantic feature space models have been used as a knowledge representation for some time.[11] Such models aim to quantify and categorize semantic similarities between linguistic items based on their distributional properties in large samples of language data. The underlying idea that "a word is characterized by the company it keeps" was proposed in a 1957 article by John Rupert Firth,[12] but also has roots in the contemporaneous work on search systems[13] and in cognitive psychology.[14]
    # """
    # ]
    
    sentences = [
    "The cat slept.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a method of data analysis that automates analytical model building.",
    """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language, in particular how to program computers to 
    process and analyze large amounts of natural language data.""",
    """The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings 
    endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical 
    philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. 
    This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract 
    essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously 
    discussing the possibility of building an electronic brain."""
    ]
    

    embeddings, attention_mask = encode_sentences(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")

    # Create batch-aware KDE functions with different PDFs
    kde_gaussian = create_batch_aware_vectorized_kde(embeddings, attention_mask, pdf_type='gaussian')
    kde_epanechnikov = create_batch_aware_vectorized_kde(embeddings, attention_mask, pdf_type='epanechnikov')
    kde_triangular = create_batch_aware_vectorized_kde(embeddings, attention_mask, pdf_type='triangular')

    # Plot the KDEs
    plot_batch_kde(kde_gaussian, -3, 3, 100, "Gaussian KDE", dims_to_plot=4)
    plot_batch_kde(kde_epanechnikov, -3, 3, 100, "Epanechnikov KDE", dims_to_plot=4)
    plot_batch_kde(kde_triangular, -3, 3, 100, "Triangular KDE", dims_to_plot=4)

    # Ensure plots are displayed
    plt.show()

print("Batch-aware KDE plots have been generated and displayed.")