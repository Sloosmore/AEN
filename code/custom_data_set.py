from typing import Any, Iterator
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import re
import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Union, List
import multiprocessing as mp
import dask.dataframe as dd
from tqdm import tqdm



# With version 2 we no longer use an RNN so that means grabing the data is pretty simple.

# 1. grab this data and make sure it is in the right format
# 2. for the transcripts convert the speaker into a token at the front 
# 3. Reduce transctipt into an array of prompts 
# 3. Split up RNN task into a series of responces (thinking 5 splitups per set but vary the leng obv)

class Load_Py_CSV(Dataset):
    def __init__(self, csv_file: Union[str, List[str]], tokenizer, train: bool = True, test_size: float = 0.2, random_state: int = 42):
        
        if isinstance(csv_file, list):
            data = dd.concat([dd.read_csv(file) for file in csv_file])
        else:
            data = dd.read_csv(csv_file)

        data['transcript'] = data['transcript'].apply(self.update_ts, meta=('transcript', 'str'))
        data['prompts'] = data['prompts'].apply(self.clean_thresholds, meta=('prompts', 'str'))
        data['labels'] = data['labels'].astype(float)
        
        data_pd = data.compute()
        
        print("Cleaning done, now splitting")
        
        # Split data into training and test sets
        if test_size == 0:
            train_data = data_pd
            test_data = pd.DataFrame(columns=data_pd.columns)
        elif test_size == 1:
            train_data = pd.DataFrame(columns=data_pd.columns)
            test_data = data_pd
        else:
            # Split data into training and test sets
            train_data, test_data = train_test_split(data_pd, test_size=test_size, random_state=random_state)

        # Store tokenizer
        self.tokenizer = tokenizer

        self.data = train_data if train else test_data
        
        self.data['transcript_length'] = self.batch_calculate_length(self.data['transcript'].tolist())
        self.data = self.data.sort_values('transcript_length').reset_index(drop=True)
 
        print(f"Loaded {len(self.data)} samples, train={train}")

    def calculate_length(self, text):
        return len(self.tokenizer(text)['input_ids'])
    

    def batch_calculate_length(self, texts, batch_size=1024):
        lengths = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating lengths", total=(len(texts) + batch_size - 1) // batch_size):
            batch = texts[i:i+batch_size]
            tokenized = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            lengths.extend([len(ids) for ids in tokenized['input_ids']])
        return lengths


    
    def update_ts(self, caption):
        if isinstance(caption, list):
            if len(caption) == 1:
                caption = caption[0]
            else: 
                raise ValueError('the caption should not be a list longer then one')
        colon_index = caption.find(':')
        cleaned_message = caption[colon_index + 2:] if colon_index != -1 else caption.strip()
        return cleaned_message
    
    
    def clean_thresholds(self, threshold:str)->str:
        if threshold.startswith("When someone "):
            modified_phrase = threshold[13:]  # Remove the first 5 characters ("When ")
            threshold = modified_phrase[0].upper() + modified_phrase[1:]
        return threshold
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray)):
            batch_data = self.data.iloc[idx]
            return (
                batch_data['transcript'].tolist(),
                batch_data['prompts'].tolist(),
                torch.tensor(batch_data['labels'].values, dtype=torch.float32)
            )
        else:
            return (
                self.data.loc[idx, 'transcript'],
                self.data.loc[idx, 'prompts'],
                torch.tensor(self.data.loc[idx, 'labels'], dtype=torch.float32)
            )
             

class BatchLenSampler(Sampler):
    def __init__(self, data_source, batch_size, seed: int = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.seed = seed if seed is not None else np.random.randint(0, 77)
        self.num_samples = len(self.data_source)
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size  # Ceiling division
        self.rng = np.random.default_rng(self.seed)

    def __len__(self):
        return self.num_batches
   
    def __iter__(self) -> Iterator:    
        indices = np.arange(self.num_samples)
        self.rng.shuffle(indices)
        
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_samples)
            yield indices[start_idx:end_idx]


class Collate_Fn:
    def __init__(self) -> None:
        pass
    
    def __call__(self, batch) -> Any:
        caption, thresh, labels = batch[0] if isinstance(batch,list) else batch
        caption = [item for item in caption] if isinstance(caption, tuple) else caption
        thresh = [labels for labels in thresh] if isinstance(thresh, tuple) else thresh

        return caption, thresh, labels


if __name__ == "__main__":
    random_seed = 42
    data_path = ["data_pipeline/data/500k/final_data.csv", "data_pipeline/data/500k2/final_data.csv", "data_pipeline/data/1m/explode_final.csv"]
    batch_size = 32
    MiniLM_L6 = {'path':'sentence-transformers/all-MiniLM-L6-v2', 'size':384}
    tokenizer = AutoTokenizer.from_pretrained(MiniLM_L6['path'])
    test_size = .025
    collate = Collate_Fn()

    test_dataset  = Load_Py_CSV(data_path, tokenizer=tokenizer, test_size=test_size, train=False)
    train_dataset = Load_Py_CSV(data_path, tokenizer=tokenizer, test_size=test_size)

    train_sampler = BatchLenSampler(data_sourse=train_dataset, batch_size=batch_size, seed=random_seed)
    test_sampler = BatchLenSampler(data_sourse=test_dataset, batch_size=batch_size, seed=random_seed)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, collate_fn=collate)

    # Iterate over the dataloader to get a random batch
    print(next(iter(train_dataset)))
    caption, thresh, labels = (next(iter(train_dataloader)))
    print(labels)
    print(caption)



    """
    print(caption)
    print(thresh)
    print(labels)
    """
    
    
    