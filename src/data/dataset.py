from transformers import AutoTokenizer
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import torch
from ..models.model import tokenizer

def preprocess_func(example, max_input_length, max_target_length):
    inputs = example['Eng']
    targets = example['Rus']
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]

    model_inputs['input_ids'] = torch.as_tensor(model_inputs['input_ids'], dtype=torch.int32)
    model_inputs['attention_mask'] = torch.as_tensor(model_inputs['attention_mask'], dtype=torch.int32)
    model_inputs['labels'] = torch.as_tensor(model_inputs['labels'], dtype=torch.int32)
    return model_inputs

class TranslationDataset(Dataset):
    '''Dataset class for translation'''
    def __init__(self, csv_file : str, 
                 max_input_length=128, 
                 max_target_length=512):

        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.tokenized = self.df.apply(preprocess_func, axis=1, args=(max_input_length, max_target_length))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: any):
        return self.tokenized[index]

if __name__ == '__main__':
    #test functional
    
    dataset = TranslationDataset('data/matlab/train.csv', tokenizer)
    print(dataset[1])
    print(dataset.tokenized)
    
    print(dataset[0])