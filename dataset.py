import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class ScoringDataset(Dataset):
    def __init__(self, csv_path, max_object_tokens=100, max_standard_tokens=100):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_object_tokens = max_object_tokens
        self.max_standard_tokens = max_standard_tokens
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        standard = str(row['standard'])
        obj = str(row['object'])
        score = float(row['score'])
        
        # 分别编码standard和object
        standard_tokens = self.tokenizer(
            standard,
            max_length=self.max_standard_tokens,
            truncation=True,
            add_special_tokens=False
        )
        
        object_tokens = self.tokenizer(
            obj,
            max_length=self.max_object_tokens,
            truncation=True,
            padding='max_length',
            add_special_tokens=False
        )
        
        # 拼接为BERT输入: [CLS] + standard + [SEP] + object + [SEP]
        input_ids = [self.tokenizer.cls_token_id] + \
                    standard_tokens['input_ids'] + \
                    [self.tokenizer.sep_token_id] + \
                    object_tokens['input_ids'] + \
                    [self.tokenizer.sep_token_id]
        
        attention_mask = [1] + \
                         [1] * len(standard_tokens['input_ids']) + \
                         [1] + \
                         object_tokens['attention_mask'] + \
                         [1]
        
        # 确保不超过BERT最大长度(512)
        input_ids = input_ids[:512]
        attention_mask = attention_mask[:512]
        
        # 计算object部分在序列中的起始位置
        object_start = 1 + len(standard_tokens['input_ids']) + 1  # [CLS] + standard + [SEP]
        
        # 创建object mask (100维): 1=真实token, 0=padding
        object_mask = torch.tensor(object_tokens['attention_mask'][:self.max_object_tokens])
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'object_start': object_start,
            'object_mask': object_mask,  # 用于屏蔽padding token的修正
            'score': torch.tensor(score, dtype=torch.float)
        }

def collate_fn(batch):
    """动态填充batch内序列到相同长度"""
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids_batch = []
    attention_mask_batch = []
    object_start_batch = []
    object_mask_batch = []
    scores_batch = []
    
    for item in batch:
        # 填充input_ids和attention_mask
        pad_len = max_len - len(item['input_ids'])
        input_ids = torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)])
        attention_mask = torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        
        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        object_start_batch.append(item['object_start'])
        object_mask_batch.append(item['object_mask'])
        scores_batch.append(item['score'])
    
    return {
        'input_ids': torch.stack(input_ids_batch),
        'attention_mask': torch.stack(attention_mask_batch),
        'object_start': torch.tensor(object_start_batch, dtype=torch.long),
        'object_mask': torch.stack(object_mask_batch),
        'scores': torch.stack(scores_batch)
    }