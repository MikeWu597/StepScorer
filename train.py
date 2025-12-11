import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import ScoringModel
from dataset import ScoringDataset, collate_fn

# 配置参数
CONFIG = {
    'batch_size': 16,
    'epochs': 20,
    'lr': 0.001,
    'hidden_size': 128,
    'max_steps': 100,
    'lambda_reg': 0.05,  # 时间加权正则化系数
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_path': 'data/sample_data.csv',
    'model_save_path': 'scoring_model.pt'
}

def train():
    # 1. 准备数据
    dataset = ScoringDataset(CONFIG['data_path'], max_object_tokens=CONFIG['max_steps'])
    
    # 划分训练/验证集
    train_idx, val_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=42
    )
    
    train_loader = DataLoader(
        [dataset[i] for i in train_idx],
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        [dataset[i] for i in val_idx],
        batch_size=CONFIG['batch_size'],
        collate_fn=collate_fn
    )
    
    # 2. 初始化模型
    model = ScoringModel(
        hidden_size=CONFIG['hidden_size'],
        max_steps=CONFIG['max_steps']
    ).to(CONFIG['device'])
    
    # 3. 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化非BERT参数
        lr=CONFIG['lr']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 4. 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["epochs"]}'):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            object_start = batch['object_start'].to(CONFIG['device'])
            object_mask = batch['object_mask'].to(CONFIG['device'])
            scores = batch['scores'].to(CONFIG['device'])
            
            # 前向传播
            outputs = model(input_ids, attention_mask, object_start, object_mask)
            pred_scores = outputs['final_score']
            
            # 计算损失
            loss = criterion(pred_scores, scores)
            
            # 时间加权正则化: 鼓励早期修正
            deltas = outputs['deltas']  # [batch, 100]
            time_weights = torch.arange(1, CONFIG['max_steps'] + 1, device=CONFIG['device']).float()
            reg_loss = (time_weights * deltas.abs()).mean()
            total_loss = loss + CONFIG['lambda_reg'] * reg_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防止梯度爆炸
            optimizer.step()
            
            train_loss += total_loss.item() * input_ids.size(0)
        
        # 验证集评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(CONFIG['device'])
                attention_mask = batch['attention_mask'].to(CONFIG['device'])
                object_start = batch['object_start'].to(CONFIG['device'])
                object_mask = batch['object_mask'].to(CONFIG['device'])
                scores = batch['scores'].to(CONFIG['device'])
                
                outputs = model(input_ids, attention_mask, object_start, object_mask)
                pred_scores = outputs['final_score']
                loss = criterion(pred_scores, scores)
                val_loss += loss.item() * input_ids.size(0)
        
        # 计算平均损失
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"Saved best model with val loss: {val_loss:.4f}")

if __name__ == "__main__":
    train()