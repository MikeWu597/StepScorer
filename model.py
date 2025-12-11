import torch
import torch.nn as nn
from transformers import BertModel

class ScoringModel(nn.Module):
    def __init__(self, hidden_size=128, bert_model_name='bert-base-uncased', max_steps=100):
        super().__init__()
        self.max_steps = max_steps
        
        # 冻结BERT参数
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 评分演化模块
        self.gru = nn.GRU(
            input_size=768,  # BERT hidden size
            hidden_size=hidden_size,
            batch_first=True
        )
        
        self.delta_predictor = nn.Sequential(
            nn.Linear(hidden_size + 768, 64),  # +768 for current token embedding
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 标准文本的全局表示投影层
        self.standard_proj = nn.Linear(768, hidden_size)
    
    def forward(self, input_ids, attention_mask, object_start, object_mask):
        batch_size = input_ids.size(0)
        
        # 冻结BERT的前向传播
        with torch.no_grad():
            bert_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            all_embeddings = bert_output.last_hidden_state  # [batch, seq_len, 768]
        
        # 提取标准文本的[CLS]表示 (位置0)
        standard_repr = all_embeddings[:, 0, :]  # [batch, 768]
        standard_ctx = self.standard_proj(standard_repr)  # [batch, hidden_size]
        
        # 提取object部分的100个token表示
        object_embeddings = []
        for i in range(batch_size):
            start = object_start[i].item()
            end = min(start + self.max_steps, all_embeddings.size(1))
            obj_emb = all_embeddings[i, start:end, :]
            
            # 如果不足100个token，用零向量填充
            if obj_emb.size(0) < self.max_steps:
                padding = torch.zeros(
                    self.max_steps - obj_emb.size(0), 
                    768, 
                    device=obj_emb.device
                )
                obj_emb = torch.cat([obj_emb, padding], dim=0)
            object_embeddings.append(obj_emb[:self.max_steps])  # 严格取100个
        
        object_embeddings = torch.stack(object_embeddings)  # [batch, 100, 768]
        
        # 初始化GRU状态 (用标准文本表示初始化)
        h0 = standard_ctx.unsqueeze(0)  # [1, batch, hidden_size]
        
        # GRU处理object序列
        gru_output, _ = self.gru(object_embeddings, h0)  # [batch, 100, hidden_size]
        
        # 预测每一步的修正量
        combined = torch.cat([gru_output, object_embeddings], dim=-1)  # [batch, 100, hidden_size+768]
        deltas = self.delta_predictor(combined).squeeze(-1)  # [batch, 100]
        
        # 应用object_mask: 将padding位置的修正量置零
        deltas = deltas * object_mask  # [batch, 100]
        
        # 累计评分 (从0开始)
        scores = torch.cumsum(deltas, dim=1)  # [batch, 100]
        final_score = scores[:, -1]  # 取第100步的评分
        
        # 返回所有中间步骤（用于可视化）
        return {
            'final_score': final_score,
            'all_scores': scores,
            'deltas': deltas
        }