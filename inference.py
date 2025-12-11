import torch
import numpy as np
from transformers import BertTokenizer
from model import ScoringModel
import json

CONFIG = {
    'hidden_size': 128,
    'max_steps': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'scoring_model.pt',
    'bert_model': 'bert-base-uncased'
}

class ScoringPredictor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(CONFIG['bert_model'])
        self.model = ScoringModel(
            hidden_size=CONFIG['hidden_size'],
            max_steps=CONFIG['max_steps']
        ).to(CONFIG['device'])
        self.model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        self.model.eval()
    
    def preprocess(self, standard, obj):
        """预处理单个样本"""
        # 编码standard
        standard_tokens = self.tokenizer(
            standard,
            max_length=100,
            truncation=True,
            add_special_tokens=False
        )
        
        # 编码object (固定100 tokens)
        object_tokens = self.tokenizer(
            obj,
            max_length=CONFIG['max_steps'],
            truncation=True,
            padding='max_length',
            add_special_tokens=False
        )
        
        # 拼接输入
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
        
        # 截断到512
        input_ids = input_ids[:512]
        attention_mask = attention_mask[:512]
        
        # 计算object起始位置
        object_start = 1 + len(standard_tokens['input_ids']) + 1
        
        # 创建object mask
        object_mask = torch.tensor(object_tokens['attention_mask'][:CONFIG['max_steps']])
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'object_start': object_start,
            'object_mask': object_mask
        }
    
    def predict(self, standard, obj, return_steps=False):
        """预测评分"""
        # 预处理
        sample = self.preprocess(standard, obj)
        
        # 添加batch维度
        input_ids = sample['input_ids'].unsqueeze(0).to(CONFIG['device'])
        attention_mask = sample['attention_mask'].unsqueeze(0).to(CONFIG['device'])
        object_start = torch.tensor([sample['object_start']], device=CONFIG['device'])
        object_mask = sample['object_mask'].unsqueeze(0).to(CONFIG['device'])
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, object_start, object_mask)
        
        # 获取结果
        final_score = outputs['final_score'].item()
        
        if return_steps:
            # 返回每一步的评分和修正量
            all_scores = outputs['all_scores'][0].cpu().numpy()
            deltas = outputs['deltas'][0].cpu().numpy()
            
            steps = []
            for step in range(CONFIG['max_steps']):
                steps.append({
                    'step': step + 1,
                    'cumulative_score': float(all_scores[step]),
                    'delta': float(deltas[step])
                })
            
            return {
                'final_score': final_score,
                'steps': steps
            }
        
        return {'final_score': final_score}

# 使用示例
if __name__ == "__main__":
    predictor = ScoringPredictor()
    
    # 获取用户输入
    print("请输入评分标准和待评分对象:")
    standard = input("评分标准 (standard): ")
    obj = input("待评分对象 (obj): ")
    
    # 进行预测
    try:
        result = predictor.predict(
            standard=standard,
            obj=obj,
            return_steps=True
        )
        
        # 输出最终分数
        print(f"\n最终评分: {result['final_score']:.4f}")
        
        # 保存步骤数据用于可视化
        output_filename = 'scoring_steps.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"详细步骤已保存到 {output_filename}")
        
    except Exception as e:
        print(f"推理过程中发生错误: {e}", file=sys.stderr)
        sys.exit(1)