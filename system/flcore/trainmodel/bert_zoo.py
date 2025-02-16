from transformers import (
    DistilBertConfig, DistilBertModel,
    BertConfig, BertModel,
    AlbertConfig, AlbertModel
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# class Bertzoo(nn.Module):
#     def __init__(self, model_name, vocab_size=98635, hidden_dim=512, num_layers=6, num_heads=12, ff_dim=3072, 
#                  num_classes=10, dropout=0.1):
#         super(Bertzoo, self).__init__()

#         if model_name == 'distilbert':
#             # DistilBERT 配置
#             config = DistilBertConfig(
#                 vocab_size=vocab_size,   
#                 hidden_dim=hidden_dim,   
#                 n_layers=num_layers,    
#                 n_heads=num_heads,      
#                 intermediate_size=ff_dim, 
#                 dropout=dropout,
#                 num_labels=num_classes  # 设置分类标签数
#             )
#             # 初始化 DistilBERT（不加载预训练权重）
#             self.bert = DistilBertForSequenceClassification(config)

#         elif model_name == 'bert':
#             # BERT 配置
#             config = BertConfig(
#                 vocab_size=vocab_size,   
#                 hidden_size=hidden_dim,   
#                 num_hidden_layers=num_layers,    
#                 num_attention_heads=num_heads,      
#                 intermediate_size=ff_dim, 
#                 hidden_dropout_prob=dropout,
#                 num_labels=num_classes  # 设置分类标签数
#             )
#             # 初始化 BERT（不加载预训练权重）
#             self.bert = BertForSequenceClassification(config)

#         elif model_name == 'roberta':
#             # RoBERTa 配置
#             config = RobertaConfig(
#                 vocab_size=vocab_size,   
#                 hidden_size=hidden_dim,   
#                 num_hidden_layers=num_layers,    
#                 num_attention_heads=num_heads,      
#                 intermediate_size=ff_dim, 
#                 hidden_dropout_prob=dropout,
#                 num_labels=num_classes  # 设置分类标签数
#             )
#             # 初始化 RoBERTa（不加载预训练权重）
#             self.bert = RobertaForSequenceClassification(config)

#         elif model_name == 'albert':
#             # ALBERT 配置
#             config = AlbertConfig(
#                 vocab_size=vocab_size,   
#                 hidden_size=hidden_dim,   
#                 num_hidden_layers=num_layers,    
#                 num_attention_heads=num_heads,      
#                 intermediate_size=ff_dim, 
#                 hidden_dropout_prob=dropout,
#                 num_labels=num_classes  # 设置分类标签数
#             )
#             # 初始化 ALBERT（不加载预训练权重）
#             self.bert = AlbertForSequenceClassification(config)

#         elif model_name == 'tinybert':
#             # TinyBERT 配置（通常tinybert具有更小的hidden_size）
#             config = BertConfig(
#                 vocab_size=vocab_size,   
#                 hidden_size=256,  # TinyBERT 更小的 hidden_size
#                 num_hidden_layers=4,    
#                 num_attention_heads=4,      
#                 intermediate_size=1024, 
#                 hidden_dropout_prob=dropout,
#                 num_labels=num_classes  # 设置分类标签数
#             )
#             # 初始化 TinyBERT（不加载预训练权重）
#             self.bert = BertForSequenceClassification(config)

#         elif model_name == 'camembert':
#             # CamemBERT 配置（针对法语）
#             config = RobertaConfig(
#                 vocab_size=vocab_size,   
#                 hidden_size=hidden_dim,   
#                 num_hidden_layers=num_layers,    
#                 num_attention_heads=num_heads,      
#                 intermediate_size=ff_dim, 
#                 hidden_dropout_prob=dropout,
#                 num_labels=num_classes  # 设置分类标签数
#             )
#             # 初始化 CamemBERT（不加载预训练权重）
#             self.bert = RobertaForSequenceClassification(config)

#         elif model_name == 'longformer':
#             # Longformer 配置（处理长序列）
#             config = RobertaConfig(
#                 vocab_size=vocab_size,   
#                 hidden_size=hidden_dim,   
#                 num_hidden_layers=num_layers,    
#                 num_attention_heads=num_heads,      
#                 intermediate_size=ff_dim, 
#                 hidden_dropout_prob=dropout,
#                 num_labels=num_classes  # 设置分类标签数
#             )
#             # 初始化 Longformer（不加载预训练权重）
#             self.bert = RobertaForSequenceClassification(config)

#         else:
#             raise ValueError("Unsupported model_name. Choose from 'distilbert', 'bert', 'roberta', 'albert', 'tinybert', 'camembert', or 'longformer'.")

#     def forward(self, x):
#         if isinstance(x, list):  # 兼容 (text, text_length) 这种格式
#             text, _ = x
#         else:
#             text = x

#         # 获取输出 logits
#         out = self.bert(input_ids=text).logits
        
#         # 返回 log softmax 结果
#         out = F.log_softmax(out, dim=1)
#         return out


class Bertzoo(nn.Module):
    def __init__(self, model_name, vocab_size=98635, hidden_dim=512, num_layers=6, num_heads=12, ff_dim=3072,
                 num_classes=10, dropout=0.1):
        super(Bertzoo, self).__init__()

        # 选择模型类型并加载相应的模型 backbone
        if model_name == 'distilbert':
            # DistilBERT 配置
            config = DistilBertConfig(
                vocab_size=vocab_size
            )
            # 只加载模型 backbone
            self.bert = DistilBertModel(config)

        elif model_name == 'bert':
            # BERT 配置
            config = BertConfig(
                vocab_size=vocab_size
            )
            # 只加载模型 backbone
            self.bert = BertModel(config)

        elif model_name == 'albert':
            # ALBERT 配置
            config = AlbertConfig(
                vocab_size=vocab_size
            )
            # 只加载模型 backbone
            self.bert = AlbertModel(config)

        elif model_name == 'tinybert':
            # TinyBERT 配置（通常tinybert具有更小的hidden_size）
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=256,  # TinyBERT 更小的 hidden_size
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                hidden_dropout_prob=dropout,
            )
            # 只加载模型 backbone
            self.bert = BertModel(config)

        else:
            raise ValueError("Unsupported model_name. Choose from 'distilbert', 'bert', 'albert', 'tinybert'.")

        # 添加分类头（self.fc）
        self.fc = nn.Linear(config.hidden_size, num_classes)  # 分类头

    def forward(self, x):
        if isinstance(x, list):  # 兼容 (text, text_length) 这种格式
            text, _ = x
        else:
            text = x

        # 获取 backbone 输出
        out = self.bert(input_ids=text).last_hidden_state[:, 0, :]  # 获取 [CLS] 位置的向量

        # 通过分类器层
        out = self.fc(out)

        # 返回 log softmax 结果
        out = F.log_softmax(out, dim=1)
        return out


if __name__ == "__main__":
    for name in ['distilbert', 'bert', 'albert', 'tinybert']:
        model = Bertzoo(name)
        print(name)
        print(model)