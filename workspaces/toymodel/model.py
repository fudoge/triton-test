"""
SMILES Regression Toy Model for Triton Inference Server Testing

입력: SMILES string (one-hot encoded) [batch_size, seq_len, vocab_size]
출력: Regression value [batch_size, 1]

Transformer 기반 간단한 regression 모델
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Transformer용 Positional Encoding"""
    
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Positional encoding 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SMILESTransformerRegression(nn.Module):
    """
    SMILES one-hot encoding을 입력받아 실수 값을 예측하는 Transformer 모델
    
    Architecture:
    1. Input: one-hot encoded SMILES [batch, seq_len, vocab_size]
    2. Embedding: Linear projection to d_model
    3. Positional Encoding
    4. Transformer Encoder
    5. Global Average Pooling
    6. Regression Head: Linear -> ReLU -> Linear -> Sigmoid
    7. Output: [batch, 1] (0~1 사이의 실수)
    """
    
    def __init__(
        self, 
        vocab_size=101,      # vocab_chars.json 크기
        d_model=128,         # Transformer hidden dimension
        nhead=4,             # Attention heads
        num_layers=2,        # Transformer encoder layers
        dim_feedforward=256, # FFN dimension
        dropout=0.1,
        max_seq_len=512
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 1. One-hot embedding to d_model
        self.embedding = nn.Linear(vocab_size, d_model)
        
        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # [batch, seq, feature] 형태 사용
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 4. Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # 0~1 사이의 값 출력
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: one-hot encoded SMILES [batch_size, seq_len, vocab_size]
        
        Returns:
            output: regression value [batch_size, 1]
        """
        # 1. Embedding
        x = self.embedding(x)  # [batch, seq_len, d_model]
        
        # 2. Positional encoding
        x = self.pos_encoder(x)  # [batch, seq_len, d_model]
        
        # 3. Transformer encoding
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        
        # 4. Global average pooling (sequence dimension)
        x = x.mean(dim=1)  # [batch, d_model]
        
        # 5. Regression head
        output = self.regression_head(x)  # [batch, 1]
        
        return output


def create_toy_model(vocab_size=101):
    model = SMILESTransformerRegression(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=512
    )
    return model
