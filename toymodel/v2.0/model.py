"""
SMILES and FASTA sequence based Drug-Target Affinity (DTA) Toy Model

입력: 
1. SMILES string (one-hot encoded) [batch_size, smiles_seq_len, smiles_vocab_size]
2. FASTA sequence (one-hot encoded) [batch_size, fasta_seq_len, fasta_vocab_size]

출력: Regression value (binding affinity) [batch_size, 1]

Architecture:
1. SMILES -> Transformer Encoder -> Drug Feature Vector
2. FASTA   -> Transformer Encoder -> Protein Feature Vector
3. Concat [Drug Feature, Protein Feature]
4. 3-Layer MLP -> Regression Output
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Transformer용 Positional Encoding"""
    
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SequenceTransformerEncoder(nn.Module):
    """
    One-hot encoded sequence를 입력받아 feature vector를 출력하는 공용 Transformer Encoder
    SMILES (drug)와 FASTA (protein) 처리에 모두 사용됩니다.
    """
    def __init__(
        self, 
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=1024
    ):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Global Average Pooling
        x = x.mean(dim=1)
        return x


class DrugTargetAffinityModel(nn.Module):
    """
    SMILES와 FASTA 입력을 받아 결합 친화도를 예측하는 DTA 모델
    """
    def __init__(
        self,
        smiles_vocab_size,
        fasta_vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_smiles_len=512,
        max_fasta_len=1024
    ):
        super().__init__()
        
        # 1. SMILES Encoder
        self.smiles_encoder = SequenceTransformerEncoder(
            vocab_size=smiles_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_smiles_len
        )
        
        # 2. FASTA (Protein) Encoder
        self.fasta_encoder = SequenceTransformerEncoder(
            vocab_size=fasta_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_fasta_len
        )
        
        # 3. Regression Head (3-layer MLP)
        # Concatenated feature vector size = d_model (smiles) + d_model (fasta)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
            # DTA에서는 보통 Sigmoid를 사용하지 않으므로 마지막 활성화 함수 제거
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, smiles_input, fasta_input):
        """
        Args:
            smiles_input: one-hot encoded SMILES [batch, smiles_len, smiles_vocab]
            fasta_input: one-hot encoded FASTA [batch, fasta_len, fasta_vocab]
        
        Returns:
            output: regression value (affinity) [batch, 1]
        """
        # Feature vector 추출
        smiles_feat = self.smiles_encoder(smiles_input) # [batch, d_model]
        fasta_feat = self.fasta_encoder(fasta_input)   # [batch, d_model]
        
        # 두 feature vector를 concat
        combined_feat = torch.cat([smiles_feat, fasta_feat], dim=1) # [batch, d_model * 2]
        
        # 최종 예측
        output = self.regression_head(combined_feat) # [batch, 1]
        
        return output


def create_dta_toy_model(smiles_vocab_size, fasta_vocab_size):
    """
    Triton 테스트용 DTA 토이 모델 생성
    """
    model = DrugTargetAffinityModel(
        smiles_vocab_size=smiles_vocab_size,
        fasta_vocab_size=fasta_vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_smiles_len=512,
        max_fasta_len=1024
    )
    return model
