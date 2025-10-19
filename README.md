# Triton Test

[Triton Inference Server](https://github.com/triton-inference-server/server)를 테스트하기 위한 임시 레포지토리 입니다.

## 폴더 구조 관례
모델 및 `config.pbtxt`는 `/artifacts`아래와 같이 정리해주세요:
```bash
.
├── dta_model
│   ├── v1.0
│   │   └── model.pt
│   └── config.pbtxt
└── toymodel
    ├── v1.0
    │   └── model.pt
    └── config.pbtxt
```


나머지 관련 파일들은 `workspaces/모델명`으로 분리해놔주세요

## toymodel
```
SMILES Regression Toy Model for Triton Inference Server Testing

입력: SMILES string (one-hot encoded) [batch_size, seq_len, vocab_size]
출력: Regression value [batch_size, 1]

Transformer 기반 간단한 regression 모델
```

## dta_model

```
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
```

## multi-label model

```
SMILES and FASTA sequence based Multi-Output DTA Model

입력: 
1. SMILES string (one-hot encoded) [batch_size, smiles_seq_len, smiles_vocab_size]
2. FASTA sequence (one-hot encoded) [batch_size, fasta_seq_len, fasta_vocab_size]

출력: 2 Regression values [batch_size, 2]

Architecture:
1. SMILES -> Transformer Encoder -> Drug Feature Vector
2. FASTA   -> Transformer Encoder -> Protein Feature Vector
3. Concat [Drug Feature, Protein Feature]
4. 3-Layer MLP -> 2 Regression Outputs
```