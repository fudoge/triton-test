# v1.0

```
SMILES Regression Toy Model for Triton Inference Server Testing

입력: SMILES string (one-hot encoded) [batch_size, seq_len, vocab_size]
출력: Regression value [batch_size, 1]

Transformer 기반 간단한 regression 모델
```

# v2.0

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