# **MedicalGPT: Custom Transformer Decoder for Clinical Response Generation**

---

## **Project Overview**
MedicalGPT is a custom **Decoder-only Transformer Language Model** built entirely from scratch using PyTorch. The model is specifically designed and trained for **single turn patient symptom to clinical response generation**. 

Rather than relying on generic pre-trained LLMs, this project implements the full pipeline from medical domain text tokenization to autoregressive transformer training and evaluation culminating in an interactive Streamlit dashboard.

---

## **Key Features**
1. **Decoder-Only Transformer Architecture**: Custom implementation of embeddings, positional encoding, multi-head causal self-attention, feed-forward networks, and layer normalization in PyTorch.
2. **Medical SentencePiece BPE Tokenizer**: A custom Byte Pair Encoding (BPE) tokenizer trained on domain specific medical corpora, achieving a compact and efficient vocabulary size of $8,000$.
3. **Clinical Conversation dataset**: Trained and validated on the **HealthCareMagic** dataset containing over **101,000** patient doctor dialogue sequences.
4. **Hyperparameter Tuning & Evaluation Dashboard**: Interactive Streamlit application to compare performance metrics (Perplexity, BLEU, and ROUGE scores) across different architecture dimensions and test symptom queries in real time.

---

## **Mathematical Model & Architecture**

The network is modeled as a causal, autoregressive GPT-style decoder.

### **1. Token and Positional Embeddings**
Given an input sequence of token IDs $X = [x_1, x_2, \dots, x_T]$, we embed them into a continuous space:
$$E_{tok} = \text{Embedding}(X) \cdot \sqrt{d_{model}}$$
Since Transformers do not have recurrence, we inject absolute spatial configurations via a learnable positional embedding matrix $E_{pos}$:
$$E = E_{tok} + E_{pos}$$

### **2. Causal Multi-Head Attention (MHA)**
For each head $h$, the query ($Q$), key ($K$), and value ($V$) vectors are projected from the input hidden states:
$$Q = X W_q, \quad K = X W_k, \quad V = X W_v$$
Causal relation is maintained by masking future tokens using a lower-triangular matrix, ensuring output position $t$ depends only on inputs $\le t$:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_{head}}} + M \right) V$$
where the mask $M_{ij} = 0$ if $i \ge j$, and $-\infty$ otherwise. Multi-Head outputs are concatenated and projected:
$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_o$$

### **3. Feed-Forward Network (FFN)**
A position-wise network with a GELU activation function:
$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

### **4. Decoder Layer ResNet Skip-Connections**
Pre-Layer Normalization is used for training stability:
$$x^{(1)} = x + \text{MHA}(\text{LayerNorm}(x))$$
$$x^{(2)} = x^{(1)} + \text{FFN}(\text{LayerNorm}(x^{(1)}))$$

---

## **Hyperparameter & Experiment Sweeps**

We conducted 4 distinct training sweeps to evaluate performance improvements as capacity increased:

| Experiment Name | Layers | Heads | $d_{model}$ | $d_{ffn}$ | Test Loss | Perplexity ↓ | BLEU % ↑ | ROUGE-1 % ↑ | ROUGE-2 % ↑ | ROUGE-L % ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **exp1_heads4_layers6** | 6 | 4 | 256 | 1024 | 3.2323 | 25.34 | 2.7105 | 22.9796 | 4.0597 | 14.3600 |
| **exp2_heads4_layers8** | 8 | 4 | 256 | 1024 | 3.1642 | 23.67 | 2.8066 | 23.0615 | 4.0799 | 14.4144 |
| **exp3_d512_heads4_layers8** | 8 | 4 | 512 | 2048 | **2.9217** | **18.57** | **3.1699** | 23.4861 | 4.3753 | 14.7120 |
| **exp4_d512_heads2_layers8** | 8 | 2 | 512 | 2048 | 2.9507 | 19.12 | 3.1682 | **23.5538** | **4.4419** | **14.7713** |

---

## **Project Structure**
<pre>
MedicalGPT/
│── notebooks/
│   ├── step1_data_cleaning.ipynb      # Filtering and preparation of dialogue corpus
│   ├── step2_tokenizer.ipynb          # SentencePiece training and vocabulary export
│   ├── step3_prepare_sequences.ipynb  # Sequence padding and token conversion
│   ├── step4_model.ipynb              # Causal self-attention model implementation
│   ├── step5_train.ipynb              # PyTorch training run notebook
│   ├── step5_train.py                 # PyTorch training script (nohup ready)
│   ├── step6.ipynb                    # Evaluation (Perplexity, BLEU, ROUGE)
│── tokenizer/
│   ├── config.json                    # Vocabulary config parameters
│   ├── medical_bpe.model              # Trained SentencePiece model binary
│   ├── medical_bpe.vocab              # BPE token vocabulary map
│── experiments/                       # Output checkpoints and performance metrics
│   ├── exp1_heads4_layers6/
│   ├── exp2_heads4_layers8/
│   ├── exp3_d512_heads4_layers8/
│   ├── exp4_d512_heads2_layers8/
│── ui/
│   ├── app.py                         # Streamlit user interface dashboard
│── requirements.txt                   # Dependency file
│── .gitignore
│── README.md                          # Documentation
</pre>

---

## **Installation & Setup**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Launch the Streamlit Dashboard**
Start the clinical consultation interface to run inference on the trained checkpoints:
```bash
streamlit run ui/app.py
```

### **3. Run Step-by-Step Training Pipelines**
To re-train the models or run a new experiment:
1. Set up the dataset in `notebooks/step1_data_cleaning.ipynb`.
2. Train the SentencePiece BPE tokenizer using `notebooks/step2_tokenizer.ipynb`.
3. Pre-process sequences and generate token arrays using `notebooks/step3_prepare_sequences.ipynb`.
4. Configure your experimental hyperparameters inside `notebooks/step5_train.py` and run:
   ```bash
   python notebooks/step5_train.py
   ```
5. Evaluate performance metrics using `notebooks/step6.ipynb`.
