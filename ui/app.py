import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import json, math, os, time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MedicalGPT",
    page_icon="🩺",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0a0e1a;
    --surface: #111827;
    --surface2:#1a2235;
    --border:  #1f2d45;
    --accent:  #00c9a7;
    --accent2: #3b82f6;
    --warn:    #f59e0b;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --mono:    'Space Mono', monospace;
    --sans:    'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: var(--sans);
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Sidebar headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--mono);
    font-size: 0.85rem !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin-bottom: 0.5rem;
}

/* Metric cards */
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.metric-label {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.metric-value { font-family: var(--mono); font-size: 1rem; color: var(--accent); font-weight: 700; }
.metric-value.blue  { color: var(--accent2); }
.metric-value.amber { color: var(--warn); }

/* Info rows */
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
}
.info-key { color: var(--muted); font-family: var(--mono); font-size: 0.8rem; }
.info-val { color: var(--text); font-weight: 500; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}

/* Chat input */
[data-testid="stChatInputTextArea"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
    font-family: var(--sans) !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,201,167,0.15) !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
    background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 16px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

hr { border-color: var(--border) !important; }

/* Title */
.main-title {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
    margin-bottom: 2px;
}
.main-subtitle {
    font-family: var(--sans);
    font-size: 0.95rem;
    color: var(--muted);
    margin-bottom: 1rem;
}
.accent-dot { color: var(--accent); }

/* Notes box */
.notes-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.7;
    word-break: break-all;
}

/* Empty state */
.empty-state { text-align: center; padding: 60px 20px; color: var(--muted); }
.empty-icon  { font-size: 3rem; margin-bottom: 12px; }
.empty-text  { font-family: var(--sans); font-size: 0.95rem; }

/* Stat pill */
.stat-pill {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    margin-right: 6px;
    margin-top: 6px;
}
.stat-pill span { color: var(--accent); font-weight: 700; }

/* Best badge */
.best-badge {
    display: inline-block;
    background: rgba(0,201,167,0.15);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--mono);
    font-size: 0.6rem;
    padding: 1px 7px;
    border-radius: 20px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-left: 6px;
    vertical-align: middle;
}

/* Section label */
.section-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
    margin-top: 20px;
}

/* Architecture box */
.arch-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--text);
    margin-bottom: 8px;
    position: relative;
}
.arch-box.accent-border { border-color: var(--accent); border-width: 1.5px; }
.arch-label { font-size: 0.65rem; color: var(--muted); margin-top: 4px; }
.arch-arrow { text-align: center; color: var(--accent); font-size: 1.2rem; margin: 2px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, pad_id, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.dropout   = nn.Dropout(dropout)
        self.d_model   = d_model
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(self.token_emb(x) * math.sqrt(self.d_model) + self.pos_emb(positions))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.n_heads = n_heads; self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model); self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model); self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, causal_mask=None, pad_mask=None):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        K = self.W_k(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        V = self.W_v(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_head)
        if causal_mask is not None: scores = scores.masked_fill(causal_mask==0, float('-inf'))
        if pad_mask   is not None: scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2)==0, float('-inf'))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        return self.W_o(self.dropout(attn).matmul(V).transpose(1,2).contiguous().view(B,T,self.d_model))

class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model,ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim,d_model), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn=MultiHeadAttention(d_model,n_heads,dropout); self.ffn=FeedForward(d_model,ffn_dim,dropout)
        self.norm1=nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model); self.dropout=nn.Dropout(dropout)
    def forward(self, x, causal_mask=None, pad_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), causal_mask, pad_mask))
        return x + self.ffn(self.norm2(x))

class MedicalGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ffn_dim, max_seq_len, pad_id, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.embeddings = Embeddings(vocab_size, d_model, max_seq_len, pad_id, dropout)
        self.layers  = nn.ModuleList([DecoderLayer(d_model,n_heads,ffn_dim,dropout) for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_emb.weight
    def make_causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len,seq_len,device=device)).unsqueeze(0).unsqueeze(0)
    def forward(self, input_ids, attention_mask=None):
        B,T = input_ids.shape
        x   = self.embeddings(input_ids)
        cm  = self.make_causal_mask(T, input_ids.device)
        for layer in self.layers: x = layer(x, causal_mask=cm, pad_mask=attention_mask)
        return self.lm_head(self.norm(x))

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR       = '/home/sem4/NLP/NLP'
TOKENIZER_PATH = f'{BASE_DIR}/tokenizer/medical_bpe.model'
EXP_DIR        = f'{BASE_DIR}/experiments'

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
BEST_EXP = "exp3_d512_heads4_layers8"

def get_experiments():
    exps = []
    if os.path.exists(EXP_DIR):
        for name in sorted(os.listdir(EXP_DIR)):
            if os.path.exists(f'{EXP_DIR}/{name}/checkpoints/best_model.pt') and \
               os.path.exists(f'{EXP_DIR}/{name}/config.json'):
                exps.append(name)
    return exps

@st.cache_resource
def load_model(exp_name):
    config_path = f'{EXP_DIR}/{exp_name}/config.json'
    model_path  = f'{EXP_DIR}/{exp_name}/checkpoints/best_model.pt'
    with open(config_path) as f:
        config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedicalGPT(
        vocab_size=config['vocab_size'], d_model=config['d_model'],
        n_heads=config['n_heads'],       n_layers=config['n_layers'],
        ffn_dim=config['ffn_dim'],       max_seq_len=config['max_seq_len'],
        pad_id=config['pad_id'],
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    return model, sp, config, device

def generate(model, sp, config, device, symptom_text, max_new_tokens=150, temperature=0.7, top_k=50):
    EOS_ID, MAX_SEQ_LEN = config['eos_id'], config['max_seq_len']
    input_ids    = sp.Encode(f"<patient> {symptom_text} <doctor>", out_type=int)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.size(1) >= MAX_SEQ_LEN: break
            logits      = model(input_tensor)
            next_logits = logits[0, -1, :] / temperature
            top_k_vals, top_k_idx = torch.topk(next_logits, top_k)
            probs       = torch.softmax(top_k_vals, dim=-1)
            next_token  = top_k_idx[torch.multinomial(probs, 1)].view(1,1)
            if next_token.item() == EOS_ID: break
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
    elapsed = time.time() - t0
    out = sp.Decode(input_tensor[0].tolist())
    if '<doctor>' in out: out = out.split('<doctor>')[-1].strip()
    tokens_generated = input_tensor.size(1) - len(input_ids)
    return out, elapsed, tokens_generated

def metric_card(label, value, color=""):
    return f"""<div class="metric-card">
        <span class="metric-label">{label}</span>
        <span class="metric-value {color}">{value}</span>
    </div>"""

def info_row(key, val):
    return f"""<div class="info-row">
        <span class="info-key">{key}</span>
        <span class="info-val">{val}</span>
    </div>"""

def load_all_eval_results():
    """Load eval results from all experiments."""
    results = {}
    for name in get_experiments():
        path = f"{EXP_DIR}/{name}/eval_results.json"
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
    return results

def plotly_dark_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,34,53,0.6)',
        font=dict(family='Space Mono, monospace', color='#e2e8f0', size=11),
        xaxis=dict(gridcolor='#1f2d45', linecolor='#1f2d45', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#1f2d45', linecolor='#1f2d45', tickfont=dict(size=10)),
        legend=dict(bgcolor='rgba(17,24,39,0.8)', bordercolor='#1f2d45', borderwidth=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 MedicalGPT")
    st.markdown('<p style="font-size:0.82rem;color:#64748b;margin-top:-8px;font-family:\'Space Mono\',monospace;">Single-Turn Medical Response Generator</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("#### Select Experiment")
    experiments = get_experiments()
    if not experiments:
        st.error("No experiments found.")
        st.stop()

    # Add best badge to dropdown label
    def exp_label(name):
        return f"★ {name}" if name == BEST_EXP else name

    selected_exp = st.selectbox("", experiments,
        format_func=exp_label,
        label_visibility="collapsed")

    st.divider()

    with st.spinner("Loading weights..."):
        model, sp, config, device = load_model(selected_exp)

    total_params = sum(p.numel() for p in model.parameters())

    # Architecture info
    st.markdown("#### Architecture")
    arch_html = "".join([
        info_row("d_model",    str(config['d_model'])),
        info_row("n_heads",    str(config['n_heads'])),
        info_row("n_layers",   str(config['n_layers'])),
        info_row("ffn_dim",    str(config['ffn_dim'])),
        info_row("vocab_size", str(config['vocab_size'])),
        info_row("Parameters", f"{total_params:,}"),
        info_row("Device",     str(device).upper()),
    ])
    st.markdown(arch_html, unsafe_allow_html=True)
    st.divider()

    # Eval results
    eval_path = f"{EXP_DIR}/{selected_exp}/eval_results.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            ev = json.load(f)
        st.markdown("#### Eval Results")
        ppl  = ev.get('test_perplexity', ev.get('perplexity', None))
        bleu = ev.get('bleu', None)
        r1   = ev.get('rouge1', ev.get('rouge_1', ev.get('ROUGE-1', None)))
        r2   = ev.get('rouge2', ev.get('rouge_2', ev.get('ROUGE-2', None)))
        rl   = ev.get('rougeL', ev.get('rouge_L', ev.get('ROUGE-L', None)))
        cards = ""
        if ppl  is not None: cards += metric_card("Perplexity ↓", f"{float(ppl):.2f}")
        if bleu is not None: cards += metric_card("BLEU %",       f"{float(bleu):.4f}", "blue")
        if r1   is not None: cards += metric_card("ROUGE-1 %",    f"{float(r1):.4f}",  "blue")
        if r2   is not None: cards += metric_card("ROUGE-2 %",    f"{float(r2):.4f}",  "blue")
        if rl   is not None: cards += metric_card("ROUGE-L %",    f"{float(rl):.4f}",  "blue")
        if cards: st.markdown(cards, unsafe_allow_html=True)
        st.divider()

    # Notes
    notes_path = f"{EXP_DIR}/{selected_exp}/notes.txt"
    if os.path.exists(notes_path):
        with open(notes_path) as f:
            notes = f.read().strip()
        st.markdown("#### Notes")
        st.markdown(f'<div class="notes-box">{notes}</div>', unsafe_allow_html=True)
        st.divider()

    # Generation settings
    st.markdown("#### Generation")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1,
                            help="Higher = more creative, Lower = more focused")
    top_k       = st.slider("Top-K",       10,  100, 50,  10,
                            help="Number of top tokens to sample from")
    max_tokens  = st.slider("Max Tokens",  50,  300, 150, 25)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────
best_html = '<span class="best-badge">★ best</span>' if selected_exp == BEST_EXP else ''
st.markdown(f"""
<div class="main-title">Medical<span class="accent-dot">GPT</span></div>
<div class="main-subtitle">
    Single-turn symptom-to-response generation · Not a conversational agent
    &nbsp;·&nbsp; <strong>{selected_exp}</strong>{best_html}
    &nbsp;·&nbsp; {total_params:,} params &nbsp;·&nbsp; {str(device).upper()}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬  CHAT",
    "📈  TRAINING CURVES",
    "📊  EXPERIMENT COMPARISON",
    "🧠  ARCHITECTURE"
])

# ══════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🩺</div>
            <div class="empty-text">Describe your symptoms below to get started</div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🩺"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                st.markdown(
                    f'<div style="margin-top:6px">'
                    f'<span class="stat-pill">⏱ <span>{meta["time"]:.2f}s</span></span>'
                    f'<span class="stat-pill">🔤 <span>{meta["tokens"]} tokens</span></span>'
                    f'<span class="stat-pill">🌡 <span>temp {meta["temp"]}</span></span>'
                    f'<span class="stat-pill">🎯 <span>top-k {meta["topk"]}</span></span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    if prompt := st.chat_input("Describe your symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Analyzing symptoms..."):
                response, elapsed, n_tokens = generate(
                    model, sp, config, device, prompt, max_tokens, temperature, top_k)
            st.markdown(response)
            st.markdown(
                f'<div style="margin-top:6px">'
                f'<span class="stat-pill">⏱ <span>{elapsed:.2f}s</span></span>'
                f'<span class="stat-pill">🔤 <span>{n_tokens} tokens</span></span>'
                f'<span class="stat-pill">🌡 <span>temp {temperature}</span></span>'
                f'<span class="stat-pill">🎯 <span>top-k {top_k}</span></span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.session_state.messages.append({
            "role": "assistant", "content": response,
            "meta": {"time": elapsed, "tokens": n_tokens, "temp": temperature, "topk": top_k}
        })

# ══════════════════════════════════════════════
# TAB 2 — TRAINING CURVES
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Training Loss Curves</div>', unsafe_allow_html=True)

    loss_path = f"{EXP_DIR}/{selected_exp}/training_log.txt"
    png_path  = f"{EXP_DIR}/{selected_exp}/loss_curve.png"

    col1, col2 = st.columns([1, 1])

    # Try to read training log and plot interactively
    if os.path.exists(loss_path):
        epochs, train_losses, val_losses = [], [], []
        with open(loss_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    parts = line.split()
                    # Try common log formats
                    ep = tl = vl = None
                    for i, p in enumerate(parts):
                        if 'epoch' in p.lower() and i+1 < len(parts):
                            try: ep = int(parts[i+1].strip(':|,'))
                            except: pass
                        if 'train' in p.lower() and i+1 < len(parts):
                            try: tl = float(parts[i+1].strip(':|,'))
                            except: pass
                        if 'val' in p.lower() and i+1 < len(parts):
                            try: vl = float(parts[i+1].strip(':|,'))
                            except: pass
                    if ep is not None and tl is not None:
                        epochs.append(ep)
                        train_losses.append(tl)
                        if vl is not None: val_losses.append(vl)
                except: continue

        if epochs:
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs, y=train_losses, mode='lines+markers',
                    name='Train Loss', line=dict(color='#00c9a7', width=2),
                    marker=dict(size=5)
                ))
                if val_losses:
                    fig.add_trace(go.Scatter(
                        x=epochs, y=val_losses, mode='lines+markers',
                        name='Val Loss', line=dict(color='#3b82f6', width=2, dash='dash'),
                        marker=dict(size=5)
                    ))
                fig.update_layout(
                    title=dict(text=f"Loss — {selected_exp}", font=dict(size=12, color='#00c9a7')),
                    xaxis_title="Epoch", yaxis_title="Loss",
                    **plotly_dark_layout()
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to PNG
            with col1:
                if os.path.exists(png_path):
                    st.image(png_path, caption=f"Loss Curve — {selected_exp}", use_column_width=True)
                else:
                    st.info("No training log or loss curve found for this experiment.")
    elif os.path.exists(png_path):
        with col1:
            st.image(png_path, caption=f"Loss Curve — {selected_exp}", use_column_width=True)
    else:
        with col1:
            st.info("No training log or loss curve found for this experiment.")

    # All experiments loss curves (PNG)
    with col2:
        st.markdown('<div class="section-label">All Experiments</div>', unsafe_allow_html=True)
        for exp in get_experiments():
            p = f"{EXP_DIR}/{exp}/loss_curve.png"
            if os.path.exists(p):
                label = f"★ {exp} (best)" if exp == BEST_EXP else exp
                st.image(p, caption=label, use_column_width=True)

# ══════════════════════════════════════════════
# TAB 3 — EXPERIMENT COMPARISON
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Experiment Comparison</div>', unsafe_allow_html=True)

    all_results = load_all_eval_results()

    if not all_results:
        st.info("No eval_results.json files found.")
    else:
        # Build dataframe
        rows = []
        for exp, ev in all_results.items():
            rows.append({
                "Experiment": exp,
                "Perplexity": float(ev.get('test_perplexity', ev.get('perplexity', 0))),
                "BLEU %":     float(ev.get('bleu', 0)),
                "ROUGE-1 %":  float(ev.get('rouge1', ev.get('rouge_1', ev.get('ROUGE-1', 0)))),
                "ROUGE-2 %":  float(ev.get('rouge2', ev.get('rouge_2', ev.get('ROUGE-2', 0)))),
                "ROUGE-L %":  float(ev.get('rougeL', ev.get('rouge_L', ev.get('ROUGE-L', 0)))),
            })
        df = pd.DataFrame(rows)

        # Metric selector
        metric = st.selectbox(
            "Select Metric to Compare",
            ["Perplexity", "BLEU %", "ROUGE-1 %", "ROUGE-2 %", "ROUGE-L %"]
        )

        # Highlight best bar
        colors = []
        for exp in df["Experiment"]:
            if exp == BEST_EXP:
                colors.append("#00c9a7")
            else:
                colors.append("#3b82f6")

        fig_bar = go.Figure(go.Bar(
            x=df["Experiment"],
            y=df[metric],
            marker_color=colors,
            text=[f"{v:.4f}" for v in df[metric]],
            textposition="outside",
            textfont=dict(family="Space Mono", size=10, color="#e2e8f0"),
        ))
        note = "(lower is better)" if metric == "Perplexity" else "(higher is better)"
        fig_bar.update_layout(
            title=dict(text=f"{metric} — All Experiments {note}", font=dict(size=13, color="#00c9a7")),
            xaxis_title="Experiment",
            yaxis_title=metric,
            showlegend=False,
            **plotly_dark_layout()
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Radar chart — all metrics together
        st.markdown('<div class="section-label">Multi-Metric Radar</div>', unsafe_allow_html=True)
        metrics_radar = ["BLEU %", "ROUGE-1 %", "ROUGE-2 %", "ROUGE-L %"]
        palette = ["#00c9a7", "#3b82f6", "#f59e0b", "#ec4899"]

        fig_radar = go.Figure()
        for i, row in df.iterrows():
            vals = [row[m] for m in metrics_radar]
            vals += [vals[0]]  # close polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=metrics_radar + [metrics_radar[0]],
                fill='toself',
                name=row["Experiment"],
                line=dict(color=palette[i % len(palette)], width=2),
                fillcolor=palette[i % len(palette)].replace("#", "rgba(") + ",0.1)",
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(26,34,53,0.6)',
                radialaxis=dict(visible=True, gridcolor='#1f2d45', color='#64748b'),
                angularaxis=dict(gridcolor='#1f2d45', color='#64748b'),
            ),
            showlegend=True,
            **plotly_dark_layout()
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # Summary table
        st.markdown('<div class="section-label">Full Results Table</div>', unsafe_allow_html=True)
        display_df = df.copy()
        display_df["Experiment"] = display_df["Experiment"].apply(
            lambda x: f"★ {x}" if x == BEST_EXP else x
        )
        st.dataframe(
            display_df.style
                .highlight_min(subset=["Perplexity"], color="#1a3a2a")
                .highlight_max(subset=["BLEU %","ROUGE-1 %","ROUGE-2 %","ROUGE-L %"], color="#1a2a3a")
                .format({"Perplexity": "{:.4f}", "BLEU %": "{:.4f}",
                         "ROUGE-1 %": "{:.4f}", "ROUGE-2 %": "{:.4f}", "ROUGE-L %": "{:.4f}"}),
            use_container_width=True, hide_index=True
        )

# ══════════════════════════════════════════════
# TAB 4 — ARCHITECTURE
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">Model Architecture — MedicalGPT</div>', unsafe_allow_html=True)

    d      = config['d_model']
    h      = config['n_heads']
    L      = config['n_layers']
    ffn    = config['ffn_dim']
    d_head = d // h
    vocab  = config['vocab_size']

    col_arch, col_info = st.columns([1, 1])

    with col_arch:
        st.markdown(f"""
        <div class="arch-box" style="border-color:#3b82f6;background:rgba(59,130,246,0.08)">
            <b>Input Tokens</b>
            <div class="arch-label">symptom text → SentencePiece BPE → token IDs</div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-box accent-border">
            <b>Embeddings</b>
            <div class="arch-label">Token Embedding + Positional Embedding → d_model = {d}</div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-box" style="border-color:#f59e0b;background:rgba(245,158,11,0.06)">
            <b>× {L} Decoder Layers</b>
            <div class="arch-label" style="margin-top:8px">
                ┌─ LayerNorm → Multi-Head Attention ({h} heads × {d_head} d_head) + Causal Mask<br>
                ├─ Residual Add + Dropout<br>
                ├─ LayerNorm → FeedForward ({d} → {ffn} → {d}) + GELU<br>
                └─ Residual Add
            </div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-box accent-border">
            <b>LayerNorm</b>
            <div class="arch-label">Final normalisation</div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-box accent-border">
            <b>LM Head (Linear)</b>
            <div class="arch-label">d_model {d} → vocab_size {vocab} | Weight tied to token embeddings</div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-box" style="border-color:#ec4899;background:rgba(236,72,153,0.07)">
            <b>Output Logits → Top-K Sampling</b>
            <div class="arch-label">Temperature scaling → Top-K → Softmax → Multinomial sample → next token</div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="section-label">Key Design Decisions</div>', unsafe_allow_html=True)

        decisions = [
            ("Decoder-only (GPT-style)", "Sequence completion task — no encoder needed. Input and output are both medical text."),
            ("Causal Masking", "Each token attends only to past tokens — enforces autoregressive generation."),
            ("Pre-Norm (LayerNorm before attention)", "More stable training than post-norm, especially for deeper models."),
            ("Weight Tying", "LM Head shares weights with token embeddings — reduces params, improves generalisation."),
            ("SentencePiece BPE", "Handles rare/OOV medical terms better than word-level tokenisation."),
            ("GELU Activation", "Smoother gradient flow than ReLU — standard in modern transformers."),
            ("Warmup + Cosine Decay", "Prevents early instability, smooth decay avoids sharp loss spikes at end."),
            ("CrossEntropyLoss (ignore PAD)", "Padding tokens do not contribute to loss — cleaner gradient signal."),
        ]

        for title, desc in decisions:
            st.markdown(f"""
            <div class="metric-card" style="flex-direction:column;align-items:flex-start;gap:4px">
                <span class="metric-label">{title}</span>
                <span style="font-size:0.82rem;color:#e2e8f0;font-family:'DM Sans',sans-serif">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="section-label">Parameter Breakdown</div>', unsafe_allow_html=True)

        # Parameter breakdown chart
        emb_params  = vocab * d * 2       # token + pos (approx, weight tied so lm_head same)
        attn_params = L * 4 * d * d       # W_q, W_k, W_v, W_o per layer
        ffn_params  = L * (d*ffn + ffn*d) # two linears per layer
        norm_params = L * 2 * d * 2 + d * 2  # norms

        labels = ["Embeddings", "Attention", "FFN", "LayerNorm"]
        values = [emb_params, attn_params, ffn_params, norm_params]

        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.5,
            marker=dict(colors=["#00c9a7","#3b82f6","#f59e0b","#ec4899"],
                        line=dict(color='#0a0e1a', width=2)),
            textfont=dict(family="Space Mono", size=10),
        ))
        fig_pie.update_layout(
            showlegend=True,
            **plotly_dark_layout(),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)