# LangFormer

> **Encoder–decoder Transformer built from scratch in PyTorch for English→French neural machine translation.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![BLEU](https://img.shields.io/badge/BLEU-0.2870-brightgreen?style=flat)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat)]()

---

## What is LangFormer?

LangFormer is a ground-up PyTorch implementation of the Transformer architecture for sequence-to-sequence language translation. Built through three progressive stages — Bahdanau attention, multi-head attention, and a full Transformer — the project benchmarks each architectural decision against the last, with ablation studies confirming exactly what earns the performance gain.

**Final result: 22% BLEU improvement over the Seq2Seq baseline.**

---

## Architecture Progression

| Stage | Model | Best BLEU |
|-------|-------|-----------|
| Part 1 | Seq2Seq + Bahdanau Attention (LSTM) | 0.2552 |
| Part 2 | Seq2Seq + Multi-Head Attention (4 heads) | 0.2741 |
| Part 3 | Full Encoder–Decoder Transformer | **0.2870** |

Each stage is a deliberate upgrade. Bahdanau attention gives a soft alignment signal over the encoder hidden states. Multi-head attention learns parallel attention patterns simultaneously. The full Transformer replaces recurrence entirely with self-attention, positional encoding, and parallel computation — enabling faster convergence and better generalization.

---

## Key Results

### Hyperparameter Sensitivity (Part 1)

The best Bahdanau config (`emb_dim=256`, `hid_dim=512`, `lr=0.0005`, `dropout=0.2`) confirmed that capacity and learning rate interact tightly — a larger model requires a lower learning rate or it overfits early. Config A (`hid_dim=128`, `lr=0.005`) diverged by epoch 10; Config C trained stably through epoch 20.

### Multi-Head Scaling (Part 2)

| Heads | BLEU |
|-------|------|
| 1 | 0.1824 |
| 2 | 0.1260 |
| 4 | 0.2178 |
| 8 | 0.1860 |

4 heads was the sweet spot — enough diversity across attention patterns without redundancy. Beyond 4 heads, some heads collapsed onto similar tokens and validation BLEU dropped.

### Ablation Study (Part 3)

| Config | BLEU |
|--------|------|
| Full Transformer (PE + SA) | 0.2870 |
| No Positional Encoding | 0.2623 |
| No Self-Attention | 0.2169 |

Self-attention is load-bearing. Removing it collapses translation quality by 24% and stalls validation loss improvement throughout training. Positional encoding matters most on longer sequences where word order is harder to infer from content alone.

---

## Training Setup

- **Dataset:** English–French sentence pairs (Tatoeba-style short sentences)
- **Optimizer:** Adam with gradient clipping
- **Schedule:** Fixed LR with early stopping on validation loss
- **Evaluation:** Corpus BLEU score on held-out test set
- **Epochs:** 20 per configuration
- **Tracked:** Train/val loss, perplexity curves, attention heatmaps

---

## Attention Visualization

Bahdanau attention heatmaps show the model learns sharp token-level alignment for common imperatives (`go → va`, `hi → salut`) but produces diffuse weights on out-of-vocabulary mappings (`run → pas` in underpowered configs).

Multi-head attention weight visualizations across 8 heads confirm that different heads specialize: some track the main verb, others track punctuation or surrounding context. This head specialization drives the BLEU gain from 1-head to 4-head configurations.

---

## Project Structure

```
langformer/
├── part1_basic_attention/
│   ├── model.py          # Encoder–decoder LSTM + Bahdanau attention
│   ├── train.py          # Training loop with hyperparameter grid
│   └── attention_viz.py  # Heatmap generation
├── part2_multihead/
│   ├── model.py          # Multi-head attention module
│   └── train.py          # Head count sweep (1, 2, 4, 8)
├── part3_transformer/
│   ├── model.py          # Full Transformer (encoder + decoder stacks)
│   ├── train.py          # Ablation configs (PE on/off, SA on/off)
│   └── inference.py      # CLI / API for translation inference
└── evaluate.py           # BLEU scoring utility
```

---

## Inference

```bash
# Translate a sentence
python part3_transformer/inference.py --text "I love machine learning."

# Output:
# → J'adore l'apprentissage automatique.
```

The inference module exposes both a CLI and a callable API:

```python
from part3_transformer.inference import Translator

translator = Translator.from_checkpoint("checkpoints/transformer_best.pt")
result = translator.translate("She runs every morning.")
# → "Elle court chaque matin."
```

---

## What I Learned

Building this from scratch — no HuggingFace, no pretrained weights — forced close contact with every design decision: why scaled dot-product attention needs the `1/√d_k` factor, how positional encodings interact with embedding magnitude, why gradient clipping matters more in Transformers than LSTMs, and how BLEU score can mask failure modes that attention heatmaps reveal immediately.

The ablation study was the most valuable part. It's easy to believe self-attention works because the paper says so. It's more useful to watch BLEU collapse to 0.2169 when you turn it off.

---

## Tech Stack

`Python` · `PyTorch` · `NumPy` · `Matplotlib` · `NLTK (BLEU)` · `argparse`

---

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Bahdanau et al., [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473) (2015)
- Tatoeba Project — English–French sentence pairs

---

<p align="center">Built by <a href="https://github.com/shivani-kalal">Shivani Kalal</a></p>
