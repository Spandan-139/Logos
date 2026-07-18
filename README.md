# Logos

A decoder-only transformer language model built from scratch in PyTorch, developed through a
21-experiment ablation study — from a 32M-parameter baseline to a 124M-parameter model
benchmarked head-to-head against GPT-2, Pythia, and SmolLM2.

## Headline result

Evaluated against four public models on the same held-out data, using **bits-per-byte** rather
than perplexity (perplexity is not comparable across different tokenizers):

| Model | Params | bits/byte | ms/token | Peak VRAM |
|---|---|---|---|---|
| SmolLM2-135M | 134.5M | **1.005** | 59.6 | 3098 MB |
| Pythia-160M-deduped | 162.3M | 1.073 | 21.6 | 3054 MB |
| GPT-2 Small | 124.4M | 1.074 | 14.9 | 3099 MB |
| Pythia-70M-deduped | 70.4M | 1.219 | 11.5 | 2747 MB |
| **Logos Exp18** | 124.0M | 1.286 | **9.15** | **1372 MB** |

Logos places last on quality and **first on both latency and memory** — roughly 1.6× faster
inference and 2.3× lower peak VRAM than GPT-2 Small at effectively identical parameter count.
On a composite efficiency score weighting quality against latency, memory, and model size, it
ranks **2nd of 5**.

The quality gap is a training-compute gap, not an architecture gap: Logos saw ~245M tokens
(15k steps × 16.4k tokens/step), while GPT-2 was trained on roughly 40× more. The loss curve
was still descending at the final step.

## Findings

**Weight tying is the highest-leverage change in the study.** Tying the `lm_head` to the token
embedding cut checkpoint size 40% (122 MB → 73 MB) for a 0.001 change in validation loss. It is
carried into every subsequent experiment.

**SwiGLU beat both GELU and RMSNorm as isolated changes.** Against the baseline's 318 perplexity,
SwiGLU reached 315.9, GELU regressed to 333.3, and RMSNorm was roughly neutral at 321.2. Only
SwiGLU and weight tying were carried forward; the final model uses standard LayerNorm.

**FP16 quantization is strictly better than the FP32 model it came from.** Lower validation loss
(3.930 vs 3.993), 34% smaller on disk, 9% less VRAM, and marginally faster. It is the top-ranked
variant on the deployment efficiency index. Dynamic INT8 on CPU compressed further (267 MB) but
cost 2.2× inference latency.

**Scaling was compute-bound, not capacity-bound.** At 124M parameters, extending training from
3k to 15k steps moved perplexity 137.6 → 53.6 with no architecture change at all.

## Experiment ladder

| # | Experiment | Params | Perplexity |
|---|---|---|---|
| 00 | Baseline (32M control) | 32.2M | 318.4 |
| 01 | Weight tying | 19.2M | 320.3 |
| 02 | GELU activation | 32.2M | 333.3 |
| 03 | RMSNorm | 32.2M | 321.2 |
| 04 | SwiGLU FFN | 32.2M | 315.9 |
| 05 | Weight tying + SwiGLU | 19.2M | 306.4 |
| 06 | + 512 context | 19.3M | 345.4 |
| 07 | Deeper / narrower | 29.9M | 331.3 |
| 08 | Wider / shallower | 33.1M | 318.0 |
| 09 | Wider + WT + SwiGLU | 18.5M | 313.9 |
| 10 | Clean-data filtering | 18.5M | 338.5 |
| 11 | WT + SwiGLU, 3k steps | 19.2M | 202.8 |
| 12 | Wider + WT + SwiGLU, 3k steps | 18.5M | 210.0 |
| 13 | Medium scaling | 51.2M | 154.8 |
| 14 | Large scaling, 3k steps | 124.0M | 137.6 |
| 15 | Large, 5k steps | 124.0M | 87.0 |
| 16 | Large, 7.5k steps | 124.0M | 68.4 |
| 17 | Large, 10k steps | 124.0M | 61.4 |
| 18 | Large, 15k steps | 124.0M | **53.6** |
| 19 | Quantization (FP16 / INT8) | 124.0M | 50.9 (FP16) |
| 20 | Efficiency index | — | — |
| 21 | Public model comparison | — | — |

## Final architecture (Exp18)

```
Decoder-only transformer      12 layers x 12 heads x 768 dim (head_dim 64)
Context length                512
FFN                           SwiGLU, hidden 2048
Normalization                 LayerNorm (pre-norm)
Weight tying                  lm_head tied to token embedding
Parameters                    124,031,232
Tokenizer                     tiktoken GPT-2 (50257 vocab)
Dataset                       OpenWebText subset, 100k documents
Optimizer                     AdamW, lr 3e-4, wd 0.1, grad clip 1.0
Schedule                      400-step warmup, cosine decay to 5% of peak
Training                      15,000 steps, batch 32, 16,384 tokens/step
Precision                     AMP float16 with GradScaler
Final val loss                3.981  (perplexity 53.57)
Training time                 3,408 s, ~277 Wh measured GPU energy
```

Attention uses `F.scaled_dot_product_attention(..., is_causal=True)`. Sampling supports
temperature, top-k, and nucleus (top-p).

## The efficiency index

Experiment 20 defines a composite **Logos Efficiency Index** scoring each run on quality against
checkpoint size, peak VRAM, inference latency, measured GPU energy, and parameter count. Three
variants are reported — deployment-weighted, training-aware, and a simple ratio — because a model
that is cheap to serve is not necessarily cheap to train. Full methodology and per-experiment
scores are in `reports/Logos_Experiment_20_Efficiency_Index_Report.pdf` and
`results/exp20_efficiency_index_corrected.csv`.

## Repository layout

```
notebooks/    22 notebooks, one per experiment, run top to bottom
reports/      Per-experiment PDF write-ups and the final leaderboard
results/      Training curves, run configs, and benchmark data (CSV / JSON)
figures/      Charts from the efficiency index analysis
```

Notebooks retain their executed outputs, so results are readable without a GPU.

## Running it

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Each notebook is self-contained: it downloads its data slice, trains, evaluates, and writes a
summary. Experiments 19, 20, and 21 depend on the Exp18 checkpoint, so run `exp18` first or
download the released weights.

Training was done on a single GPU with ~48 GB VRAM. Exp18 peaks at 17.7 GB; the smaller
ablations fit in about 11 GB.

## Checkpoints

Model weights are not stored in this repository. The Exp18 checkpoint (473 MB) and its FP16
variant (310 MB) are published separately — see Releases.

## Known issues

Documented in `Logos_Code_Review.md`. The three worth noting:

1. `torch.load` is called without `weights_only=True`. Safe here since all checkpoints are
   self-produced, but it should be fixed before loading any external file.
2. Two near-duplicate benchmarking paths exist in Exp19/20, and they disagree on
   `encode` vs `encode_ordinary` — one will raise on prompts containing special-token syntax.
3. Dynamic INT8 quantization targets `nn.Linear` only, which silently breaks the embedding /
   `lm_head` weight tie in the INT8 variant.

## Notes on methodology

The validation split is the trailing 10% of a streamed, unshuffled corpus pull rather than a
random holdout. This is adequate for comparing runs drawn from the same pull, but absolute
perplexities are not strictly comparable across experiments using different `OWT_SAMPLES` values.

Cross-model comparison in Experiment 21 uses bits-per-byte specifically because Logos, GPT-2,
Pythia, and SmolLM2 do not share a tokenizer, which makes raw perplexity comparison meaningless.

## License

MIT — see [LICENSE](LICENSE).
