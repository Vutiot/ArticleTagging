# Experiment Log

## Project

**ArticleTagging** — Fine-tuning Qwen3-VL-2B for structured attribute extraction from fashion product listings (title + image).

Inspired by [How 1 hour of fine-tuning beat 3 weeks of RAG engineering](https://medium.com/leboncoin-tech) (leboncoin tech, Mar 2026).

## Reproducing Results

### Prerequisites

```bash
# 1. Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[training]"
pip install qwen-vl-utils

# 2. Download dataset (requires Kaggle API key in ~/.kaggle/kaggle.json)
mkdir -p data/raw/fashion
kaggle datasets download paramaggarwal/fashion-product-images-small \
  -p data/raw/fashion/ --unzip

# 3. Import CSV to JSONL
python3 -c "
from pathlib import Path
from article_tagging.scraping.importers import import_csv, export_jsonl, ImportMapping

mapping = ImportMapping(
    title_field='productDisplayName',
    image_field='id',
    attribute_fields={
        'gender': 'gender', 'masterCategory': 'masterCategory',
        'subCategory': 'subCategory', 'articleType': 'articleType',
        'baseColour': 'baseColour', 'season': 'season', 'usage': 'usage',
    },
)
listings = import_csv(Path('data/raw/fashion/styles.csv'), mapping)
for l in listings:
    if l.image_urls:
        l.image_urls = [f'data/raw/fashion/images/{l.image_urls[0]}.jpg']
export_jsonl(listings, Path('data/raw/fashion/listings.jsonl'))
"

# 4. Prepare dataset (clean, split, format)
python -m article_tagging.cli.main prepare \
  --config configs/dataset_fashion.yaml \
  --raw-data data/raw/fashion/listings.jsonl \
  --output-dir data/processed/fashion \
  --image-dir data/raw/fashion/images
```

### Run V0 and V0+ baselines

```bash
source .venv/bin/activate

# Both V0 and V0+ on 50 samples (default, ~5 min)
python scripts/eval_baseline.py

# Only V0+ on 100 samples
python scripts/eval_baseline.py --run v0+ --samples 100

# Full test set (3241 samples, ~3 hours)
python scripts/eval_baseline.py --samples 0

# Custom paths
python scripts/eval_baseline.py \
  --test-data data/processed/fashion/test.jsonl \
  --schema configs/schemas/fashion.yaml \
  --output-dir reports/ \
  --seed 42
```

Results are saved to `reports/v0_baseline/` and `reports/v0_plus_prompt/`.

---

## Dataset

**Source**: [Kaggle Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

| Metric | Value |
|--------|-------|
| Raw listings | 44,446 |
| After cleaning | 32,398 |
| Dropped (invalid attrs) | 843 |
| Dropped (duplicates) | 11,200 |
| Dropped (no image) | 5 |
| Train | 25,918 |
| Val | 3,239 |
| Test | 3,241 |

### Attributes to extract (7)

| Attribute | Type | Unique values | Notes |
|-----------|------|--------------|-------|
| gender | enum | 5 | Men, Women, Boys, Girls, Unisex |
| masterCategory | enum | 7 | Apparel, Accessories, Footwear, Personal Care, ... |
| subCategory | enum | 43 | Topwear, Bottomwear, Watches, Shoes, ... |
| articleType | string | 143 | Tshirts (16%), Shirts (7%), Casual Shoes (6%), ... long tail |
| baseColour | enum | 41 | Black, White, Blue, Navy Blue, Grey, ... |
| season | enum | 4 | Summer, Winter, Fall, Spring |
| usage | enum | 8 | Casual, Sports, Formal, Ethnic, ... |

### articleType distribution (top 20 / 143)

| articleType | Count | % |
|-------------|------:|--:|
| Tshirts | 7,070 | 15.9% |
| Shirts | 3,217 | 7.2% |
| Casual Shoes | 2,846 | 6.4% |
| Watches | 2,542 | 5.7% |
| Sports Shoes | 2,036 | 4.6% |
| Kurtas | 1,844 | 4.1% |
| Tops | 1,762 | 4.0% |
| Handbags | 1,759 | 4.0% |
| Heels | 1,323 | 3.0% |
| Sunglasses | 1,073 | 2.4% |
| Wallets | 936 | 2.1% |
| Flip Flops | 916 | 2.1% |
| Sandals | 897 | 2.0% |
| Briefs | 849 | 1.9% |
| Belts | 813 | 1.8% |
| Backpacks | 724 | 1.6% |
| Socks | 686 | 1.5% |
| Formal Shoes | 637 | 1.4% |
| Perfume and Body Mist | 614 | 1.4% |
| Jeans | 609 | 1.4% |
| *... 123 more types* | | |

---

## Model

**Qwen3-VL-2B-Instruct** — 2B parameter Vision-Language Model

- Loaded in FP16 (~4 GB VRAM)
- GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
- Inference: ~2-4s per sample (no batching, no vLLM)

---

## Experiments

### Run 1: V0 Baseline — Naive Prompt (no valid values)

**Date**: 2026-03-24
**Samples**: 50 (random from test set, seed=42)

**System prompt**:
```
You extract product attributes from the title and image.
Respond with valid JSON only.
Attributes to extract: gender, masterCategory, subCategory, articleType, baseColour, season, usage
```

**Hypothesis**: The base model can extract attributes from title + image without any guidance on valid values.

#### Results

| Attribute | Accuracy |
|-----------|----------|
| gender | 88.0% |
| baseColour | 76.0% |
| usage | 32.0% |
| articleType | 8.0% |
| masterCategory | 4.0% |
| season | 4.0% |
| subCategory | 2.0% |
| **Exact Match** | **0.0%** |

#### Per-category exact match

| Category | Exact Match |
|----------|-------------|
| Apparel | 0.0% |
| Accessories | 0.0% |
| Footwear | 0.0% |
| Personal Care | 0.0% |

#### Analysis

- **gender** (88%) and **baseColour** (76%): Model extracts these well from title text and image visual cues.
- **masterCategory** (4%), **subCategory** (2%): Model invents its own taxonomy (e.g., "Shirts" instead of "Apparel", "Plaid" instead of "Topwear").
- **articleType** (8%): Close but wrong format — "Shirt" instead of "Shirts", "Watch" instead of "Watches".
- **season** (4%): Model defaults to "All" — cannot infer season from image.
- **usage** (32%): Partially guessable from context.
- **Exact match 0%**: Not a single sample got all 7 attributes correct.

**Conclusion**: Without valid value constraints, the model understands the task but uses its own vocabulary. Matches the leboncoin article's V0 findings.

---

### Run 2: V0+ — Enum Values in Prompt

**Date**: 2026-03-24
**Samples**: 50 (same seed=42 for fair comparison)

**System prompt** (1,332 chars):
```
You extract product attributes from the title and image.
Respond with valid JSON only. Use ONLY values from the lists below.

gender (valid values): Men, Women, Boys, Girls, Unisex
masterCategory (valid values): Apparel, Accessories, Footwear, Personal Care, Free Items, Sporting Goods, Home
subCategory (valid values): Topwear, Bottomwear, Innerwear, Dress, ... (43 values)
articleType: free text string
baseColour (valid values): Black, White, Blue, Navy Blue, ... (41 values)
season (valid values): Summer, Winter, Fall, Spring
usage (valid values): Casual, Sports, Formal, Ethnic, Smart Casual, Travel, Party, Home
```

**Hypothesis**: Listing valid enum values in the prompt will force the model to use the correct taxonomy, dramatically improving category/type attributes.

#### Results

| Attribute | V0 | V0+ | Delta |
|-----------|------|------|-------|
| gender | 88.0% | 98.0% | +10.0% |
| masterCategory | 4.0% | 46.0% | **+42.0%** |
| subCategory | 2.0% | 82.0% | **+80.0%** |
| articleType | 8.0% | 10.0% | +2.0% |
| baseColour | 76.0% | 86.0% | +10.0% |
| season | 4.0% | 34.0% | **+30.0%** |
| usage | 32.0% | 84.0% | **+52.0%** |
| **Exact Match** | **0.0%** | **4.0%** | **+4.0%** |

#### Per-category exact match

| Category | V0 | V0+ | Delta |
|----------|-----|------|-------|
| Apparel | 0.0% | 9.1% | +9.1% |
| Accessories | 0.0% | 0.0% | +0.0% |
| Footwear | 0.0% | 0.0% | +0.0% |
| Personal Care | 0.0% | 0.0% | +0.0% |

#### Analysis

- **Massive improvements** on enum-constrained attributes: subCategory (+80%), usage (+52%), masterCategory (+42%), season (+30%).
- **articleType still at 10%**: Defined as free text (no enum), so no improvement. With 143 unique values, listing them all in the prompt would be impractical (leboncoin article's lesson: large label spaces overwhelm the model).
- **Exact match only 4%**: The articleType bottleneck drags down exact match since it's required for a full match.
- **gender near-perfect at 98%**: With valid values listed, the model almost never makes mistakes on simple attributes.

**Conclusion**: Enum constraints in the prompt give huge gains for structured attributes. But free-text / large-label-space attributes remain the bottleneck. This is exactly where fine-tuning should shine — the model learns the exact label vocabulary from thousands of examples.

#### Variance analysis (3 seeds x 50 samples)

To verify results are stable, V0+ was run with seeds 42, 123, and 7.

| Attribute | seed=42 | seed=123 | seed=7 | Mean | Std |
|-----------|---------|----------|--------|------|-----|
| gender | 98.0% | 100.0% | 98.0% | 98.7% | 1.2% |
| masterCategory | 46.0% | 60.0% | 54.0% | 53.3% | 7.0% |
| subCategory | 82.0% | 90.0% | 86.0% | 86.0% | 4.0% |
| articleType | 10.0% | 14.0% | 12.0% | 12.0% | 2.0% |
| baseColour | 86.0% | 82.0% | 84.0% | 84.0% | 2.0% |
| season | 34.0% | 38.0% | 30.0% | 34.0% | 4.0% |
| usage | 84.0% | 70.0% | 80.0% | 78.0% | 7.2% |
| **Exact Match** | 4.0% | 4.0% | 0.0% | 2.7% | 2.3% |
| **Avg per-attribute** | 62.9% | 64.9% | 63.4% | **63.7%** | **1.0%** |

Results are stable: average per-attribute accuracy is 63.7% +/-1.0% across seeds. The noisiest attributes are `masterCategory` and `usage` (+/-7%), which are more context-dependent. 50 samples is sufficient for reliable comparison.

---

## Next Steps

### Run 3: V2 — LoRA Fine-tuned

**Date**: 2026-03-25
**Samples**: 50 (same seed=42)

#### Training configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-VL-2B-Instruct |
| Quantization | 4-bit (bitsandbytes QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q,k,v,o,gate,up,down_proj |
| Gradient checkpointing | unsloth |
| Training samples | 25,918 |
| Validation samples | 3,239 |
| Epochs | 3 |
| Batch size | 1 (effective: 8 with grad_accum) |
| Learning rate | 2e-4 |
| Warmup steps | 50 |
| Total steps | 9,720 |
| Training time | ~7 hours |
| VRAM used | 4.7 GB / 8 GB |

#### Training loss curve

```
Step     Loss     Eval Loss    LR
──────   ──────   ─────────    ──────
10       3.011                  2.0e-4
50       0.330                  2.0e-4
100      0.210                  2.0e-4
200      0.196    0.215         2.0e-4
500      0.194    0.168         1.9e-4
1000     0.159    0.156         1.8e-4
2000     0.127    0.147         1.6e-4
3000     0.114    0.138         1.4e-4
4000     0.101    0.129         1.2e-4
5000     0.100    0.125         1.0e-4
6000     0.087    0.121         7.5e-5
6400     0.085    0.120 (best)  6.7e-5
7000     0.084    0.123         5.6e-5
```

Loss dropped 97% (3.01 → 0.08). Best eval loss at step 6400 (0.120). No overfitting — eval loss plateaued rather than increasing.

#### Results

| Attribute | V0 | V0+ | V2 (LoRA) | V0+→V2 |
|-----------|-----|------|-----------|--------|
| gender | 88.0% | 98.0% | 96.0% | -2.0% |
| masterCategory | 4.0% | 46.0% | **98.0%** | **+52.0%** |
| subCategory | 2.0% | 82.0% | **96.0%** | +14.0% |
| articleType | 8.0% | 10.0% | **98.0%** | **+88.0%** |
| baseColour | 76.0% | 86.0% | **92.0%** | +6.0% |
| season | 4.0% | 34.0% | **76.0%** | **+42.0%** |
| usage | 32.0% | 84.0% | **92.0%** | +8.0% |
| **Exact Match** | **0.0%** | **4.0%** | **58.0%** | **+54.0%** |

#### Per-category exact match

| Category | V0 | V0+ | V2 (LoRA) | V0+→V2 |
|----------|-----|------|-----------|--------|
| Accessories | 0.0% | 0.0% | 71.4% | +71.4% |
| Apparel | 0.0% | 9.1% | 45.5% | +36.4% |
| Footwear | 0.0% | 0.0% | 61.5% | +61.5% |
| Personal Care | 0.0% | 0.0% | 100.0% | +100.0% |

#### Sample predictions

```
"Hidekraft Women Black Clutch"
  + gender: Women | + masterCategory: Accessories | + subCategory: Bags
  + articleType: Clutches | + baseColour: Black | + season: Summer | + usage: Casual
  → 7/7 correct (exact match)

"Quiksilver Men Stripes Blue Caps"
  + gender: Men | + masterCategory: Accessories | + subCategory: Headwear
  + articleType: Caps | + baseColour: Blue | + season: Summer | + usage: Casual
  → 7/7 correct (exact match)

"Jockey MC Men Grey Rio Briefs 8033"
  + gender: Men | + masterCategory: Apparel | + subCategory: Innerwear
  + articleType: Briefs | - baseColour: Grey (exp: Grey Melange) | + season: Summer
  + usage: Casual
  → 6/7 (colour shade mismatch)

"Jealous 21 Women Blue Jegging"
  + gender: Women | + masterCategory: Apparel | + subCategory: Bottomwear
  + articleType: Jeans | + baseColour: Blue | - season: Summer (exp: Fall)
  + usage: Casual
  → 6/7 (season mismatch — hard to infer from image alone)
```

#### Analysis

- **articleType: 10% → 98%** — The biggest win. The model learned all 143 article types from examples. This was impossible with prompting alone (too many values to list).
- **masterCategory: 46% → 98%** — Fine-tuning taught the exact taxonomy. Near-perfect.
- **season: 34% → 76%** — The hardest attribute. Improved significantly but still the weakest — season is genuinely ambiguous from a product image.
- **Exact match: 4% → 58%** — From nearly zero to majority correct. Matches the leboncoin article's finding that fine-tuning dramatically outperforms prompt engineering.
- **Remaining errors**: mostly subtle colour shades ("Grey" vs "Grey Melange") and season prediction (inherently ambiguous).
- **0 JSON parsing errors** — the fine-tuned model always outputs valid JSON in the expected format.

**Conclusion**: Fine-tuning on 25k examples for 7 hours pushed exact match from 4% to 58%, confirming the article's thesis. The model learned the complete label vocabulary, inter-attribute dependencies, and output format — all implicitly from examples, with no complex RAG or cascade engineering.

---

### Run 4: V2 + Guided Decoding (vLLM `guided_json`)

**Date**: 2026-03-26
**Samples**: 50 (same seed=42)
**Decoding**: Greedy (temperature=0) via vLLM OpenAI-compatible API

The fine-tuned model was retrained from scratch (Qwen3-VL-2B-Instruct base) with the same config as Run 3. Training again early-stopped at **step 7000** with best eval loss **0.121** at step 6400 — nearly identical to the previous run (0.120).

The model was served with vLLM 0.18.0 using `--enable-lora` and evaluated with `guided_json` schema constraints that force the output to match the JSON Schema (enum values, required fields, no extra properties).

#### Hypothesis

Guided JSON decoding should eliminate remaining enum violations (e.g., "Grey" instead of "Grey Melange") and push exact match above 58%.

#### Configurations tested

All 4 combinations of prompt style × guided decoding were evaluated for a complete picture:

| Config | Prompt | Guided | EM | gender | master | sub | article | colour | season | usage |
|--------|--------|--------|-----|--------|--------|-----|---------|--------|--------|-------|
| V2 | V0 (short) | No | **58.0%** | 100% | 100% | 98% | 96% | 92% | 74% | 92% |
| V2 | V0 (short) | Yes | **58.0%** | 100% | 100% | 98% | 96% | 92% | 74% | 92% |
| V2 | V0+ (enums) | No | 52.0% | 94% | 98% | 100% | 96% | 92% | 68% | 92% |
| V2 | V0+ (enums) | Yes | 52.0% | 94% | 98% | 100% | 96% | 92% | 68% | 92% |

#### Per-category exact match (V0 prompt configs)

| Category | V2 (no guided) | V2 + Guided |
|----------|----------------|-------------|
| Accessories | 71.4% | 71.4% |
| Apparel | 40.9% | 40.9% |
| Footwear | 69.2% | 69.2% |
| Personal Care | 100.0% | 100.0% |

#### Analysis

**1. Guided decoding has zero impact on the fine-tuned model (58% = 58%)**

The fine-tuned model with the V0 prompt (matching training distribution) already produces **100% valid JSON** with correct enum values. The `guided_json` constraint had nothing to fix — every token the model generated was already consistent with the schema.

This means the remaining 42% errors are **semantic mistakes** (wrong attribute value, not wrong format), which guided decoding cannot address:
- **season (74%)**: Inherently ambiguous from image/title alone
- **baseColour (92%)**: Subtle shades ("Grey" vs "Grey Melange") where the model picks a valid but wrong value
- **subCategory (98%)**: Rare edge cases

**2. V0+ prompt hurts the fine-tuned model (58% → 52%)**

Switching from the short V0 prompt (used during training) to the longer V0+ prompt (with enum values) causes a **distribution shift**. The model was never trained with enum values in the prompt, so providing them at inference degrades performance:
- gender drops 100% → 94%
- season drops 74% → 68%
- Personal Care exact match drops 100% → 0%

**Lesson**: Always match the inference prompt to the training prompt. Prompt engineering and fine-tuning are complementary strategies — mixing them naively can hurt.

**3. Guided + V0+ are identical to non-guided equivalents**

With temperature=0 (greedy decoding), guided JSON had zero effect in all configurations. The model's greedy output was already valid JSON in every case, so the FSM constraints were never activated.

#### Conclusion

For this dataset and model, **guided JSON decoding provides no improvement** over the fine-tuned model alone. The 58% exact match ceiling is bounded by semantic difficulty (season ambiguity, colour shade distinctions), not by format or vocabulary errors. Future improvements should focus on: (a) more training data, (b) larger models, (c) better image features for ambiguous attributes, or (d) multi-label approaches for season.

---

### Run 5: V3 — Qwen3-VL-30B-A3B (MoE VLM, ~3B active params)

**Date**: TBD
**Samples**: 50 (same seed=42)
**Hardware**: Lightning AI Studio (NVIDIA L4, 24 GB VRAM)
**Model**: Qwen3-VL-30B-A3B-Instruct (MoE: 30B total, ~3B active per token)

#### Hypothesis

A larger MoE VLM with ~3B active parameters (vs 2B dense) may improve on the
hardest attributes (season, baseColour) where the smaller model plateaus at
58% exact match. The MoE architecture keeps inference cost similar to the 2B
model while providing access to a larger parameter space during training.

#### Training configuration

| Setting | V2 (2B) | V3 (30B-A3B) |
|---------|---------|--------------|
| Model | Qwen3-VL-2B-Instruct | Qwen3-VL-30B-A3B-Instruct |
| Architecture | Dense 2B | MoE 30B (~3B active) |
| Quantization | 4-bit QLoRA | 4-bit QLoRA |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| Target modules | q,k,v,o,gate,up,down_proj | q,k,v,o,gate,up,down_proj |
| Epochs | 3 | 3 |
| Batch size | 1 (effective: 8) | 1 (effective: 8) |
| Learning rate | 2e-4 | 2e-4 |
| GPU | RTX 4070 (8 GB) | L4 (24 GB) |

#### Results

*To be filled after running the benchmark on Lightning AI.*

```
python scripts/benchmark_qwen3vl_30b.py --phase all
```

Results will be saved to `reports/v3_qwen3vl_30b/eval_result.json`.

---

## Summary

| Run | Method | Exact Match | Avg Per-Attr | Engineering effort |
|-----|--------|-------------|-------------|-------------------|
| V0 | Naive prompt | 0% | 31% | 5 min |
| V0+ | Enum values in prompt | 4% | 63% | 30 min |
| **V2** | **LoRA fine-tuning (3 epochs)** | **58%** | **93%** | **7 hours training** |
| V2+Guided | V2 + vLLM guided_json | 58% | 93% | +10 min setup |
| V3 | Qwen3-VL-30B-A3B LoRA | TBD | TBD | TBD |

The fine-tuned model achieves **93% average per-attribute accuracy** and **58% exact match** (all 7 attributes correct simultaneously), up from 0% exact match with naive prompting — confirming that a few hours of fine-tuning beats weeks of prompt engineering for structured extraction tasks.

Guided JSON decoding provides a safety net for production (guarantees valid output) but does not improve accuracy for a well-trained model that already produces valid JSON.

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GB VRAM |
| CUDA | 13.1 |
| PyTorch | 2.11.0+cu130 |
| Model | Qwen/Qwen3-VL-2B-Instruct (FP16, ~4 GB) |
| Training VRAM | 4.7 GB peak (4-bit QLoRA + gradient checkpointing) |

## Software

| Package | Version |
|---------|---------|
| unsloth | 2026.3.11 |
| trl | 0.23.0 |
| transformers | 4.57.2 |
| torch | 2.11.0+cu130 |
| peft | 0.18.1 |
| bitsandbytes | 0.49.2 |
| vllm | 0.18.0 |
