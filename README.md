# llm_dream_coder

Semi-automated Hall/Van de Castle dream content coding using large language models.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model: Claude Opus](https://img.shields.io/badge/model-claude--opus--4--6-orange.svg)](https://www.anthropic.com/)

---

## Overview

The Hall/Van de Castle (H/VdC) system is the standard framework for quantitative analysis of dream content. Manual coding is reliable but labor-intensive, requiring coders trained across a detailed rulebook covering characters, social interactions, activities, emotions, settings, and more.

`llm_dream_coder` is a modular, open-source toolkit that uses Claude (Anthropic) to semi-automate H/VdC coding. Each coding category is implemented as an independent Python script. Outputs are intended for researcher review rather than fully automated pipeline use — the tool is designed to reduce coding burden while keeping human judgment in the loop.

The tool is **generalizable**: it operates on any dream report without per-series customization. All rules are derived directly from the H/VdC codebook. No dataset-specific dictionaries or fine-tuning are required.

---

## Coding Categories

| Category | Status | Non-family F1 (held-out) | Script |
|---|---|---|---|
| Characters | Complete | 0.70 | `characters.py` |
| Social Interactions | In progress | — | `social_interactions.py` |
| Activities | Planned | — | — |
| Striving | Planned | — | — |
| Emotions | Planned | — | — |
| Misfortunes & Good Fortunes | Planned | — | — |
| Settings | Planned | — | — |

---

## How It Works

Each module follows the same pipeline:

1. Reads dream reports from a structured CSV file
2. Sends each report to Claude with a full H/VdC codebook prompt and few-shot examples
3. Parses the structured JSON response into H/VdC codes
4. Evaluates against ground-truth codings when available
5. Saves per-dream results (predicted codes, ground truth, F1, reasoning) to CSV for human review

Prompt caching via the Anthropic API (`cache_control: ephemeral`) keeps inference costs low when processing large batches, since the codebook system prompt is cached across calls.

---

## Installation

```bash
git clone https://github.com/cogcommscience-lab/llm_dream_coder.git
cd llm_dream_coder

pip install anthropic pandas

export ANTHROPIC_API_KEY=your_key_here
```

**Requirements:** Python 3.9+, an Anthropic API key with access to `claude-opus-4-6`.

---

## Data Format

The tool expects two CSV files. If you have a Dreambank XML export, run `xml_to_csv.py` to generate them:

```bash
python xml_to_csv.py
```

This produces:

- **`coded_dreams.csv`** — one row per dream, with columns: `dream_id`, `collection_id`, `collection_sex`, `collection_type`, `dream_report`, etc.
- **`dreambank_codings.csv`** — one row per individual coding, with columns: `dream_id`, `coding_type`, `code`, etc.

If you have dream reports in another format, create a `coded_dreams.csv` with at minimum `dream_id`, `collection_id`, and `dream_report` columns. Ground-truth codings are optional — the tool will still generate predicted codes and reasoning; it just will not compute evaluation metrics.

---

## Usage — Characters Module

```bash
# Default: sample of 50 dreams from the b-baseline collection
python characters.py

# Specific number of dreams
python characters.py --n 20

# One specific collection
python characters.py --collection norms-f

# Series mode: builds a character registry from explicit relationship
# statements as it processes dreams in order
python characters.py --collection emma --series-mode

# Single dream by ID
python characters.py --dream-id b-baseline_0003

# Full dataset
python characters.py --all
```

Output is printed to the terminal and saved to `characters_results.csv`:

```
  ✓  b-baseline_0003  pred=['1MKA', '1IKA', '1MKA', '2ISC', '1MKA']  gt=[...]  F1=1.00

Results saved → characters_results.csv
Dreams evaluated : 50
Exact match      : 19/50  (38.0%)
Mean F1          : 0.728
Non-family F1    : 0.743  (excl. H/W/D/B/T/M/F/X/A/Y/C/I/R)
```

---

## Evaluation

Two F1 scores are reported for the Characters module:

- **Overall F1** — all character codes, including family/relative codes
- **Non-family F1** — excludes family/relative codes (H, W, D, B, T, M, F, X, A, Y, C, I, R)

Non-family F1 is the primary metric for a generalizable, single-dream tool. Family and relative codes (e.g., husband, daughter, father) require biographical knowledge of the dreamer that is not present in individual dream texts. Annotators working on dream series often have this knowledge from external sources; a single-dream classifier cannot recover it.

**Characters benchmark results:**

| Collection | Type | n | Overall F1 | Non-family F1 |
|---|---|---|---|---|
| b-baseline | Series (dev) | 50 | 0.73 | 0.74 |
| norms-f | Normative (held-out) | 50 | 0.68 | 0.70 |
| emma | Series (held-out) | 50 | 0.51 | 0.54 |

Emma's lower scores reflect annotator biographical knowledge bias: coders who knew the dreamer personally applied family relationship codes from the very first dream, even when the dream text alone does not support them. Normative collections (one dream per anonymous individual) are the most appropriate benchmark for a generalizable tool.

---

## Known Limitations

- **Family/relative codes**: Codes for immediate family (H/W/D/B/T/M/F/X/A/Y/C/I/R) require biographical context not present in a single dream report. Use `--series-mode` for series data to build a registry from explicit relationship statements.
- **Annotator biographical bias**: Dream series coded by researchers with personal knowledge of the dreamer will show lower F1 on identity codes. This reflects a fundamental constraint on single-dream coding, not a failure of the prompt.
- **API cost**: `claude-opus-4-6` is used for accuracy. Expect approximately $0.02–0.05 per dream at current pricing. Prompt caching substantially reduces cost for large batches.
- **JSON parse errors**: Very long dream reports occasionally cause malformed JSON output from the model. Failed dreams are logged, skipped, and marked with `None` values in the results CSV.

---

## Citation

If you use this tool in published research, please cite:

```
Kee, R. (2026). llm_dream_coder: Semi-automated Hall/Van de Castle dream content
coding using large language models. GitHub.
https://github.com/cogcommscience-lab/llm_dream_coder
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Contact: raachaelkee@gmail.com
