# llm_dream_coder

Semi-automated Hall/Van de Castle dream content coding using large language models.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model: Claude Opus](https://img.shields.io/badge/model-claude--opus--4--6-orange.svg)](https://www.anthropic.com/)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://cogcommscience-lab.github.io/llm_dream_coder/)

**[Full Documentation](https://cogcommscience-lab.github.io/llm_dream_coder/)**

---

## Overview

The Hall/Van de Castle (H/VdC) system is the standard framework for quantitative analysis of dream content. Manual coding is reliable but labor-intensive, requiring coders trained across a detailed rulebook covering characters, social interactions, activities, emotions, settings, and more.

`llm_dream_coder` is a modular, open-source toolkit that uses Claude (Anthropic) to semi-automate H/VdC coding. Each coding category is implemented as an independent Python script. Outputs are intended for researcher review rather than fully automated pipeline use — the tool is designed to reduce coding burden while keeping human judgment in the loop.

The tool is **generalizable**: it operates on any dream report without per-series customization. All rules are derived directly from the H/VdC codebook. No dataset-specific dictionaries or fine-tuning are required.

---

## Coding Categories

| Category | Status | Non-family F1 (attribute-level, held-out) | Script |
|---|---|---|---|
| Characters | Complete | 0.89 | `characters.py` |
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
Dreams evaluated : 49
Exact match      : 19/49  (38.8%)
Mean F1 (attr)   : 0.877
Non-family F1    : 0.848  (excl. H/W/D/B/T/M/F/X/A/Y/C/I/R)
Attr number    : 0.934
Attr gender    : 0.892
Attr identity  : 0.715
Attr age       : 0.938
```

---

## Evaluation

F1 is computed at the **attribute level**, not the whole-code level. Each H/VdC code is decomposed into its constituent slots — number, gender, identity, age (and ANI/CZZ type for animals/creatures) — and F1 is computed via Counter intersection over those `(slot, value)` tuples. This gives partial credit when the model gets 3 of 4 attributes right (e.g., predicting `1MKA` when the truth is `1FKA` is no longer treated as a total miss).

Two attribute-level F1 scores are reported:

- **Overall F1 (attribute-level)** — all character codes, including family/relative codes
- **Non-family F1 (attribute-level)** — excludes family/relative codes (H, W, D, B, T, M, F, X, A, Y, C, I, R)

Non-family F1 is the primary metric for a generalizable, single-dream tool. Family and relative codes require biographical knowledge of the dreamer that is not present in individual dream texts. Annotators working on dream series often have this knowledge from external sources; a single-dream classifier cannot recover it.

**Characters benchmark results (attribute-level F1):**

| Collection | Type | n | Overall F1 | Non-family F1 |
|---|---|---|---|---|
| b-baseline | Series (dev) | 49 | 0.877 | 0.848 |
| norms-f | Normative (held-out) | 50 | 0.873 | **0.889** |
| emma | Series (held-out) | 50 | 0.807 | 0.706 |

**Per-attribute breakdown (norms-f, held-out):**

| Attribute | F1 |
|---|---|
| Number | 0.915 |
| Gender | 0.850 |
| Identity | 0.719 |
| Age | 0.910 |

Identity is the weakest slot — most remaining errors are K/S/U/O/R confusions where human coders applied subjective judgment. Number, gender, and age are more constrained and score above 0.85 across all collections.

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
Kee, R. & Huskey, R. (2026). llm_dream_coder: Semi-automated Hall/Van de Castle dream content
coding using large language models. GitHub.
https://github.com/cogcommscience-lab/llm_dream_coder
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Contact: rlkee@ucdavis.edu
