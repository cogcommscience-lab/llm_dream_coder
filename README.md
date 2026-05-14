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

| Category | Status | Overall F1 (attribute-level, held-out) | Script |
|---|---|---|---|
| Characters | Complete | 0.87 (non-family 0.89) | `characters.py` |
| Social Interactions | Validated, awaiting final test | agg 0.77 / fri 0.79 / sex 0.97 (norms-f validation) | `social_interactions.py` |
| Activities | Validated, awaiting final test | 0.72 validation (dev 0.77) | `activities.py` |
| Success and Failure | Validated, awaiting final test | 0.91 dev / 0.89 validation (norms-f) | `success_failure.py` |
| Misfortunes & Good Fortunes | Validated, awaiting final test | MF 0.73 / GF 1.00 (norms-f validation) | `misfortunes_good_fortunes.py` |
| Emotions | Validated, awaiting final test | 0.935 mean F1 (norms-f validation) | `emotions.py` |
| Settings | Validated, awaiting final test | 0.775 mean F1 (norms-f validation) | `settings.py` |
| Objects | Validated, awaiting final test | 0.758 mean F1 (norms-f validation) | `objects.py` |

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

## Social Interactions Module — In Progress

The Social Interactions module codes all aggressive (agg), friendly (fri), and sexual (sex) interactions in a dream report. Each interaction is a four-field tuple: `(init, rec, type, code)` where `init` and `rec` are character codes (or `D` for the dreamer), `type` is one of `agg`/`fri`/`sex`, and `code` combines a sub-type number with a direction symbol (`>` one-way, `=` mutual, `R` rejected, `*` self-directed).

**Usage:**

```bash
python social_interactions.py --collection norms-f --n 50
python social_interactions.py --dream-id b-baseline_0003
```

**Current benchmark results (attribute-level F1):**

| Collection | Role | n | F1 agg | F1 fri | F1 sex |
|---|---|---|---|---|---|
| b-baseline | Development | 20 | 0.760 | 0.674 | 0.970 |
| norms-f | Validation | 50 | 0.769 | **0.787** | 0.968 |
| emma | Validation (annotator-bias) | 50 | 0.777 | 0.682 | 0.921 |
| norms-m | **Reserved final test** | 500 | — | — | — |

**Sub-type F1s are the primary reported metric.** Researchers typically use one or two interaction types depending on their study (e.g., only aggression for conflict research, only friendly for social connectedness). Each sub-type is an independent H/VdC scale; an aggregate overall F1 collapses them artificially and does not correspond to any standard research use case.

Aggression and sex coding are strong (0.76–0.97). Friendly interaction F1 reached **0.787 on the held-out norms-f validation set** after iterative refinement (animal-interaction handling, conservative threshold rule, three new F4-focused few-shot examples). Friendly remains slightly below target on the b-baseline development set (0.674) and on emma (0.682, where annotator biographical bias inflates the human-coded count). Sub-type confusion is rare (91% sub-type agreement when init/rec/direction match) — the remaining error mode is type-level over- and under-coding.

The same attribute-level F1 metric is used as for Characters: each tuple is decomposed into its constituent slots (init/rec character attributes, type, sub-type, direction) and scored via Counter intersection, giving partial credit when (for example) the type and sub-type are correct but the character code is mis-coded.

**Methodology note.** During iterative development, b-baseline served as the development set, while norms-f and emma were used as validation sets — error patterns on these sets informed prompt refinements across multiple rounds. The norms-m collection (n=500) has been reserved as the **untouched final test set** and will be evaluated only once per module after all classifier development is complete; those will be the published held-out numbers.

This module remains under active development pending final test on norms-m.

---

## Activities Module — In Progress

The Activities module codes what every character (and the dreamer) does in a dream. Each activity is a three-field tuple: `(init, rec, code)` where `init` is the actor, `rec` is the recipient (or `null` for solo acts), and `code` is a sub-type letter plus an optional direction modifier.

**Eight sub-types:** P (Physical), M (Movement on foot), L (Locomotion by vehicle), V (Verbal), S (Visual), A (Auditory), E (Expressive), C (Cognitive).

**Direction modifiers:** *(none)* = solo, `>` = directed, `=` = mutual, `R` = reciprocated.

**Usage:**

```bash
python activities.py                           # sample of 50 norms-f dreams (dev partition)
python activities.py --n 20                    # specific number of dreams
python activities.py --collection norms-f      # full collection
python activities.py --dream-id norms-f_0001   # single dream
python activities.py --skip 50 --n 50          # dreams 51-100 (validation partition)
```

**Data partitioning note.** Activities have ground-truth codings only in norms-f and norms-m. b-baseline and emma have zero activity codings. Development uses norms-f dreams 1–50; the validation set is norms-f dreams 51–491; norms-m (n=494) is reserved as the untouched final test set.

**Current benchmark results (attribute-level F1):**

| Partition | Role | n | Overall F1 | P | M | V | S | L | C | E | A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| norms-f 1–50 | Development | 50 | 0.767 | 0.723 | 0.830 | 0.812 | 0.816 | 0.900 | 0.860 | 0.957 | 0.980 |
| norms-f 51–100 | Validation (held-out) | 50 | **0.723** | 0.634 | 0.777 | 0.856 | 0.741 | 0.865 | 0.827 | 0.925 | 0.960 |
| norms-m | **Reserved final test** | 494 | — | — | — | — | — | — | — | — | — |

Overall validation F1 (0.72) meets the ≥ 0.70 target. P is the weakest sub-type on validation (0.634); remaining P errors are largely attributable to character-code identity confusion (K/S/U) inherited from the Characters module rather than wrong P decisions.

Attribute-level F1 is computed via Counter intersection over decomposed `(slot, value)` tuples: each tuple is split into init/rec character attributes, sub-type, and direction, giving partial credit when sub-type or direction is correct but the character code is mis-coded.

This module remains under active development pending final test on norms-m.

---

## Success and Failure Module — In Progress

The Success and Failure module codes goal-directed behavior outcomes: which characters (including the dreamer) achieved a success and which experienced a failure. Unlike Activities, Success and Failure is not coded by sub-type — it codes simply who succeeded and who failed, identifying the characters involved.

**Three elements required for a Success and Failure code:**
1. A stated **goal** the character was trying to achieve
2. Explicit **effort** toward that goal
3. A clear **outcome**: success or failure

Success and Failure coding is intentionally sparse — most dreams have 0–2 such events. Routine locomotion, emotional reactions, and accidental events are not coded even when they involve effort.

**Usage:**

```bash
python success_failure.py                           # sample of 50 norms-f dreams (dev partition)
python success_failure.py --n 20                    # specific number of dreams
python success_failure.py --collection norms-f      # full collection
python success_failure.py --dream-id norms-f_0003   # single dream
python success_failure.py --skip 50 --n 50          # dreams 51-100 (validation partition)
```

**Current benchmark results (attribute-level F1):**

| Partition | Role | n | F1 succ | F1 fail | Mean F1 |
|---|---|---|---|---|---|
| norms-f 1–50 | Development | 50 | 0.900 | 0.913 | **0.907** |
| norms-f 51–100 | Validation (held-out) | 50 | 0.880 | 0.893 | **0.887** |
| norms-m | **Reserved final test** | 494 | — | — | — |

Both F1 sub-types (succ and fail) are reported separately and averaged for Mean F1. Success and Failure ground truth exists only in norms-f and norms-m. The dev partition is norms-f dreams 1–50; the validation set is norms-f dreams 51–100; norms-m is reserved as the untouched final test set.

Attribute-level F1 is computed via Counter intersection over decomposed `(slot, value)` tuples for character codes (number, gender, identity, age), giving partial credit when the character type is correct but one attribute is mis-coded.

This module remains under active development pending final test on norms-m.

---

## Misfortunes & Good Fortunes Module — In Progress

The Misfortunes & Good Fortunes module codes two categories of passive events in dream reports:

- **Misfortunes (MF)**: Negative events that happen TO a character from outside — not chosen by the character, but undesirable events that befall them (accidents, threats, illness, death). Coded with a sub-type 1–6.
- **Good Fortunes (GF)**: Unexpectedly positive events that happen TO a character without deliberate goal-directed effort — windfalls, lucky rescues, unexpected gifts. No sub-type.

**Six MF sub-types:**
- **1 — Apprehension**: The character's worry, anxiety, or dread is the primary narrative element
- **2 — Physical accident/mishap**: Unintentional physical harm not caused by another person's deliberate act
- **3 — Adverse situation**: Being in a seriously unpleasant circumstance (social scrutiny, displacement, hostile environment)
- **4 — Physical jeopardy**: Passive victim of acute physical danger from an external source (reckless vehicle, fire, threatening creature)
- **5 — Physical suffering**: Actual illness, injury, pain, or deprivation (already materialized, not merely feared)
- **6 — Death**: A character literally dies within the dream narrative

**Usage:**

```bash
python misfortunes_good_fortunes.py                              # all b-baseline dreams (dev partition)
python misfortunes_good_fortunes.py --n 50                       # first 50 b-baseline dreams
python misfortunes_good_fortunes.py --collection norms-f         # validation partition
python misfortunes_good_fortunes.py --collection norms-f --n 50  # first 50 norms-f dreams
python misfortunes_good_fortunes.py --dream-id b-baseline_0062   # single dream
python misfortunes_good_fortunes.py --all                        # full dataset
```

**Data partitioning note.** MF/GF ground-truth codings exist in b-baseline, norms-f, and norms-m. b-baseline (250 dreams, 107 with MF codings, 5 with GF codings) is the development partition. norms-f is the validation partition. norms-m (494 dreams) is reserved as the untouched final test set. Unlike Activities and Success & Failure, MF/GF is coded in b-baseline — however, b-baseline is a dream series with series-specific character codes ("Q") that the model cannot identify without series context, causing systematic character mis-coding in a subset of dreams. Normative collections (norms-f, norms-m) do not have this issue.

**Current benchmark results (attribute-level F1):**

MF F1 is computed at the attribute level: each (character, sub-type) pair is decomposed into `(slot, value)` tuples — number, gender, identity, age, sub-type — then scored via Counter intersection. This gives partial credit when the character is correct but the sub-type is wrong (or vice versa). GF F1 is computed over character attributes only (no sub-type).

| Partition | Role | n dreams | MF GT dreams | F1 MF | F1 GF |
|---|---|---|---|---|---|
| b-baseline (all) | Development | 250 | 107 | 0.587* | 0.760 |
| norms-f 1–100 | Validation | 100 | 26 | **0.728** | **1.000** |
| norms-m | **Reserved final test** | 494 | — | — | — |

*b-baseline MF F1 reflects the series-specific "Q" character code limitation (see note above). norms-f validation F1 is the primary benchmark.

This module remains under active development pending final test on norms-m.

---

## Emotions Module — In Progress

The Emotions module codes which characters (including the dreamer) explicitly experience one of five emotional states in a dream report. The key requirement: the emotion must be stated in words — the model does not infer feelings from events or actions.

**Five emotion types:**
- **AP — Apprehension**: fear, terror, panic, dread, nervousness, anxiety, embarrassment, being upset or disturbed in a threatening context
- **HA — Happiness**: joy, happiness, pleasure, contentment, excitement, elation, satisfaction, delight
- **AN — Anger**: anger, rage, annoyance, disgust, hostility, irritation, fury, indignation
- **CO — Confusion**: confusion, bewilderment, puzzlement, surprise, perplexity, disorientation, finding something peculiar
- **SD — Sadness**: sadness, grief, sorrow, depression, disappointment, despair, crying accompanied by a stated loss

Each emotion is coded as a `(type, character)` pair — the simplest H/VdC format, with no sub-direction or initiator/recipient structure. The dreamer is coded as `D`; other characters use standard H/VdC codes.

**Usage:**

```bash
python emotions.py                           # sample of 50 norms-f dreams (dev partition)
python emotions.py --n 20                    # specific number of dreams
python emotions.py --collection norms-f      # full collection
python emotions.py --dream-id norms-f_0001   # single dream
python emotions.py --skip 50 --n 50          # dreams 51-100 (validation partition)
```

**Data partitioning note.** Emotion codings exist in norms-f, norms-m, and b-baseline. Development uses norms-f dreams 1–50 (39/50 have at least one emotion coding); validation uses norms-f dreams 51–100; norms-m (494 dreams) is reserved as the untouched final test set.

**Current benchmark results (attribute-level F1):**

Per-type F1 is computed at the attribute level: for each emotion type, predicted and ground-truth character lists are decomposed into `(slot, value)` tuples (number, gender, identity, age) and scored via Counter intersection. Mean F1 is the average of the five per-type F1s. Overall F1 pools all `(emotion_type, char_attrs)` pairs together.

| Partition | Role | n | AP | HA | AN | CO | SD | Mean F1 |
|---|---|---|---|---|---|---|---|---|
| norms-f 1–50 | Development | 50 | 0.970 | 0.995 | 1.000 | 0.940 | 1.000 | **0.981** |
| norms-f 51–100 | Validation (held-out) | 50 | 0.935 | 0.938 | 0.960 | 0.920 | 0.922 | **0.935** |
| norms-m | **Reserved final test** | 494 | — | — | — | — | — | — |

At 0.935 validated mean F1, Emotions is the highest-scoring module in the project. AP is the most common emotion type (most dreams with a coding have at least one AP); AN is the most consistently coded (1.000 dev F1). Remaining errors are concentrated in character identity ambiguity (known vs. stranger when not explicit in the text) and a small number of AP/CO boundary cases.

This module remains under active development pending final test on norms-m.

---

## Settings Module — In Progress

The Settings module codes the physical environment of each distinct scene in a dream. Each location receives a 2-letter code: `[Location][Familiarity]`.

**Location types (first letter):**
- **I — Indoor**: building, room, enclosed structure (dormitory, classroom, store, house)
- **O — Outdoor**: outside in open air (street, road, field, lake, forest)
- **A — Ambiguous**: physical setting exists but cannot be classified as indoor or outdoor (camp, vague location, town setting)
- **N — No Setting**: truly no physical environment (NS only — spiritual/abstract dream with no described location)

**Familiarity types (second letter):**
- **F — Familiar**: place named by dreamer, or dreamer's own possessive space ("my room," named street)
- **Q — Questionable**: place exists but no familiarity signal either way
- **U — Unfamiliar**: explicit signal the dreamer doesn't recognize the place ("surroundings were unfamiliar," "a new home," "a strange town")
- **D — Distorted**: dreamer knows the place type but it is physically wrong ("my room but it didn't look like mine," impossible features like seeing through walls)
- **G — Geographic (outdoor only)**: named real-world location (city, state, lake, country)

Multiple codes per dream are common (average 1.31 per dream). Unlike other modules, Settings codes have no character fields — each code is the 2-letter setting code alone. Every dream receives at least one code.

**Usage:**

```bash
python settings.py                           # sample of 50 norms-f dreams (dev partition)
python settings.py --n 20                    # specific number of dreams
python settings.py --collection norms-f      # full collection
python settings.py --dream-id norms-f_0018   # single dream
python settings.py --skip 50 --n 50          # dreams 51-100 (validation partition)
```

**Data partitioning note.** Settings codings exist in norms-f and norms-m only (b-baseline and emma have zero settings codings). Development uses norms-f dreams 1–50; validation uses norms-f dreams 51–100; norms-m (494 dreams) is reserved as the untouched final test set.

**Current benchmark results (F1):**

Each 2-letter code is decomposed into two attribute tuples — `("location", I/O/A/N)` and `("familiarity", F/Q/U/D/G)` — and scored via Counter intersection at the attribute level. This gives 50% partial credit when location is correct but familiarity is wrong (or vice versa), and 0% for a completely incorrect code.

| Partition | Role | n | Mean F1 | Precision | Recall |
|---|---|---|---|---|---|
| norms-f 1–50 | Development | 50 | **0.936** | 0.933 | 0.950 |
| norms-f 51–100 | Validation (held-out) | 50 | **0.775** | 0.756 | 0.811 |
| norms-m | **Reserved final test** | 494 | — | — | — |

**Per-type breakdown (validation):**

| Type | F1 (validation) | n dreams |
|---|---|---|
| Indoor (I) | 0.880 | 26 |
| Outdoor (O) | 0.840 | 13 |
| Ambiguous (A) | 0.464 | 11 |
| Familiar (F) | 0.812 | 21 |
| Questionable (Q) | 0.746 | 22 |
| Unfamiliar (U) | 0.825 | 4 |
| Distorted (D) | 0.734 | 4 |

The AF (ambiguous + familiar) code is the most challenging — it appears frequently in validation dreams but rarely in the development set, requiring the model to generalize from limited examples. I/O location codes and U familiarity code perform strongly; the Q/F boundary remains the most common error mode.

This module remains under active development pending final test on norms-m.

---

## Objects Module

The Objects module codes all distinct physical objects that appear in a dream scene. Each object receives a 2-letter category code. Unlike most other modules, objects codes can repeat within a dream — each distinct instance of the same object type gets its own code (e.g., two suitcases = TR + TR; three food items = FO + FO + FO).

**The 25 object codes:**

| Code | Category | Examples |
|---|---|---|
| AR | Residential | house, apartment, dorm room, bedroom |
| AV | Vocational | store, office, classroom, laboratory |
| AE | Entertainment | restaurant, theater, museum, stadium |
| AI | Institutional | hospital, church, courthouse, prison |
| AD | Architectural Details | door, window, staircase, fireplace |
| AM | Architectural Misc. | corridor, passageway, tower, fountain, fence |
| AB | Structural Elements | rare — structural building components |
| BH | Head | face, hair, eyes, nose, mouth |
| BE | Extremities | arm, hand, leg, foot, fingers |
| BT | Torso | shoulder, chest, abdomen, back |
| BA | Anatomy | internal organs, blood, bones, growths |
| BS | Sex | reproductive/excretory body parts |
| CL | Clothing | dress, shoes, jewelry, accessories |
| CM | Communication | book, letter, phone, newspaper |
| FO | Food | all food and drink items |
| HH | Household | furniture, appliances, containers |
| IR | Recreation Implements | sporting goods, games, toys, instruments in use |
| IT | Tools | tools, machinery, apparatus |
| IW | Weapons | gun, sword, bomb |
| MO | Money | currency, checks, bank books |
| MS | Miscellaneous | objects not fitting any other category |
| NA | Nature | trees, lakes, terrain, weather, animals |
| RG | Regions | cities, towns, states, parks, yards |
| ST | Streets | roads, bridges, railroads, sidewalks |
| TR | Travel | cars, planes, boats, luggage |

**Usage:**

```bash
python objects.py                            # sample of 50 norms-f dreams (dev partition)
python objects.py --n 20                     # specific number of dreams
python objects.py --collection norms-f       # full collection
python objects.py --dream-id norms-f_0029   # single dream
python objects.py --skip 50 --n 50          # dreams 51-100 (validation partition)
```

**Data partitioning note.** Objects codings exist in norms-f and norms-m only (b-baseline and emma have zero objects codings). Development uses norms-f dreams 1–50; validation uses norms-f dreams 51–100; norms-m (494 dreams) is reserved as the untouched final test set. Average ~5.25 object codes per dream.

**Important:** The Nature code "NA" must not be converted to NaN — the loader uses `keep_default_na=False`.

**Current benchmark results (F1):**

Evaluation uses raw Counter-based F1 (no attribute decomposition — codes are atomic). Partial credit for getting the right count of each code type.

| Partition | Role | n | Mean F1 | Precision | Recall |
|---|---|---|---|---|---|
| norms-f 1–50 | Development | 50 | **0.821** | 0.848 | 0.825 |
| norms-f 51–100 | Validation (held-out) | 50 | **0.758** | 0.798 | 0.739 |
| norms-m | **Reserved final test** | 494 | — | — | — |

Objects is the most categorically complex module (25 codes, repeating instances, no character fields). The primary challenge is the MS (miscellaneous) category, which is inherently unpredictable, and correct counting of object instances.

This module remains under active development pending final test on norms-m.

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
