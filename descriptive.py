#!/usr/bin/env python3
"""
descriptive.py — Hall/Van de Castle Descriptive Elements Coder

Semi-automated classifier for the Descriptive Elements (modifiers) category
of the H/VdC coding system. Codes explicit descriptive quality words in dream
reports using 18 polarity codes (9 dimensions × +/−), then evaluates against
human-coded ground truth.

Usage:
    python descriptive.py                                        # 50 norms-f dreams (dev partition)
    python descriptive.py --n 20                                 # sample of 20 dreams
    python descriptive.py --all                                  # full dataset
    python descriptive.py --collection norms-f                   # one collection
    python descriptive.py --dream-id norms-f_0013               # single dream
    python descriptive.py --skip 50 --n 50                       # dreams 51–100 (validation partition)

Output:
    descriptive_results.csv — per-dream results with predicted and ground-truth
    descriptor code lists, plus F1 for each dream.

Notes on evaluation:
    Codes are atomic — each code is matched directly (no decomposition).
    A Counter-based intersection gives credit for each correct code instance,
    including multiple instances of the same code (e.g., I+×3 vs I+×2 gives
    partial credit). Codes that appear multiple times in GT must appear that
    many times in the prediction to get full credit.

Methodology note:
    Descriptive elements are coded in norms-f and norms-m only. Development
    uses norms-f dreams 1–50; norms-m is reserved as the untouched final test
    set. About 62 of 491 norms-f dreams have ZERO modifier codings — an empty
    list is a valid and common output.
"""

import os
import sys
import json
import time
import argparse
from collections import Counter

import pandas as pd
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
MODEL             = "claude-opus-4-6"
INPUT_CSV         = "coded_dreams.csv"
CODINGS_CSV       = "dreambank_codings.csv"
OUTPUT_CSV        = "descriptive_results.csv"
SAMPLE_SIZE       = 50
SAMPLE_COLLECTION = "norms-f"
DELAY_SECONDS     = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# CODEBOOK
# ─────────────────────────────────────────────────────────────────────────────
CODEBOOK = """
Hall/Van de Castle (H/VdC) Descriptive Elements Coding Rules
=============================================================

## Overview

Code every EXPLICIT descriptive quality word in the dream report that falls
into one of 9 quality dimensions. Each dimension has two poles: positive (+)
and negative (−). Each distinct occurrence of a codeable descriptor gets its
own code — repeated qualifiers each earn a separate code.

"very" appearing three times = I+ × 3.
"large" and "tall" = S+ + S+.
An empty dream (no explicit quality words) → output [].

Average: about 3 descriptor codes per dream (among dreams with any codings).
About 1 in 7 dreams has NO codeable descriptors — output [] for those.

─────────────────────────────────────────────────────────────────────────────
## THE 9 DIMENSIONS (18 CODES)

**C — Color**
  C+ : Chromatic colors — any named color with hue: red, blue, green, yellow,
       orange, purple, pink, violet, gold, silver (when used as a color).
  C− : Achromatic colors — black, white, gray, dark, pale, colorless.
  Rule: Code each distinct color word applied to a distinct object.
  DO NOT CODE: Colors used as emotional metaphors ("felt blue," "saw red"),
  colors referring to character race/ethnicity, or colors clearly embedded
  in a proper name with no descriptive force.

**S — Size**
  S+ : Large, big, huge, enormous, immense, vast, tall, long, wide, deep,
       broad, gigantic, towering, spacious; also quantity words: lots of, a
       lot of, millions of, tons of (large implied amount = S+).
  S− : Small, tiny, little, short, narrow, thin, slight, miniature, petite,
       shallow, compact.
  Rule: Code each explicit size word applying to a distinct entity.
  "Little" preceding a person noun = S−: "little boy," "little girl,"
  "little sister," "little child" all code as S− for small physical stature.
  DO NOT CODE: Metaphorical uses ("short time," "narrow escape," "big deal"
  meaning important). Size must describe a physical dimension.

**A — Age**
  A+ : Old, ancient, aged, elderly, antique, archaic, aging; comparative
       older/eldest.
  A− : Young, new, modern, fresh, recent, novel; comparative younger/youngest;
       youth-related adjectives ("youthful," "childlike appearance").
  Rule: Code comparative terms and age adjectives.
  DO NOT CODE: Specific ages given in parentheses or numbers ("He is 48,"
  "(20)"). Those are character-coding notation, not descriptors.

**D — Density**
  D+ : Full, filled, crowded, packed, jammed, overflowing, crammed, stuffed,
       overloaded, swollen, bulging, abundant, teeming.
  D− : Empty, hollow, vacant, bare, deserted, void, sparse.
  Rule: Requires a physically BOUNDED container or enclosed space. "Alone"
  and "no one" are NOT codeable — social absence, not physical density.
  DO NOT CODE D− for open landscapes: "barren landscape," "deserted road,"
  "desolate countryside" describe setting atmosphere, not a bounded container
  being empty. D− applies to things like "the drawer was empty," "a hollow
  tree," "the room was vacant."

**T — Thermal**
  T+ : Hot, warm, heated, boiling, burning, scorching, steaming.
  T− : Cold, cool, freezing, icy, chilly, frigid, frozen.
  Rule: ONLY explicit, literal temperature references. Inferential knowledge
  that something is hot/cold (knowing fire is hot) does NOT code. Metaphorical
  temperature ("cold shoulder," "warm welcome," "heated argument") does NOT code.

**V — Velocity**
  V+ : Fast, quickly, rapidly, swiftly, hurriedly, rushing, dashing, sprinting,
       racing, at full speed, too fast.
  V− : Slowly, sluggishly, gradually, leisurely, crawling (speed).
  Rule: Code explicit speed/velocity qualifiers applied to movement or action.
  DO NOT CODE: "suddenly" — per codebook, "suddenly" NEVER codes as V+.

**L — Linearity**
  L+ : Straight, flat, level, direct, linear, smooth (surface).
  L− : Curved, curving, crooked, twisted, winding, bent, zigzag, jagged,
       irregular (shape).
  Rule: Code only when the dream explicitly draws attention to the shape/path
  quality. Knowledge that something is inherently curved (a banana, a road
  that bends) is NOT sufficient — the text must describe it.

**I — Intensity**
  I+ : Very, extremely, terribly, incredibly, absolutely, completely, entirely,
       totally, deeply, greatly, tremendously, intensely, fiercely, violently,
       strongly, highly, quite, too (as intensifier), so (as intensifier),
       really, utterly, awfully, dreadfully, remarkably.
  I− : Slightly, somewhat, barely, hardly, scarcely, a little, a bit, mildly,
       gently, rather (as downgrader), calmly, uninterested, oblivious,
       indifferent (words expressing low engagement or mild quality).
  Rule: Code each separate intensifier word as its own I+ or I− instance.
  Emotions need explicit intensity modifiers — "I was afraid" does NOT code;
  "I was terribly afraid" DOES code I+ (for "terribly").
  DO NOT CODE: "suddenly" (never codes), degree words embedded in idioms.

**E — Evaluation**
  E+ : Aesthetic/moral praise: beautiful, lovely, gorgeous, handsome,
       magnificent, wonderful, excellent, perfect, admirable, splendid,
       elegant, graceful, fine, nice, good, pretty, swell, attractive.
  E− : Aesthetic/moral criticism: ugly, hideous, horrible, awful, terrible,
       shabby, disreputable, disgusting, revolting, dirty (as value judgment),
       wrong, immoral, abhorrent, shameful, foul, dreadful (as evaluation),
       strange (when used to convey wrongness or distaste).
  Rule: LIMITED to aesthetic judgments (beauty/ugliness of appearance) and
  moral judgments (right/wrong, ethical/unethical). Do NOT code for functional
  descriptions, emotional reactions, or general negative/positive situations.
  "I was angry" = NO. "The outfit was disreputable" = E−.

─────────────────────────────────────────────────────────────────────────────
## CODING RULES

### What to code
- Explicit quality words that directly modify a noun, verb, or situation
- Each distinct occurrence of a codeable word earns its own code
- Comparative and superlative forms count (older = A+; fastest = V+)
- Quantity words implying large/small amounts: "lots of," "a lot of,"
  "millions of," "tons of" = S+ (large quantity); "a tiny bit of" = S−

### What NOT to code
- Inferential qualities: don't code T+ because fire is mentioned, or L−
  because a winding road is implied; the TEXT must state the quality
- Metaphorical uses: "short time," "narrow escape," "cold behavior"
- Character ages given as numbers in parentheses: (20), (48), (50)
- The word "suddenly" — per codebook, NEVER codes as V+
- Simple emotion words without an intensity modifier: "afraid," "happy,"
  "angry," "excited" alone do NOT code I
- "old" meaning previous or former ("her old room," "the old days," "her old
  diapers") → NOT A+; code A+ only when "old" means aged or ancient
- Atmospheric/setting darkness: "it was dark," "all dark," "dark room," "dark
  night" describe setting atmosphere, NOT a specific object's color — do NOT
  code C− for these. Reserve C− for explicit object/person color descriptions
  ("black hat," "dark-haired woman," "gray car," "silver ring")

### Metacognitive frame — NEVER code
Do NOT code descriptor words that appear in statements about the dreamer's
memory or the dream report itself, rather than the dream's content:
  ✗ "Dream was very hazy" — describes memory quality, not dream content
  ✗ "I can remember very little of this dream" — memory report
  ✗ "I have forgotten a lot of the details" — memory report
  ✗ "It was very vivid" (describing the dream as a whole) — meta-report
  ✗ "All I remember dreaming was..." — framing sentence
Only code words that describe qualities of the dream's events, objects, and
people — not the dreamer's recollection of them.

### Intensity (I) — use conservatively
The coder is selective: NOT every "very" or intensifier gets coded.
  - Code I+ for strong, salient intensifiers modifying dream-content words:
    "very happy," "extremely relieved," "as much force as possible"
  - "Quite," "sort of," "rather," "pretty" are BORDERLINE — code only when
    they clearly intensify a salient quality ("quite beautiful," "quite
    terrified"). Do NOT code in logical/comparative contexts ("quite contrary
    to," "quite fond of"), in atmospheric descriptions, or in weakly emotional
    sentences where the coder tends not to code
  - Negated intensity modifiers ("didn't seem to be terribly upset") usually
    do NOT code — the intensity is cancelled by the negation
  - I− is RARE — be very conservative. "Quite indifferent," "a bit thwarted,"
    "sort of happy" typically do NOT reach the threshold. Code I− only for
    explicitly weak engagement in a prominent descriptive context ("calmly
    sitting," "he was utterly uninterested," "only slightly troubled")
  - How the dreamer speaks or reports in the dream (told him very incoherently,
    whispered quietly) typically does NOT code I — code only salient qualities
    of objects, people, and events, not the manner of the dreamer's speech

### D (Density) vs S (Size) for quantities
"Lots of food," "full of people," "millions of chairs" — these are better
coded as S+ (large quantity/amount) than D+. Reserve D+ for explicit
descriptions of a bounded container being physically full or jammed:
  ✓ D+ — "the place was jammed with people," "the drawer was packed full"
  ✓ S+ — "lots of food," "a lot of money," "millions of chairs"

### Evaluation (E) — expanded list
  E+ includes: nice, fine, swell, better (as positive evaluation),
    beautiful, handsome, lovely, gorgeous, wonderful, excellent, admirable,
    charming, pleasant, simple (when praised), welcoming; explicit aesthetic
    or moral praise words.
  E− includes: disgusted/disgusting, dirty (as moral judgment), pigs/pig
    (used pejoratively), false (statements, accusations), wrong (moral sense),
    abhorrent, horrible, ugly, hideous, shabby, disreputable, glaring (harsh
    aesthetic), immoral, perverted, squalid, dreadful, poor (as in "poor
    home life," "poor conditions," "poor quality of life" = bad/inadequate)
  "Disgusted" and "perplexed" are different: disgust = moral/aesthetic
  condemnation (E−); perplexed = confused reaction (not E)
  DO NOT code E+ for social/relational uses of "good": "a good friend,"
  "a good man," "a good person," "how good it was to see her" are social
  phrases, NOT aesthetic evaluations. E+ requires explicit aesthetic praise
  (beautiful, handsome, lovely) or explicit moral approval (honest, virtuous).

### Dreams with no codeable descriptors → output []
Many dreams describe events without quality words in the 9 categories.
If no words clearly fall into the dimensions, output [].

### Repeat for each instance
If "very" appears twice applied to dream content, output I+, I+. If two
objects are each described as large, output S+, S+.

─────────────────────────────────────────────────────────────────────────────
## OUTPUT FORMAT

Respond with ONLY valid JSON — no prose before or after — in this exact shape:

{
  "descriptors": ["I+", "S+", "C-", "V+"],
  "reasoning": {
    "I+ (very confused)": "The word 'very' is an explicit intensity modifier.",
    "S+ (large rings)": "'Large' explicitly states size.",
    "C- (black rings)": "'Black' is an achromatic color.",
    "V+ (hurriedly)": "'Hurriedly' expresses rapid movement.",
    "NOT CODED — 'very strange'": "Judgment: 'strange' is a subjective perception, not a standard quality dimension in this codebook."
  }
}

Field rules:
  - "descriptors": flat list of codes, one per distinct descriptor instance.
  - Codes can repeat (I+, I+, I+ for three intensity words).
  - Valid codes: A+, A-, C+, C-, D+, D-, E+, E-, I+, I-, L+, L-, S+, S-,
    T+, T-, V+, V-
  - Include reasoning for each coded word and notable NOT CODED judgments.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES  (drawn from norms-f dev partition: dreams 1–50)
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT = [
    # Example 1: Color, Intensity, Size, Velocity — variety across dimensions
    {
        "dream": (
            "I dreamed I was to be married to the boy I go with (20) when he appeared. "
            "However, it was not him but his father (50). Everything was very confused. "
            "I remember emptying my father's (48) suitcase in my bedroom and then "
            "hurriedly putting everything back in again, remembering asking my girlfriend "
            "(19) whether the wedding announcements were sent out. Then I remember having "
            "rings on my finger. Father of the boy I go with changed into boyfriend. "
            "Rings were large and black made of wood and etched with gold. "
            "I thought them very strange."
        ),
        "descriptors": ["C+", "C-", "I+", "S+", "V+"],
        "reasoning": {
            "I+ (very confused)": "'Very' is an explicit high-intensity modifier.",
            "V+ (hurriedly)": "'Hurriedly' expresses rapid movement — velocity+.",
            "S+ (large rings)": "'Large' explicitly states a positive size dimension.",
            "C- (black)": "'Black' is an achromatic color — color−.",
            "C+ (gold)": "'Gold' is a chromatic color — color+.",
            "NOT CODED — ages (20), (50), (48), (19)": "Parenthetical ages are character-coding notation, not descriptor words.",
            "NOT CODED — 'very strange'": "Judgment call: 'strange' is a subjective perception, not clearly in the aesthetic/moral (E) or any other standard dimension. The coder chose not to code this instance.",
        },
    },
    # Example 2: Age, Color, Density, Evaluation, Intensity, Size — rich example
    {
        "dream": (
            "I was at a large party which seemed to be a New Year's Eve gathering. "
            "It took place in the Hotel Carter Ballroom and the place was jammed with "
            "people in formal dress. I was dressed in a disreputable outfit of blue "
            "jeans and a dirty shirt, and every time I tried to make conversation, I "
            "was completely ignored. There was a tall, dark, handsome young man whose "
            "attention I was trying to attract, but he too ignored me, so I went to a "
            "building across the street where there was another party. After going up "
            "in several different elevators, I arrived in a long, narrow hallway also "
            "crowded with people in formal dress, who likewise ignored me."
        ),
        "descriptors": ["A-", "C+", "D+", "E+", "E-", "E-", "I+", "S+", "S+"],
        "reasoning": {
            "S+ (large party)": "'Large' is an explicit size+ word.",
            "D+ (jammed with people)": "'Jammed' means crowded/packed — density+.",
            "C+ (blue jeans)": "'Blue' is a chromatic color — color+.",
            "E- (disreputable outfit)": "'Disreputable' is a negative aesthetic/moral evaluation.",
            "E- (dirty shirt)": "'Dirty' as a value judgment — negative aesthetic evaluation.",
            "I+ (completely ignored)": "'Completely' is an explicit high-intensity modifier.",
            "S+ (tall young man)": "'Tall' is an explicit size+ word.",
            "A- (young man)": "'Young' is an explicit age− descriptor.",
            "E+ (handsome)": "'Handsome' is a positive aesthetic evaluation.",
            "NOT CODED — 'dark' (complexion)": "'Dark' here describes a person's complexion/appearance as a physical trait, not an achromatic color applied to an object in the scene.",
            "NOT CODED — 'long, narrow hallway'": "Coder judgment: these architectural descriptors were not coded, possibly because they describe the setting structure rather than a salient quality the dreamer attends to.",
            "NOT CODED — 'crowded' (second party)": "Coder coded D+ for 'jammed' but not for 'crowded' — each pass through the dream, make your best judgment; here we follow the GT.",
        },
    },
    # Example 3: Density and Intensity repeated — shows duplicate codes
    {
        "dream": (
            "I dreamt that my front teeth were loose and wobbled. Then I thought that "
            "I was chewing something hard. I spit what I was chewing into my hand and "
            "looked at it. My hand was full of blood and teeth. My mouth felt sore and "
            "swollen and I spit out what was in it. Again I saw the blood and teeth, "
            "but immediately my mouth felt full of hard bits again. I kept spitting "
            "and trying to get rid of the feeling, but it wouldn't stop. I was getting "
            "panicky and it seemed very real. I went in the bathroom and tried to drink "
            "some water, but immediately I had to spit and I couldn't drink. I became "
            "so frightened finally that I woke myself up."
        ),
        "descriptors": ["D+", "D+", "I+", "I+"],
        "reasoning": {
            "D+ (hand was full of blood and teeth)": "'Full' describes a bounded container (hand/mouth) — density+.",
            "D+ (mouth felt full of hard bits)": "Second distinct 'full' in a separate sentence — another D+ instance.",
            "I+ (seemed very real)": "'Very' is an explicit high-intensity modifier.",
            "I+ (so frightened)": "'So' is a second explicit high-intensity modifier.",
            "NOT CODED — 'hard' (chewing something hard)": "'Hard' describes texture/physical property, not a dimension in the 9 categories.",
            "NOT CODED — 'sore and swollen'": "Physical symptoms, not descriptive quality dimensions in this codebook.",
            "NOT CODED — 'panicky'": "Simple emotion word without an explicit intensity modifier.",
            "NOT CODED — 'loose' (teeth were loose)": "Physical state descriptor, not in the 9 dimensions.",
        },
    },
    # Example 4: Age, Evaluation, Linearity — shows L+ and E moral judgment
    {
        "dream": (
            "I was the only person in the dream that I recognized. I was walking past "
            "an old brick building behind our house and for some reason I turned to look "
            "back at the building. The lower left part of the wall was gone and a room "
            "that had not existed was left bare. In a straight chair was seated a naked "
            "man, his left side facing me. A naked woman was bending over him; they were "
            "in the act of intercourse. Just the two figures were visible to me and the "
            "beauty of the pose and the symphony of body movements struck me speechless. "
            "Then suddenly the figures were gone and what appeared to be the neighborhood "
            "children filed out of the room looking furtively from side to side. I was "
            "again struck, but this time by a feeling of abhorrence at the thought of "
            "children picking up sex in a manner they know is wrong."
        ),
        "descriptors": ["A+", "E+", "E-", "E-", "L+"],
        "reasoning": {
            "A+ (old brick building)": "'Old' is an explicit age+ descriptor applied to the building.",
            "L+ (straight chair)": "'Straight' explicitly describes the chair's shape — linearity+.",
            "E+ (the beauty of the pose)": "'Beauty' is a positive aesthetic evaluation.",
            "E- (a feeling of abhorrence)": "'Abhorrence' expresses strong moral/aesthetic condemnation — evaluation−.",
            "E- (in a manner they know is wrong)": "'Wrong' is an explicit moral negative evaluation.",
            "NOT CODED — 'suddenly'": "Per codebook, 'suddenly' NEVER codes as V+.",
            "NOT CODED — 'furtively'": "Behavioral manner, not a descriptor in the 9 dimensions.",
            "NOT CODED — 'bare' (room left bare)": "Judgment call — 'bare' could be D−, but the coder did not code it; the room is structurally damaged, not densely or sparsely filled in the coding sense.",
        },
    },
    # Example 5: E− for moral condemnation
    {
        "dream": (
            "I was in biology lab and had a bag of cookies and a bag of after dinner "
            "mints. I put them on the desk and said everyone was welcome to help "
            "themselves. When I went to get some, they were all gone, so I went around "
            "to each person in the lab and made them put some back. I was perplexed "
            "because they had taken advantage of my generosity and disgusted because "
            "some of them had made such pigs of themselves."
        ),
        "descriptors": ["E-"],
        "reasoning": {
            "E- (disgusted ... pigs of themselves)": (
                "'Disgusted' expresses moral/aesthetic condemnation — evaluation−. "
                "'Made pigs of themselves' reinforces the same moral judgment. "
                "Both are part of one evaluative statement — coded as a single E−."
            ),
            "NOT CODED — 'perplexed'": (
                "'Perplexed' = confused/surprised reaction, not a moral or aesthetic "
                "evaluation. Perplexity is not E."
            ),
            "NOT CODED — 'generosity'": "Noun describing the dreamer's trait, not a descriptor in the 9 dimensions.",
        },
    },
    # Example 6: Conservative coding — metacognitive frame, atmospheric dark, quantity = S+
    {
        "dream": (
            "Dream was very hazy. It had to do with Stunt Night. I remember waiting "
            "backstage till it was our turn to go on stage. Before we went on, there "
            "were other people performing - like an amateur show. I saw millions of "
            "chairs, little children's chairs backstage. It was all dark. I felt sort "
            "of a cold chill in my stomach. I was quite petrified. After the others "
            "finished, our leader said come on. Let's show em. We went and performed "
            "the whole stunt perfectly."
        ),
        "descriptors": ["S+"],
        "reasoning": {
            "S+ (millions of chairs)": (
                "'Millions' is an extreme quantity word implying a very large number "
                "of objects = S+. The dreamer explicitly draws attention to this "
                "overwhelming quantity."
            ),
            "NOT CODED — 'Dream was very hazy'": (
                "This is a METACOGNITIVE statement about the dream report, not about "
                "dream content. 'Very' in the metacognitive frame does NOT code as I+."
            ),
            "NOT CODED — 'all dark' (setting atmosphere)": (
                "'All dark' describes the general atmospheric setting — not a specific "
                "object's color. Atmospheric darkness does NOT code as C−. "
                "C− is reserved for explicit object/person color descriptions."
            ),
            "NOT CODED — 'little children's chairs'": (
                "'Little' here describes the type of chairs (sized for children) as a "
                "category label, not an explicit size judgment the dreamer is attending "
                "to. Coder chose not to code S− for this incidental modifier."
            ),
            "NOT CODED — 'sort of a cold chill'": (
                "'Sort of' is a very weak qualifier; 'cold chill' is a bodily sensation "
                "rather than an explicit thermal descriptor. 'Sort of' does not reach "
                "the threshold for I−, and 'cold' here is figurative."
            ),
            "NOT CODED — 'quite petrified'": (
                "'Quite' in this emotional context was not coded by the human coder. "
                "'Quite' is borderline — use conservatively and only when it clearly "
                "intensifies a salient quality. Prefer not to code 'quite' unless "
                "the context is clearly strong."
            ),
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    dreams  = pd.read_csv(INPUT_CSV,   keep_default_na=False)
    codings = pd.read_csv(CODINGS_CSV, keep_default_na=False)

    mod_df = codings[codings["coding_type"] == "mod"].copy()

    gt_by_dream = {}
    for dream_id, group in mod_df.groupby("dream_id"):
        codes = []
        for _, row in group.iterrows():
            code = str(row.get("code", "")).strip()
            if code:
                codes.append(code)
        gt_by_dream[dream_id] = codes

    return dreams, gt_by_dream


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────────────────
def build_system_content():
    ex_text = ""
    for ex in FEW_SHOT:
        output = {"descriptors": ex["descriptors"], "reasoning": ex["reasoning"]}
        ex_text += f"\nDream:\n{ex['dream']}\n\n"
        ex_text += f"Output:\n{json.dumps(output, indent=2, ensure_ascii=False)}\n"
        ex_text += "\n---\n"

    return (
        CODEBOOK
        + "\n\n"
        + "## Worked Examples\n"
        + ex_text
        + "\n\nRespond with ONLY valid JSON matching the output format above.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────
VALID_CODES = {
    "A+", "A-", "C+", "C-", "D+", "D-", "E+", "E-",
    "I+", "I-", "L+", "L-", "S+", "S-", "T+", "T-", "V+", "V-",
}

def call_claude(client, dream_text, system_content):
    """Call Claude and return (descriptors_list, reasoning_dict)."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": system_content,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    "Code all descriptive elements in this dream report:\n\n"
                    + dream_text
                ),
            }
        ],
    )

    raw = response.content[0].text.strip()

    if "```" in raw:
        for part in raw.split("```"):
            candidate = part.lstrip("json").strip()
            if candidate.startswith("{"):
                raw = candidate
                break

    if not raw.startswith("{"):
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(raw)

    raw_descriptors = parsed.get("descriptors", [])
    if not isinstance(raw_descriptors, list):
        raw_descriptors = []
    descriptors = [
        c for o in raw_descriptors
        if (c := str(o).strip()) in VALID_CODES
    ]
    return descriptors, parsed.get("reasoning", {})


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(pred_codes, gt_codes):
    """Counter-based F1 over a list of descriptor codes (no decomposition)."""
    pred = Counter(str(c).strip() for c in pred_codes)
    true = Counter(str(c).strip() for c in gt_codes)
    tp = sum((pred & true).values())
    precision = tp / sum(pred.values()) if pred else (1.0 if not true else 0.0)
    recall    = tp / sum(true.values()) if true else (1.0 if not pred else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision": round(precision, 3),
        "recall":    round(recall,    3),
        "f1":        round(f1,        3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="H/VdC Descriptive Elements Coder")
    parser.add_argument("--all",        action="store_true",
                        help="Run on the full dataset")
    parser.add_argument("--collection", type=str, default=None,
                        help="Restrict to one collection (e.g. norms-f, norms-m)")
    parser.add_argument("--dream-id",   type=str, default=None,
                        help="Run on a single dream by ID")
    parser.add_argument("--n",          type=int, default=None,
                        help="Override sample size (default: SAMPLE_SIZE setting)")
    parser.add_argument("--skip",       type=int, default=0,
                        help="Skip the first N dreams from the selected collection")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)
    system_content = build_system_content()

    dreams, gt_by_dream = load_data()

    # ── Select dreams ─────────────────────────────────────────────────────────
    n = args.n if args.n else SAMPLE_SIZE
    if args.dream_id:
        sample = dreams[dreams["dream_id"] == args.dream_id]
        if sample.empty:
            sys.exit(f"Dream ID '{args.dream_id}' not found.")
    elif args.all:
        sample = dreams
    elif args.collection:
        pool   = dreams[dreams["collection_id"] == args.collection]
        sample = pool.iloc[args.skip:args.skip + n]
    else:
        pool   = dreams[dreams["collection_id"] == SAMPLE_COLLECTION]
        sample = pool.iloc[args.skip:args.skip + n]

    print(f"Processing {len(sample)} dream(s) with model {MODEL}...\n")

    results = []

    for _, row in sample.iterrows():
        dream_id   = row["dream_id"]
        dream_text = row["dream_report"]

        if pd.isna(dream_text) or str(dream_text).strip() == "":
            print(f"  SKIP  {dream_id}  (missing report text)")
            continue

        gt = gt_by_dream.get(dream_id, [])

        try:
            pred, reasoning = call_claude(client, str(dream_text), system_content)
        except Exception as e:
            print(f"  ERROR {dream_id}: {e}")
            results.append({
                "dream_id":         dream_id,
                "pred_descriptors": "[]",
                "gt_descriptors":   json.dumps(gt),
                "f1":               None,
                "precision":        None,
                "recall":           None,
                "reasoning":        f"ERROR: {e}",
                "dream_report":     str(dream_text)[:300],
            })
            time.sleep(DELAY_SECONDS)
            continue

        m = evaluate(pred, gt)

        print(
            f"  {'✓' if m['f1'] == 1.0 else '✗'}  {dream_id}"
            f"  pred={pred}  gt={gt}"
            f"  F1={m['f1']:.2f}"
        )

        results.append({
            "dream_id":         dream_id,
            "pred_descriptors": json.dumps(pred, ensure_ascii=False),
            "gt_descriptors":   json.dumps(gt,   ensure_ascii=False),
            "f1":               m["f1"],
            "precision":        m["precision"],
            "recall":           m["recall"],
            "reasoning":        json.dumps(reasoning, ensure_ascii=False),
            "dream_report":     str(dream_text)[:300],
        })

        time.sleep(DELAY_SECONDS)

    # ── Save results ──────────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────────────────────
    valid   = df_out[df_out["f1"].notna()]
    n_valid = len(valid)

    print(f"\n{'─'*60}")
    print(f"Results saved → {OUTPUT_CSV}")
    print(f"Dreams evaluated : {n_valid}")

    if n_valid > 0:
        print(f"Mean F1          : {valid['f1'].mean():.3f}")
        print(f"  Precision      : {valid['precision'].mean():.3f}")
        print(f"  Recall         : {valid['recall'].mean():.3f}")

        # Per-code F1 on dreams that contain that code in GT
        print(f"\nPer-code F1 (on dreams containing that code in GT, n≥3):")
        all_codes = sorted({
            c
            for row in valid["gt_descriptors"]
            for c in json.loads(row)
        })
        for code in all_codes:
            mask = valid["gt_descriptors"].apply(
                lambda s: code in json.loads(s)
            )
            subset = valid[mask]
            if len(subset) >= 3:
                print(f"  {code:3s} : {subset['f1'].mean():.3f}  (n={len(subset)})")


if __name__ == "__main__":
    main()
