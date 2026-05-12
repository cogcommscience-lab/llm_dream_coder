#!/usr/bin/env python3
"""
emotions.py — Hall/Van de Castle Emotions Coder

Semi-automated classifier for the Emotions category of the H/VdC coding system.
Codes which characters explicitly experience one of five emotional states in a
dream report, then evaluates against human-coded ground truth.

Usage:
    python emotions.py                                        # 50 norms-f dreams (dev partition)
    python emotions.py --n 20                                 # sample of 20 dreams
    python emotions.py --all                                  # full dataset
    python emotions.py --collection norms-f                   # one collection
    python emotions.py --dream-id norms-f_0001               # single dream
    python emotions.py --skip 50 --n 50                       # dreams 51–100 (validation partition)

Output:
    emotions_results.csv — per-dream results with predicted and ground-truth
    character lists for each of the five emotion types, plus F1 for each.

Notes on evaluation:
    Each emotion coding is (type, character). Per-type F1 is computed at the
    attribute level: predicted and GT character lists for each type are
    decomposed into (slot, value) tuples — number, gender, identity, age —
    then scored via Counter intersection. Overall F1 decomposes all
    (emotion_type, char_attrs) pairs together for a single pooled score.
    mean_f1 = average of five per-type F1s.

Methodology note:
    Emotions are coded in norms-f and norms-m (and in b-baseline). We use
    norms-f dreams 1–50 as the development partition. norms-m is reserved
    as the untouched final test set.
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
OUTPUT_CSV        = "emotions_results.csv"
SAMPLE_SIZE       = 50
SAMPLE_COLLECTION = "norms-f"
DELAY_SECONDS     = 0.3

EMOTION_TYPES = ["AP", "HA", "AN", "CO", "SD"]

# ─────────────────────────────────────────────────────────────────────────────
# CODEBOOK
# ─────────────────────────────────────────────────────────────────────────────
CODEBOOK = """
Hall/Van de Castle (H/VdC) Emotions Coding Rules
=================================================

## Overview

Code every instance in which a character (including the dreamer) is explicitly
described as experiencing one of five emotional states. The key requirement:
the emotion must be STATED in words — do not infer emotions from events or
actions. If the dream text uses explicit emotional language, code it; if you
must guess how someone felt from what happened to them, do NOT code it.

Approximately 75–80% of dreams will have at least one emotion coding.
Most coded dreams have 1–3 emotions. If you find yourself coding 5+, re-check
whether each is truly explicit emotional language vs. implied mood or action.

## The Dreamer
Use "D" for the dreamer.

## Character Codes
Use standard H/VdC character codes: NUMBER + GENDER + IDENTITY + AGE
  e.g., 1MKA (one known male adult), 1FKA (one known female adult),
        1FSA (one female stranger adult), 2MKA (two known male adults)
  Family codes where relationship is explicitly stated: M (mother), F (father),
  H (husband), W (wife), B (brother/sibling), etc.
  Animals: 1ANI / 2ANI

─────────────────────────────────────────────────────────────────────────────
## THE FIVE EMOTION TYPES

### AP — Apprehension
Fear, terror, panic, dread, nervousness, anxiety, apprehension, fright,
worry, embarrassment (as social fear), horror, alarm, being upset or disturbed
in the context of a threat or frightening situation, hysteria in response to
danger or loss.
  Examples: "I was afraid," "I felt terrified," "I panicked," "I was nervous,"
  "I dreaded," "I was embarrassed," "I felt apprehensive," "I was hysterical,"
  "I was upset [by the threatening situation]," "I was very disturbed [by what
  I saw]."
  Note: "upset," "disturbed," and "hysterical" code as AP — they describe
  distress/shock, which H/VdC treats as apprehension. Do NOT code these as SD.

### HA — Happiness
Joy, happiness, pleasure, contentment, excitement, elation, satisfaction,
delight, gladness.
  Examples: "I was happy," "I felt joyful," "I was excited," "I felt pleased,"
  "I was quite satisfied," "she seemed delighted."
  Note: Relief codes as HA only when experienced WITHIN the dream itself, not
  upon waking. "When I woke up I was relieved it was a dream" = do NOT code.

### AN — Anger
Anger, rage, annoyance, disgust, hostility, irritation, fury, indignation,
resentment.
  Examples: "I was angry," "I felt disgusted," "I was irritated," "I was furious,"
  "I resented it," "she seemed hostile."

### CO — Confusion
Confusion, bewilderment, puzzlement, surprise, perplexity, disorientation,
astonishment, disbelief, finding something peculiar or strange as a personal
reaction (not merely a description of the dream's chaotic content).
  Examples: "I was confused," "I felt bewildered," "I was puzzled," "I was
  surprised," "I couldn't understand it," "I was perplexed," "this was peculiar
  to me," "I found it very strange."
  Note: "Everything was confused" or "the dream was chaotic" describes the
  narrative setting — NOT the dreamer's emotional state. Code CO only when the
  character explicitly registers their own puzzlement.

### SD — Sadness
Sadness, grief, sorrow, unhappiness, depression, disappointment, despair,
crying/weeping when accompanied by a stated loss or grief label.
  Examples: "I was sad," "I felt grief," "I was crying [over the loss],"
  "I felt devastated," "I was disappointed," "I felt sorrowful."
  Note: "Upset," "disturbed," and "distressed" alone do NOT code as SD — they
  code as AP. SD requires an explicit sadness or grief word.

─────────────────────────────────────────────────────────────────────────────
## CODING RULES

### Explicit language required
Code ONLY when the dream text uses an explicit emotional word or phrase.
Do not infer how a character must have felt from what happened to them.
  - "I was afraid" → AP ✓
  - Character is attacked by a monster → do NOT code AP unless fear is stated ✗

### The five types are exhaustive
If an emotion doesn't fit one of the five categories, do not code it.
Ambiguous or borderline emotions belong to the closest type or are not coded.

### Multiple codings per character per dream — same-scene rule
The same character can receive multiple codings of the SAME type ONLY if the
emotion arises from a NEW, DISTINCT situation or scene. The test is whether the
TRIGGER changes — not whether a new sentence appears.

When the SAME threat or cause produces two emotional expressions, even if
phrased differently and even if separated by a few sentences, it is ONE coding:
  - "I became frightened [at the stranger] … I was afraid he would shoot me"
    (same stranger, same confrontation) → ONE AP for D
  - "I was getting panicky … I became so frightened" (same nightmare, same
    threatening content, escalating in the same scene) → ONE AP for D
  - "She was completely happy and content … she was happy that she had him
    to herself" (same character's happiness about the same relationship) → ONE
    HA for her
  - "I was angry and disgusted about the article" → ONE AN for D
  - "I was very disturbed … I was very upset and felt that I had lost him"
    (same boyfriend in the same scene, walking away) → ONE AP for D

A second coding IS warranted when:
  - A clear scene change or time break separates the two emotional moments, AND
  - The trigger is a NEW situation (not the same ongoing threat or loss)
  - "I was embarrassed at the party [scene 1] … later at the hotel I got
    nervous on our wedding night [scene 2]" → TWO AP for D ✓

### Coding emotions for non-dreamer characters
Code emotions for other characters when the dream EXPLICITLY states their
inner state — typically through mental-state verbs, stated feelings, or
clear behavioral labels.
  - "He seemed terribly pleased with himself" → HA for that character ✓
  - "She appeared worried" → AP for that character ✓
  - "The man came at me aggressively" → do NOT infer his anger ✗

─────────────────────────────────────────────────────────────────────────────
## DO NOT CODE

  – Emotions inferred from events: falling into water, being chased, losing
    something — these are situations, not stated emotions. Only code the
    emotional label itself.

  – Physical sensations: pain, pleasure, discomfort, physical excitement
    (heart pounding, shaking, trembling) unless accompanied by an explicit
    emotional label ("I was shaking with fear" → AP; "my heart was pounding"
    alone → do NOT code).

  – Atmosphere or setting descriptions: "the house felt ominous," "the air
    was heavy," "it seemed threatening" — these describe the dream environment,
    not a character's stated emotion.

  – Narrative tone or implied mood: the overall tone of a dream being scary or
    pleasant does not count. Each emotion must be explicitly attributed to a
    character in the text.

  – Cognitive states that are not emotions: thinking, deciding, realizing,
    knowing, wondering (unless the wondering expresses explicit puzzlement/
    confusion). "I thought about it" or "I realized" are not emotions.

  – Actions and behaviors that are not stated emotions: running, fighting,
    smiling, laughing, crying alone — these are behaviors. "I smiled" or
    "I was crying" without an explicit emotion label (e.g., "I was crying
    because I was devastated") do NOT code. Code only the emotion word itself.

  – Emotions experienced upon waking: statements about how the dreamer felt
    after waking up are NOT dream emotions and must not be coded.
    "When I woke up I was relieved it was a dream" → do NOT code HA.
    "I was afraid to go back to sleep" (said after waking) → do NOT code AP.
    Code only emotions experienced within the dream itself.

  – Absence of emotion: "he didn't seem upset," "she wasn't worried" — these
    state non-emotions and should NOT be coded.

  – Reported speech about emotion without the character experiencing it: "she
    told me she was scared last week" — the character is not experiencing the
    emotion in the dream.

  – Narrative descriptions of the dream as confused or chaotic: "everything
    was very confused," "the dream was hazy and jumbled" — these describe the
    dream's content structure, not a character's emotional state. CO requires
    the character explicitly noting their own puzzlement or bewilderment.

  – Vague intuitions and perceptual senses without an emotion word: "I had the
    feeling that someone was there," "I sensed something was wrong," "it seemed
    threatening" — these describe perceptions or intuitions, not explicitly
    labeled emotions. Code only when an actual emotion word appears (afraid,
    nervous, confused, etc.).

  – Confusion language embedded inside another emotion's explanation: when
    "couldn't understand" or "puzzled" appears as part of the grievance or
    content of an already-coded emotion ("I was angry… and furthermore I
    couldn't understand how they got my information"), do NOT code a separate
    CO. It is part of the anger episode, not an independent confusion event.

  – Mild or ambiguous disappointment below the sadness threshold: words like
    "thwarted," "let down," "anticlimactic," or "a bit disappointed" that
    describe a mild letdown rather than explicit grief or sadness do NOT reach
    the SD threshold. SD requires an explicit sadness/grief word ("sad,"
    "grief," "sorrow," "devastated," "heartbroken") or crying tied to a stated
    loss.

─────────────────────────────────────────────────────────────────────────────
## OUTPUT FORMAT

Respond with ONLY valid JSON — no prose before or after — in this exact shape:

{
  "AP": ["D", "1FKA"],
  "HA": ["D"],
  "AN": [],
  "CO": ["1MOA"],
  "SD": [],
  "reasoning": {
    "D AP (felt terrified in hallway)": "Dreamer explicitly stated she felt terrified — AP.",
    "1FKA AP (she appeared worried)": "Known female adult described as appearing worried — AP.",
    "D HA (felt relieved at the end)": "Dreamer said she felt relief when she woke — HA.",
    "1MOA CO (puzzled by the result)": "Male older adult expressed puzzlement — CO.",
    "NOT CODED — being chased by dog": "Situation, not a stated emotion. No fear word used."
  }
}

Field rules:
  - All five keys (AP, HA, AN, CO, SD) must always appear, even if empty.
  - Each value is a list of character codes who experienced that emotion.
    Use "D" for the dreamer; standard H/VdC codes for others.
  - If a character experienced the same emotion in two distinct episodes,
    list them twice: "AP": ["D", "D"].
  - If there are no emotion codings at all:
    {"AP": [], "HA": [], "AN": [], "CO": [], "SD": [], "reasoning": {}}
  - Include reasoning for each coded emotion AND for any near-miss you
    explicitly decided NOT to code.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES  (drawn from norms-f dev partition: dreams 1–50)
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT = [
    # Example 1: HA only — simple happiness from achieving a goal
    {
        "dream": (
            "I dreamed that I was walking up Euclid Avenue on a shopping trip. I was "
            "going into the small specialty shops such as Quinn-Maah's, Milgrims, etc. "
            "looking for bargains in lingerie. In each store there was a counter with a "
            "large pile of pink slips with fancy lace trimming and in each store, after "
            "looking through the pile and disarranging it, I walked out. Finally I came "
            "to an unidentified shop operated by two elderly women. There I found what I "
            "was looking for at a much lower price and selected several yellow slips and "
            "a pair of gold handing earrings. I was quite satisfied."
        ),
        "AP": [], "HA": ["D"], "AN": [], "CO": [], "SD": [],
        "reasoning": {
            "D HA (quite satisfied)": (
                "The dreamer explicitly states 'I was quite satisfied' after finding what "
                "she was looking for — a direct happiness/contentment label."
            ),
            "NOT CODED — the shopping activities": (
                "Walking, looking through piles, entering and leaving stores are activities "
                "(locomotion, visual acts), not emotional states."
            ),
            "NOT CODED — the elderly women": (
                "No emotional states are described for the shop operators. Do not infer "
                "their mood from the description of the shop."
            ),
        },
    },
    # Example 2: SD only — explicit grief and crying at mother's death
    {
        "dream": (
            "Last night I dreamt that I received a phone call at the dorm and it was my "
            "father telling me that I shouldn't get excited, but my mother was killed. It "
            "seems she and my father were fishing on the lake, which they never do, and "
            "some sort of electric current from the motor electrocuted her. My father and "
            "little sister didn't seem to be terribly upset about the whole thing and I "
            "just couldn't understand it because I was crying and crying and kept calling "
            "home to find out if it were really so. I can remember walking in a cemetery "
            "which seemed like Forest Lawn Cemetery in Los Angeles and again I was crying. "
            "That's about all I remember except that I kept seeing my mother's face in "
            "front of me. I woke up in the middle of the night and felt as though I had "
            "been crying all night and I felt very upset."
        ),
        "AP": [], "HA": [], "AN": [], "CO": [], "SD": ["D"],
        "reasoning": {
            "D SD (crying repeatedly, felt very upset)": (
                "The dreamer cries repeatedly throughout the dream in response to her "
                "mother's death and wakes feeling 'very upset' — explicit grief and "
                "sadness. Crying here is clearly tied to a stated emotional distress."
            ),
            "NOT CODED — father and sister didn't seem terribly upset": (
                "This is a narrative observation about the ABSENCE of upset in other "
                "characters. Absence of an emotion is not coded. We cannot code SD "
                "for the father or sister based on this description."
            ),
            "NOT CODED — 'I just couldn't understand it'": (
                "This expresses the dreamer's reaction to her family's lack of grief, "
                "not a separate CO emotion episode. The predominant emotion is sadness "
                "(SD), which is already coded."
            ),
        },
    },
    # Example 3: CO + AN — two explicit emotions in one dream
    {
        "dream": (
            "I was in biology lab and had a bag of cookies and a bag of after dinner "
            "mints. I put them on the desk and said everyone was welcome to help "
            "themselves. When I went to get some, they were all gone, so I went around "
            "to each person in the lab and made them put some back. I was perplexed "
            "because they had taken advantage of my generosity and disgusted because "
            "some of them had made such pigs of themselves."
        ),
        "AP": [], "HA": [], "AN": ["D"], "CO": ["D"], "SD": [],
        "reasoning": {
            "D CO (perplexed)": (
                "The dreamer uses the word 'perplexed' — explicit confusion about why "
                "people took all the cookies despite her open invitation."
            ),
            "D AN (disgusted)": (
                "The dreamer explicitly says she was 'disgusted' — disgust falls within "
                "the Anger/Annoyance category."
            ),
            "NOT CODED — making everyone put cookies back": (
                "This is a physical/verbal activity (going around and making people "
                "comply) — it belongs to the Activities module. The emotional label is "
                "disgust, which is coded separately."
            ),
        },
    },
    # Example 4: CO for dreamer + HA for a non-dreamer character (doctor)
    {
        "dream": (
            "I dreamed I was in to see a doctor (a stranger) for consultation. The "
            "examination seemed of a trivial nature, but continued for quite some time. "
            "I was growing more puzzled by the procedure and wondering why he was taking "
            "so long. At last he seemed terribly pleased with himself and told me I had "
            "cancer. At the same time indicating because of his skill and early detection "
            "of the growth, complete cure was possible."
        ),
        "AP": [], "HA": ["1MOA"], "AN": [], "CO": ["D"], "SD": [],
        "reasoning": {
            "D CO (growing more puzzled)": (
                "The dreamer explicitly says 'I was growing more puzzled by the "
                "procedure' — direct confusion language."
            ),
            "1MOA HA (seemed terribly pleased with himself)": (
                "The doctor (coded 1MOA — one male older adult, stranger) is described "
                "as 'terribly pleased with himself' — an explicit happiness/satisfaction "
                "label. Other characters' emotions are coded when explicitly stated."
            ),
            "NOT CODED — dreamer's reaction to cancer diagnosis": (
                "The dream does not state what the dreamer felt upon hearing the cancer "
                "news. No fear, sadness, or relief word is used. The doctor immediately "
                "notes a complete cure is possible, and no emotional label is given to "
                "the dreamer at that point."
            ),
        },
    },
    # Example 5: Two separate AP episodes + one HA in same dream
    {
        "dream": (
            "I dreamed it was next summer and that I was going to be married to my "
            "boyfriend at home. Mother advised me not to, but said she would not stand "
            "in my way. He had very little money and George, after finally convincing me "
            "not to finish school, said maybe mother was right. I had made up my mind, "
            "however, and would not hear of a postponement. The next part is after we "
            "are married. Although I did not see the ceremony, I knew it was very simple. "
            "Someone asked where we should go on our honeymoon and I can't remember where "
            "I said, but they thought it was the place in Europe named the same. I was "
            "embarrassed when I said it was not. Then I thought maybe I should have "
            "married another fellow I know who would have taken me to Europe, but I "
            "immediately dismissed this from my mind. Mother said then that the place we "
            "were going was very nice, although neither of us had previously been there. "
            "This made me feel much better. Next we were sitting in a large dining room. "
            "George was drinking a beer and I had a coke. All of a sudden it struck me "
            "that this was my wedding night and I got nervous and sort of afraid. I then "
            "asked George to order me a double shot which I never did get to drink "
            "because I woke up."
        ),
        "AP": ["D", "D"], "HA": ["D"], "AN": [], "CO": [], "SD": [],
        "reasoning": {
            "D AP #1 (embarrassed about the honeymoon mix-up)": (
                "The dreamer explicitly says 'I was embarrassed' — embarrassment is a "
                "form of social apprehension, coded as AP."
            ),
            "D AP #2 (got nervous and sort of afraid on wedding night)": (
                "The dreamer says 'I got nervous and sort of afraid' — explicit fear/"
                "nervousness language, a second distinct AP episode later in the dream. "
                "Two separate AP episodes for the same character → list D twice."
            ),
            "D HA (this made me feel much better)": (
                "The dreamer explicitly says 'this made me feel much better' after "
                "mother's reassurance about the honeymoon destination — positive emotion."
            ),
            "NOT CODED — doubting the marriage / thinking about the other fellow": (
                "The dreamer thinks about whether she should have married someone else "
                "but 'immediately dismissed this' — a brief cognitive act, not a stated "
                "emotion."
            ),
            "NOT CODED — George drinking beer / the wedding ceremony": (
                "No emotional states are attributed to George or anyone else in these "
                "scenes."
            ),
        },
    },
    # Example 6: No emotions — mundane family dream with no emotional language
    {
        "dream": (
            "I was at home but it was not our house. I saw my little brother (2) first "
            "and he looked as if he had just had a haircut. Then I went into what seemed "
            "to be the living room and my mother and cousin (29) Jack were there. Jack "
            "was planning a vacation to Canada. I told him he was foolish to drive all "
            "that distance since it was snowing so hard. As I said that, I walked to the "
            "window, pushed back the curtain, and watched the snow flakes, which were "
            "extremely large, fall to the ground. Complete change then. I'm in front of "
            "a mirror shaving under my arms, but the hair seems to be at least 6 in. long."
        ),
        "AP": [], "HA": [], "AN": [], "CO": [], "SD": [],
        "reasoning": {
            "NOT CODED — telling Jack he was foolish": (
                "The dreamer expresses an opinion/judgment, not an explicitly labeled "
                "emotion. 'Foolish' describes the plan, not the dreamer's emotional state."
            ),
            "NOT CODED — watching the snow fall": (
                "Purely observational. No emotional language is attached to this moment."
            ),
            "NOT CODED — the unusual hair at the mirror": (
                "Surreal content with no stated emotional reaction. The dreamer does not "
                "say she was surprised, confused, amused, or anything else."
            ),
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    dreams  = pd.read_csv(INPUT_CSV)
    codings = pd.read_csv(CODINGS_CSV)

    emot_df = codings[codings["coding_type"] == "emot"].copy()

    gt_by_dream = {}
    for dream_id, group in emot_df.groupby("dream_id"):
        entry = {et: [] for et in EMOTION_TYPES}
        for _, row in group.iterrows():
            code = str(row.get("code", "")).strip()
            char = str(row.get("char", "")).strip()
            if code in EMOTION_TYPES and char and char != "nan":
                entry[code].append(char)
        gt_by_dream[dream_id] = entry

    return dreams, gt_by_dream


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────────────────
def build_system_content():
    ex_text = ""
    for ex in FEW_SHOT:
        output = {
            "AP":        ex["AP"],
            "HA":        ex["HA"],
            "AN":        ex["AN"],
            "CO":        ex["CO"],
            "SD":        ex["SD"],
            "reasoning": ex["reasoning"],
        }
        ex_text += f"\nDream:\n{ex['dream']}\n\n"
        ex_text += f"Output:\n{json.dumps(output, indent=2, ensure_ascii=False)}\n"
        ex_text += "\n---\n"

    return (
        CODEBOOK
        + "\n\n"
        + "## Worked Examples\n"
        + ex_text
        + "\n\nRespond with ONLY valid JSON matching the output format above.\n"
        + "If no emotions occur in the dream, return: "
        + '{"AP": [], "HA": [], "AN": [], "CO": [], "SD": [], "reasoning": {}}\n'
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(client, dream_text, system_content):
    """Call Claude and return (pred_dict, reasoning_dict)."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
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
                    "Code all emotions in this dream report:\n\n"
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

    parsed = json.loads(raw)
    pred = {et: parsed.get(et, []) for et in EMOTION_TYPES}
    return pred, parsed.get("reasoning", {})


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def decompose_character(code):
    """Decompose one H/VdC character code into (slot, value) tuples."""
    code = str(code).strip()
    if code == "D":
        return [("char", "D")]
    if "ANI" in code:
        return [("num", code.replace("ANI", "")), ("type", "ANI")]
    if "CZZ" in code:
        return [("num", code.replace("CZZ", "")), ("type", "CZZ")]
    if len(code) == 4:
        return [
            ("num", code[0]),
            ("gen", code[1]),
            ("idt", code[2]),
            ("age", code[3]),
        ]
    return [("char", code)]


def evaluate_attribute(pred_chars, gt_chars):
    """Attribute-level F1 over a list of character codes."""
    pred = Counter(item for c in pred_chars for item in decompose_character(c))
    true = Counter(item for c in gt_chars  for item in decompose_character(c))
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


def evaluate_overall(pred_dict, gt_dict):
    """
    Overall attribute-level F1 across all emotion types combined.
    Each (emotion_type, char) pair decomposes into:
      ("emotion_type", code) + char attribute tuples.
    """
    def decompose_all(d):
        items = []
        for code, chars in d.items():
            for char in chars:
                items.append(("emotion_type", code))
                items.extend(decompose_character(char))
        return Counter(items)

    pred = decompose_all(pred_dict)
    true = decompose_all(gt_dict)
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
    parser = argparse.ArgumentParser(description="H/VdC Emotions Coder")
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

        if pd.isna(dream_text):
            print(f"  SKIP  {dream_id}  (missing report text)")
            continue

        gt = gt_by_dream.get(dream_id, {et: [] for et in EMOTION_TYPES})

        try:
            pred, reasoning = call_claude(client, str(dream_text), system_content)
        except Exception as e:
            print(f"  ERROR {dream_id}: {e}")
            error_row = {"dream_id": dream_id, "reasoning": f"ERROR: {e}",
                         "dream_report": str(dream_text)[:300]}
            for et in EMOTION_TYPES:
                error_row[f"pred_{et}"] = "[]"
                error_row[f"gt_{et}"]   = json.dumps(gt.get(et, []))
                error_row[f"f1_{et}"]   = None
                error_row[f"precision_{et}"] = None
                error_row[f"recall_{et}"]    = None
            error_row["mean_f1"]           = None
            error_row["f1_overall"]        = None
            error_row["precision_overall"] = None
            error_row["recall_overall"]    = None
            results.append(error_row)
            time.sleep(DELAY_SECONDS)
            continue

        # Per-type F1
        per_type = {}
        for et in EMOTION_TYPES:
            per_type[et] = evaluate_attribute(pred.get(et, []), gt.get(et, []))

        # Overall F1
        overall = evaluate_overall(pred, gt)
        mean_f1 = round(sum(per_type[et]["f1"] for et in EMOTION_TYPES) / 5, 3)

        has_gt  = any(len(gt.get(et, [])) > 0 for et in EMOTION_TYPES)
        marker  = "" if has_gt else "  [no gt]"
        print(
            f"  {'✓' if mean_f1 == 1.0 else '✗'}  {dream_id}"
            f"  pred={{{', '.join(et+':'+str(pred.get(et,[])) for et in EMOTION_TYPES if pred.get(et))}}}"
            f"  gt={{{', '.join(et+':'+str(gt.get(et,[])) for et in EMOTION_TYPES if gt.get(et))}}}"
            f"  mean_f1={mean_f1:.2f}  overall={overall['f1']:.2f}{marker}"
        )

        result_row = {"dream_id": dream_id}
        for et in EMOTION_TYPES:
            result_row[f"pred_{et}"]      = json.dumps(pred.get(et, []),    ensure_ascii=False)
            result_row[f"gt_{et}"]        = json.dumps(gt.get(et, []),      ensure_ascii=False)
            result_row[f"f1_{et}"]        = per_type[et]["f1"]
            result_row[f"precision_{et}"] = per_type[et]["precision"]
            result_row[f"recall_{et}"]    = per_type[et]["recall"]
        result_row["mean_f1"]           = mean_f1
        result_row["f1_overall"]        = overall["f1"]
        result_row["precision_overall"] = overall["precision"]
        result_row["recall_overall"]    = overall["recall"]
        result_row["reasoning"]         = json.dumps(reasoning, ensure_ascii=False)
        result_row["dream_report"]      = str(dream_text)[:300]
        results.append(result_row)

        time.sleep(DELAY_SECONDS)

    # ── Save results ──────────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────────────────────
    valid   = df_out[df_out["mean_f1"].notna()]
    n_valid = len(valid)

    print(f"\n{'─'*60}")
    print(f"Results saved → {OUTPUT_CSV}")
    print(f"Dreams evaluated : {n_valid}")

    if n_valid > 0:
        print(f"\nPer-type F1:")
        for et in EMOTION_TYPES:
            col = f"f1_{et}"
            print(f"  {et}  : {valid[col].mean():.3f}"
                  f"  (p={valid[f'precision_{et}'].mean():.3f}"
                  f"  r={valid[f'recall_{et}'].mean():.3f})")
        print(f"\nMean F1 (avg of 5 types): {valid['mean_f1'].mean():.3f}")
        print(f"Overall F1 (pooled)      : {valid['f1_overall'].mean():.3f}")
        print(f"  Precision              : {valid['precision_overall'].mean():.3f}")
        print(f"  Recall                 : {valid['recall_overall'].mean():.3f}")

        has_gt = valid[valid[[f"gt_{et}" for et in EMOTION_TYPES]].apply(
            lambda r: any(r[f"gt_{et}"] != "[]" for et in EMOTION_TYPES), axis=1
        )]
        if len(has_gt) > 0:
            print(f"\nDreams with any GT emotion coding: {len(has_gt)}")
            print(f"  Mean F1  : {has_gt['mean_f1'].mean():.3f}")
            print(f"  Overall F1: {has_gt['f1_overall'].mean():.3f}")


if __name__ == "__main__":
    main()
