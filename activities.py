#!/usr/bin/env python3
"""
activities.py — Hall/Van de Castle Activities Coder

Semi-automated classifier for the Activities category of the H/VdC coding system.
Codes all character activities — physical, verbal, visual, cognitive, etc. — in
dream reports using Claude, then evaluates against human-coded ground truth.

Usage:
    python activities.py                                       # sample of 50 norms-f dreams (dev partition)
    python activities.py --n 15                                # sample of 15 dreams
    python activities.py --all                                 # full dataset
    python activities.py --collection norms-f                  # one collection
    python activities.py --dream-id norms-f_0001               # single dream

Output:
    activities_results.csv — per-dream results with predicted and ground-truth
    activity tuples, plus F1 overall and broken out by sub-type (P/M/V/S/L/C/E/A).

Notes on evaluation:
    Each activity is a tuple (init, rec, code). F1 is computed at the attribute
    level: each tuple is decomposed into init/rec character attributes, sub-type,
    and direction, then scored via Counter intersection. This gives partial
    credit when sub-type or direction is right but the character is mis-coded.

Methodology note:
    Activities have ground-truth codings ONLY in norms-f and norms-m. We use
    norms-f dreams 1-50 as a development partition for iteration. norms-m is
    reserved as the untouched final test set.
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
OUTPUT_CSV        = "activities_results.csv"
SAMPLE_SIZE       = 50
SAMPLE_COLLECTION = "norms-f"
DELAY_SECONDS     = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# CODEBOOK
# ─────────────────────────────────────────────────────────────────────────────
CODEBOOK = """
Hall/Van de Castle (H/VdC) Activities Coding Rules
===================================================

## Overview

Code every distinct activity performed by every character (and the dreamer)
in the dream. An activity is something a character DOES — a physical act,
movement, verbal exchange, observation, or mental act. Activities are coded
separately from social interactions (agg/fri/sex). The same event may
generate both an activity code and a social-interaction code in different
modules; this module codes only activities.

## The Dreamer
Use "D" for the dreamer when she/he performs an activity. The dreamer is
treated like any other character for activity coding.

## Character Codes
Use the same H/VdC character codes as the Characters module:
  Format: NUMBER + GENDER + IDENTITY + AGE  (e.g., 1MKA, 2IEA, 1FMA)
  Animals: 1ANI / 2ANI
  Joint actors performing one activity together: separate with " + "
    (e.g., "D + 1MKA" — dreamer and a known male adult acting together)

─────────────────────────────────────────────────────────────────────────────
## EIGHT ACTIVITY SUB-TYPES (single letters)

Each activity has a single sub-type letter:

  P = PHYSICAL — active bodily manipulation using the body purposefully
        Examples: eating, drinking, dressing, undressing, hitting, kissing,
        embracing, building, cooking, manipulating objects, washing,
        grabbing, opening doors, packing, picking up, putting down,
        physically struggling/striving against resistance.
        DO NOT code P for:
          – Passive states/postures: sitting, standing, lying down, resting,
            waiting. These are not activities — they have no action component.
          – Entering or leaving a place: "walked into the room," "entered the
            building" — use M (locomotion on foot) instead.
          – Handing or offering objects: "he held out a letter," "she handed
            me X" — minor incidental gestures are not P or P>.
          – Sexual acts mentioned as past events or backstory (not actively
            happening in the dream scene): "she had slept with him" (prior to
            the scene) is NOT coded. Sexual acts actively and explicitly
            occurring in the dream scene are coded P= (and also in Social
            Interactions as sex coding).
          – Vague or implied physical acts with no explicit description:
            "they were performing," "he examined me" (if the exam is not
            explicitly described as a physical act) — only code P when the
            physical act is clearly stated.
          – Vague physical presence: "we were there," "stayed overnight."

  M = MOVEMENT — locomotion ON FOOT (or by self-propelled body movement)
        Examples: walking, running, climbing, jumping, swimming, falling,
        crawling, entering/leaving a place on foot.
        Going somewhere or moving from place to place using one's own body.

  L = LOCATION — locomotion BY VEHICLE (a vehicle carries the character)
        Examples: driving a car, riding a bicycle, riding a motorcycle,
        flying in a plane, riding a horse, taking a bus, sailing.
        Distinguishing rule: M is on foot, L involves a vehicle.

  V = VERBAL — speaking, talking, asking, telling, saying anything aloud
        Examples: "I told him X," "she said," "I asked," "he replied,"
        announcing, greeting, lecturing, conversing.

  S = VISUAL — explicit acts of looking, watching, or seeing as a primary
        foreground activity.
        Examples: "I saw him enter," "I looked at it carefully," "I watched
        what was happening," "I was looking out the window," "I could see
        that he had changed."
        DO NOT code S for:
          – Setting/environment descriptions that are narrative backdrop:
            "the room had tables everywhere," "it was dark," "there were
            many people around" — the dreamer is not actively looking at
            these; they are scene-setting.
          – Character appearance descriptions: "she wore a fur coat," "he
            was overweight" — describing what someone looks like is not an
            act of looking.
          – Implied visual awareness: simply being present in a scene and
            perceiving it implicitly is not coded as S. Code S only when
            looking/watching is the stated primary act of that moment.

  A = AUDITORY — hearing, listening, perceiving sound
        Examples: "I heard," "I listened," "I could hear my name being
        called," sound perception of any kind.

  E = EXPRESSIVE — non-verbal emotional expression
        Examples: laughing, crying, smiling, frowning, screaming, sighing,
        gasping, sobbing. Emotional expression that is NOT verbal speech.

  C = COGNITIVE — deliberate internal mental acts where thinking IS the activity
        Code C when the act of thinking, deciding, or deliberating is itself
        the foreground event in the dream.
        DO code C for: deciding ("I made up my mind," "we decided"), plotting
        or planning as a primary activity ("we plotted," "we figured out"),
        actively thinking through a problem or situation ("I thought about
        what to do," "I thought this was peculiar," "I was trying to figure
        out," "I wondered why [X]," "I was wondering why [X] would/could").
        DO NOT code C for:
          – Dream realizations and narrative transitions: "I realized," "I
            suddenly realized," "I became aware," "I knew" (passive knowledge),
            "I noticed," "the next thing I knew," "it seemed to me," "somehow,"
            "I remember [that / doing X]" used as narrative recall (not an act
            of remembering WITHIN the dream scene).
          – Emotional states: "I was afraid," "I was confused," "I was
            astonished," "I was nervous" — these are emotions, not cognitive
            activities. NOTE: feeling puzzled or growing puzzled is an emotion;
            but "wondering why" or "trying to figure out why" is C.
          – Background plans mentioned in conversation: if someone mentions
            their plans while talking, code the talking as V>, not the
            planning as C.

─────────────────────────────────────────────────────────────────────────────
## DIRECTION MODIFIERS (appended after sub-type letter)

  (none) = solo activity, no recipient. Set rec = null.
            Examples: "I walked" → D: M ; "I thought" → D: C
  >      = directed at a recipient (one-way)
            Examples: "I told him" → D: V> → 1MKA ; "He hit me" → 1MKA: P> → D
  =      = mutual or together (both parties act jointly OR symmetrically)
            Examples: "we talked together" → D + 1MKA: V= (joint actor)
                  OR  "we kissed each other" → D: P= → 1MKA
            Use joint actors (D + 1MKA) for symmetric joint activities.
  R      = reciprocated/response (this character is responding to a prior
            activity from the recipient)
            Examples: "He asked X. I answered Y" → first 1MKA: V> → D,
            then D: VR → 1MKA. R marks the responder's reciprocal turn.

Some Dreambank entries use lowercase "r" for the same response meaning;
treat uppercase R and lowercase r as equivalent. Use uppercase R in output.

─────────────────────────────────────────────────────────────────────────────
## WHAT IS CODEABLE

Code an activity when it is:
  - Physically enacted in the dream (the character does it in the scene)
  - Stated as a clear act ("I walked," "she said," "he laughed")
  - Performed by the dreamer or any other character

Each distinct activity gets its own tuple. If the same character performs
the same sub-type multiple times (e.g., multiple separate sentences of
walking around), the human standard is to code each TEMPORALLY-DISTINCT
instance separately. But continuous description of one act across multiple
sentences = ONE code.

## WHAT IS NOT CODEABLE

Do NOT code:
  - States or conditions: "She was tall," "He was tired," "I was nervous."
    States are not activities. Activities require an ACT.
  - Setting/location descriptions: "The room was dark," "It was a sunny day."
  - Internal feelings as activities (those belong to Emotions module):
    "I felt afraid" is NOT C — it's an emotion. C is for cognitive ACTS
    like thinking, deciding, remembering. Feeling fear is not coded here.
  - Generic descriptions of "what was happening": "There was a party,"
    "Music was playing." Code only specific characters performing specific
    acts.
  - Implied or background activities not stated in the text.

## FREQUENCY GUIDANCE

A typical dream report has 3–10 activities, with denser dreams reaching
~20. If you produce far fewer or far more codes than this range, re-check
your work.

─────────────────────────────────────────────────────────────────────────────
## OUTPUT FORMAT

Respond with ONLY valid JSON — no prose before or after — in this exact shape:

{
  "activities": [
    {"init": "D",         "rec": null,    "code": "M"},
    {"init": "D",         "rec": "1MKA",  "code": "V>"},
    {"init": "1MKA",      "rec": "D",     "code": "VR"},
    {"init": "D + 1MKA",  "rec": null,    "code": "L"},
    {"init": "D",         "rec": null,    "code": "C"}
  ],
  "reasoning": {
    "D M (walking)":        "Dreamer walks down the hall — solo movement on foot.",
    "D V> 1MKA (told him)": "Dreamer tells a known male adult something — directed verbal.",
    "1MKA VR D (replied)":  "He replied to the dreamer — verbal response/reciprocation.",
    "D + 1MKA L (drove)":   "Dreamer and Mark drove together — joint vehicle locomotion.",
    "D C (thought)":        "Dreamer thought about what to do — solo cognitive act."
  }
}

Field rules:
  - "init": character code or "D". Joint actors: "D + 1MKA" or "1FKA + 1MKA".
  - "rec": character code, "D", or null (for solo activities).
  - "code": single sub-type letter (P/M/V/S/L/C/E/A) plus optional
    direction modifier (>/=/R). Examples: "M", "V>", "P=", "VR".
  - If the dream has no codeable activities (very rare), return:
    {"activities": [], "reasoning": {}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES  (drawn from norms-f dev partition: dreams 1-10)
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT = [
    {
        "dream": (
            "I dreamed it was next summer and that I was going to be married to my "
            "boyfriend at home. Mother advised me not to, but said she would not stand in "
            "my way. He had very little money and George, after finally convincing me not "
            "to finish school, said maybe mother was right. I had made up my mind, however, "
            "and would not hear of a postponement. The next part is after we are married. "
            "Although I did not see the ceremony, I knew it was very simple. Someone asked "
            "where we should go on our honeymoon and I can't remember where I said, but "
            "they thought it was the place in Europe named the same. I was embarrassed when "
            "I said it was not. Then I thought maybe I should have married another fellow "
            "I know who would have taken me to Europe, but I immediately dismissed this "
            "from my mind. Mother said then that the place we were going was very nice, "
            "although neither of us had previously been there. This made me feel much "
            "better. Next we were sitting in a large dining room. George was drinking a "
            "beer and I had a coke. All of a sudden it struck me that this was my wedding "
            "night and I got nervous and sort of afraid. I then asked George to order me "
            "a double shot which I never did get to drink because I woke up."
        ),
        "activities": [
            {"init": "1FMA", "rec": "D",    "code": "V>"},
            {"init": "1MKA", "rec": "D",    "code": "V>"},
            {"init": "D",    "rec": None,   "code": "C"},
            {"init": "1IUA", "rec": "D",    "code": "V>"},
            {"init": "D",    "rec": None,   "code": "C"},
            {"init": "1FMA", "rec": "D",    "code": "V>"},
            {"init": "D",    "rec": "1MKA", "code": "P>"},
            {"init": "D",    "rec": None,   "code": "P"},
            {"init": "D",    "rec": "1MKA", "code": "V>"},
            {"init": "D",    "rec": "1IUA", "code": "VR"},
        ],
        "reasoning": {
            "1FMA V> D (mother advised)":     "Mother gives verbal advice to dreamer — directed verbal.",
            "1MKA V> D (George said)":        "George tells dreamer mother might be right — directed verbal.",
            "D C (made up mind)":             "Dreamer makes up her mind — cognitive act, no recipient.",
            "1IUA V> D (someone asked)":      "An unspecified person asks the dreamer — directed verbal.",
            "D C (thought / dismissed)":      "Dreamer thinks about marrying another fellow then dismisses — cognitive.",
            "1FMA V> D (mother said again)":  "Mother speaks again later in the dream — second V> instance.",
            "D P> 1MKA (drinking together?)": "Dreamer drinks a coke at the dining table — physical act.",
            "D P (drinking coke)":            "Dreamer's act of having a drink — physical.",
            "D V> 1MKA (asked George)":       "Dreamer asks George to order her a drink — directed verbal.",
            "D VR 1IUA (replied honeymoon)":  "Dreamer answers the question about honeymoon location — verbal response.",
            "NOT CODED — feelings (embarrassed, nervous, afraid)": (
                "These are emotions/feelings, not cognitive ACTS. Emotions belong to a "
                "separate category. Do not code 'I was embarrassed' as C."
            ),
        },
    },
    {
        "dream": (
            "I was riding a bicycle with a boy who is a student here at school and we "
            "were going to the movies. The surroundings were unfamiliar and we both "
            "suddenly realized there would be no place to put the bikes, so we returned. "
            "On the way back, however, he had to cross a bridge and when we got to the "
            "middle of it, it opened like a draw bridge, only too fast for us to get off. "
            "We both fell into the water, which seemed to be a canal. The water was very "
            "clear and green like in a swimming pool and it took a lot of effort getting "
            "to the surface."
        ),
        "activities": [
            {"init": "D",    "rec": None, "code": "L"},
            {"init": "1MKA", "rec": None, "code": "L"},
            {"init": "D",    "rec": None, "code": "P"},
            {"init": "1MKA", "rec": None, "code": "P"},
            {"init": "D",    "rec": None, "code": "P"},
            {"init": "1MKA", "rec": None, "code": "P"},
        ],
        "reasoning": {
            "D L (riding bike)":      "Dreamer rides a bicycle — vehicle-based locomotion → L (NOT M).",
            "1MKA L (riding bike)":   "Boy also rides a bicycle — same L code, separate actor.",
            "D P (fell, swam up)":    "Dreamer falls into water and pushes to the surface — physical bodily acts.",
            "1MKA P (fell, swam up)": "Boy falls and swims to the surface — physical acts.",
            "Repeated P entries":     "Multiple physical acts (falling, swimming up) coded for each actor.",
            "NOT CODED — surroundings unfamiliar / water clear and green": (
                "These are setting/environment descriptions, not activities. The visual "
                "appearance of the water is a state, not a character's act of seeing."
            ),
        },
    },
    {
        "dream": (
            "I dreamed that Norma (18), a girl who resides in the same dormitory as I, "
            "and I decided to change rooms. I was going to take the room I had last year "
            "and she was going to take my present room. I packed some of my things in a "
            "suitcase and went down to my old room. However, the present occupant of the "
            "room hadn't removed her things from the room. Since I had no place to put "
            "my things, I took my suitcase and went back to my present room. There I saw "
            "Norma had already moved in. She had changed some of the furniture around and "
            "was standing by the window ironing. I told her I couldn't get back my old "
            "room and she only laughed. I started to take her clothes out of the drawer "
            "and to put them in her suitcase. She had a lot of money lying on top of the "
            "dresser, which I also removed and put in the suitcase."
        ),
        "activities": [
            {"init": "1FKA",         "rec": None,   "code": "P"},
            {"init": "D",            "rec": None,   "code": "M"},
            {"init": "D",            "rec": None,   "code": "S"},
            {"init": "1FKA",         "rec": None,   "code": "P"},
            {"init": "D",            "rec": "1FKA", "code": "V>"},
            {"init": "D",            "rec": None,   "code": "P"},
            {"init": "D",            "rec": None,   "code": "P"},
            {"init": "D",            "rec": None,   "code": "P"},
            {"init": "D",            "rec": None,   "code": "M"},
            {"init": "1FKA",         "rec": None,   "code": "E"},
            {"init": "D + 1FKA",     "rec": None,   "code": "C"},
        ],
        "reasoning": {
            "D + 1FKA C (decided)":   "Dreamer and Norma jointly decide to change rooms — joint cognitive act.",
            "D M (went down/back)":   "Dreamer walks down to the old room and back — locomotion on foot.",
            "D S (saw Norma)":        "Dreamer sees Norma had moved in — visual perception.",
            "D V> 1FKA (told her)":   "Dreamer tells Norma she couldn't get the old room — directed verbal.",
            "1FKA E (laughed)":       "Norma laughs in response — expressive non-verbal emotional act.",
            "D P (packed, took, removed, put)": (
                "Multiple physical acts — packing, taking out clothes, removing money. "
                "Each temporally-distinct physical act is a separate P code."
            ),
            "1FKA P (changed furniture, ironing)": (
                "Norma rearranges furniture and irons — physical bodily acts."
            ),
            "NOT CODED — the room/dorm descriptions": (
                "Setting details ('a girl who resides in the same dormitory') are not "
                "activities. Only the actual acts within the dream scene are coded."
            ),
        },
    },
    {
        "dream": (
            "I dreamed I was in the shower one evening around 10:00 when I heard my name "
            "being called on the dormitory intercommunications system. I put on my "
            "housecoat and ran down three flights of stairs to the telephone. The person "
            "on the other end of the line was Howard M. (16, a boy whom I date). He said "
            "the reason he called was to tell me that a drugstore had opened up across "
            "the street from my father's store and would take away a lot of business "
            "from my father. He hung up then and I went upstairs and told the girls that "
            "I was very disgusted because Howard had called me up just to tell me that "
            "my father had competition from a drug store across the street and didn't "
            "ask me to go out."
        ),
        "activities": [
            {"init": "D",    "rec": None,   "code": "P"},
            {"init": "1MKA", "rec": "D",    "code": "V>"},
            {"init": "1MKA", "rec": None,   "code": "P"},
            {"init": "D",    "rec": "2FKA", "code": "V>"},
            {"init": "D",    "rec": None,   "code": "A"},
            {"init": "D",    "rec": None,   "code": "M"},
            {"init": "D",    "rec": None,   "code": "M"},
        ],
        "reasoning": {
            "D A (heard name)":     "Dreamer hears her name called on the intercom — auditory perception → A.",
            "D P (put on housecoat)": "Dreamer puts on her housecoat — physical bodily act.",
            "D M (ran down stairs)": "Dreamer runs down three flights of stairs — locomotion on foot → M.",
            "1MKA V> D (Howard called)": "Howard tells the dreamer about the drugstore — directed verbal.",
            "1MKA P (hung up)":      "Howard hangs up the phone — physical act.",
            "D M (went upstairs)":   "Dreamer walks back upstairs — locomotion on foot.",
            "D V> 2FKA (told girls)": "Dreamer tells a group of known females (the girls) — directed verbal.",
            "NOT CODED — 'I was very disgusted'": (
                "This is an emotion/feeling, not a coded activity. Feelings belong to "
                "the Emotions module."
            ),
            "NOT CODED — 'didn't ask me to go out'": (
                "A non-event (Howard didn't do something) is not a coded activity. "
                "Only acts that occur in the dream are coded."
            ),
        },
    },
    {
        "dream": (
            "I dreamed I was sitting in my room at my desk. The room did not look like "
            "mine, but rather like Doris' (18, a girl who lives in the room next to mine) "
            "room. As I was sitting at my desk looking out the window, I noticed a pen on "
            "my desk which was situated in a dish of ink. I knew that a man was in my room "
            "and I immediately ran out of the room. I asked one of the girls (I can't "
            "remember what she looked like) to come to my room and see if anyone was in "
            "the closet. We both entered the room, whereupon she opened the closet door "
            "and shut it again. Then she said I saw a man in the closet. As we fled from "
            "the room, the closet door opened and an old man darted out. We ran down the "
            "stairs, closely followed by the old man. Suddenly the other girl stopped and "
            "extended her foot. The old man stumbled over it and fell down the stairs."
        ),
        "activities": [
            {"init": "D",         "rec": None,   "code": "S"},
            {"init": "D",         "rec": None,   "code": "M"},
            {"init": "D",         "rec": "1FUA", "code": "V>"},
            {"init": "D + 1FUA",  "rec": None,   "code": "M"},
            {"init": "1FUA",      "rec": None,   "code": "P"},
            {"init": "1FUA",      "rec": "D",    "code": "V>"},
            {"init": "D + 1FUA",  "rec": None,   "code": "M"},
            {"init": "1MSA",      "rec": None,   "code": "M"},
            {"init": "1FUA",      "rec": None,   "code": "P"},
        ],
        "reasoning": {
            "D S (looking out window)": (
                "Dreamer looks out the window and notices a pen — visual perception → S. "
                "The noticing of the pen is part of the same visual scan; one S code."
            ),
            "D M (ran out of room)": "Dreamer immediately runs out — locomotion on foot → M.",
            "D V> 1FUA (asked a girl)": "Dreamer asks the other girl to come check — directed verbal.",
            "D + 1FUA M (entered room together)": "Both enter the room together — joint locomotion on foot → M.",
            "1FUA P (opened/shut closet door)": "The girl opens then shuts the closet door — physical act.",
            "1FUA V> D (said she saw a man)": "The girl tells the dreamer what she saw — directed verbal.",
            "D + 1FUA M (fled from room)": "Both flee — joint locomotion → M.",
            "1MSA M (darted out)": "The old man darts out of the closet — locomotion on foot → M.",
            "1FUA P (extended foot, tripped man)": "The girl extends her foot and trips the man — physical act.",
            "NOT CODED — 'sitting at my desk'": (
                "Sitting is a passive posture, not an activity. Do NOT code P for sitting, "
                "standing, lying down, waiting, or resting."
            ),
            "NOT CODED — 'I knew that a man was in my room'": (
                "This is passive awareness/knowledge, not a deliberate cognitive act. Do "
                "NOT code C for 'I knew,' 'I realized,' 'I became aware,' or 'it seemed "
                "to me.' These are narrative transitions, not cognitive activities."
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
    acts = codings[codings["coding_type"] == "act"].copy()

    def rows_to_activities(g):
        result = []
        for _, r in g.iterrows():
            rec = r["rec"]
            result.append({
                "init": str(r["init"]) if pd.notna(r["init"]) else None,
                "rec":  str(rec)        if pd.notna(rec)       else None,
                "code": str(r["code"])  if pd.notna(r["code"]) else None,
            })
        return result

    ground_truth = (
        acts.groupby("dream_id")
        .apply(rows_to_activities)
        .to_dict()
    )
    return dreams, ground_truth


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────────────────
def build_system_content():
    ex_text = ""
    for ex in FEW_SHOT:
        output = {"activities": ex["activities"], "reasoning": ex["reasoning"]}
        ex_text += f"\nDream:\n{ex['dream']}\n\n"
        ex_text += f"Output:\n{json.dumps(output, indent=2, ensure_ascii=False)}\n"
        ex_text += "\n---\n"

    return (
        CODEBOOK
        + "\n\n"
        + "## Worked Examples\n"
        + ex_text
        + "\n\nRespond with ONLY valid JSON matching the output format above.\n"
        + "If no activities occur in the dream, return: "
        + '{"activities": [], "reasoning": {}}\n'
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(client, dream_text, system_content):
    """Call Claude and return (activities_list, reasoning_dict)."""
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
                    "Code all activities in this dream report:\n\n"
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
    return parsed.get("activities", []), parsed.get("reasoning", {})


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def _to_tuple(activity):
    """Hashable tuple for whole-tuple comparison."""
    rec = activity.get("rec")
    return (
        str(activity.get("init", "")).strip(),
        str(rec).strip() if rec is not None else "SOLO",
        str(activity.get("code", "")).strip().upper(),  # normalize R/r
    )


def _decompose_char(code, role):
    """Decompose one character code into role-prefixed (slot, value) tuples."""
    code = str(code).strip()
    if "ANI" in code:
        return [(f"{role}_num", code.replace("ANI", "")), (f"{role}_type", "ANI")]
    if "CZZ" in code:
        return [(f"{role}_num", code.replace("CZZ", "")), (f"{role}_type", "CZZ")]
    if len(code) == 4:
        return [(f"{role}_num", code[0]), (f"{role}_gen", code[1]),
                (f"{role}_idt", code[2]), (f"{role}_age", code[3])]
    return [(f"{role}_raw", code)]


def decompose_activity(activity):
    """Flatten an activity dict into a list of (slot, value) tuples.

    Splits init/rec into character attributes, and code into subtype + direction.
    """
    items = []

    code = str(activity.get("code", "")).strip()
    if code:
        # Sub-type is the first character (P/M/V/S/L/C/E/A)
        # Direction is anything after, normalized to uppercase
        subtype = code[0].upper() if code else ""
        direction_raw = code[1:].upper() if len(code) > 1 else ""
        items.append(("subtype", subtype))
        items.append(("direction", direction_raw if direction_raw else "NONE"))

    for role in ("init", "rec"):
        val = activity.get(role)
        if val is None:
            items.append((f"{role}_special", "NULL"))
            continue
        sval = str(val).strip()
        if sval in ("D", "Q") or sval == "":
            items.append((f"{role}_special", sval or "NULL"))
            continue
        if "+" in sval:
            for part in sval.split("+"):
                items.extend(_decompose_char(part.strip(), role))
        else:
            items.extend(_decompose_char(sval, role))
    return items


def evaluate_attribute(predicted, ground_truth):
    pred = Counter(item for a in predicted    for item in decompose_activity(a))
    true = Counter(item for a in ground_truth for item in decompose_activity(a))
    tp = sum((pred & true).values())
    precision = tp / sum(pred.values()) if pred else (1.0 if not true else 0.0)
    recall    = tp / sum(true.values()) if true else (1.0 if not pred else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    pred_tuples = Counter(_to_tuple(a) for a in predicted)
    true_tuples = Counter(_to_tuple(a) for a in ground_truth)

    return {
        "exact_match": pred_tuples == true_tuples,
        "precision":   round(precision, 3),
        "recall":      round(recall,    3),
        "f1":          round(f1,        3),
    }


def evaluate_by_subtype(predicted, ground_truth, subtype_letter):
    """F1 restricted to activities of a given sub-type letter (e.g., 'P', 'M')."""
    def keep(a):
        c = str(a.get("code", "")).strip().upper()
        return c[:1] == subtype_letter
    pred_t = [a for a in predicted    if keep(a)]
    true_t = [a for a in ground_truth if keep(a)]
    return evaluate_attribute(pred_t, true_t)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
SUBTYPES = ["P", "M", "V", "S", "L", "C", "E", "A"]


def main():
    parser = argparse.ArgumentParser(description="H/VdC Activities Coder")
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

    dreams, ground_truth = load_data()

    # ── Select dreams ─────────────────────────────────────────────────────────
    n = args.n if args.n else SAMPLE_SIZE
    if args.dream_id:
        sample = dreams[dreams["dream_id"] == args.dream_id]
        if sample.empty:
            sys.exit(f"Dream ID '{args.dream_id}' not found.")
    elif args.all:
        sample = dreams
    elif args.collection:
        pool = dreams[dreams["collection_id"] == args.collection]
        sample = pool.iloc[args.skip:args.skip + n]
    else:
        pool = dreams[dreams["collection_id"] == SAMPLE_COLLECTION]
        sample = pool.iloc[args.skip:args.skip + n]

    print(f"Processing {len(sample)} dream(s) with model {MODEL}...\n")

    results = []

    for _, row in sample.iterrows():
        dream_id   = row["dream_id"]
        dream_text = row["dream_report"]

        if pd.isna(dream_text):
            print(f"  SKIP  {dream_id}  (missing report text)")
            continue

        gt_activities = ground_truth.get(dream_id, [])

        try:
            pred_activities, reasoning = call_claude(
                client, str(dream_text), system_content
            )
        except Exception as e:
            print(f"  ERROR {dream_id}: {e}")
            row_out = {
                "dream_id":     dream_id,
                "predicted":    "[]",
                "ground_truth": json.dumps(gt_activities),
                "exact_match":  False,
                "precision":    None,
                "recall":       None,
                "f1_attr":      None,
                "reasoning":    f"ERROR: {e}",
                "dream_report": str(dream_text)[:300],
            }
            for st in SUBTYPES:
                row_out[f"f1_{st}"] = None
            results.append(row_out)
            time.sleep(DELAY_SECONDS)
            continue

        metrics = evaluate_attribute(pred_activities, gt_activities)
        sub_metrics = {st: evaluate_by_subtype(pred_activities, gt_activities, st)
                       for st in SUBTYPES}

        has_gt = len(gt_activities) > 0
        status = "✓" if metrics["exact_match"] else "✗"
        marker = "" if has_gt else "  [no gt]"
        print(
            f"  {status}  {dream_id}"
            f"  pred={len(pred_activities)}  gt={len(gt_activities)}"
            f"  F1={metrics['f1']:.2f}{marker}"
        )

        row_out = {
            "dream_id":     dream_id,
            "predicted":    json.dumps(pred_activities, ensure_ascii=False),
            "ground_truth": json.dumps(gt_activities,  ensure_ascii=False),
            "exact_match":  metrics["exact_match"],
            "precision":    metrics["precision"],
            "recall":       metrics["recall"],
            "f1_attr":      metrics["f1"],
            "reasoning":    json.dumps(reasoning, ensure_ascii=False),
            "dream_report": str(dream_text)[:300],
        }
        for st in SUBTYPES:
            row_out[f"f1_{st}"] = sub_metrics[st]["f1"]
        results.append(row_out)

        time.sleep(DELAY_SECONDS)

    # ── Save results ──────────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = df_out[df_out["f1_attr"].notna()]
    n_valid = len(valid)

    print(f"\n{'─'*60}")
    print(f"Results saved → {OUTPUT_CSV}")
    print(f"Dreams evaluated : {n_valid}")

    if n_valid > 0:
        exact = valid["exact_match"].sum()
        print(f"Exact match      : {exact}/{n_valid}  ({exact/n_valid:.1%})")
        print(f"Mean F1 (attr)   : {valid['f1_attr'].mean():.3f}")
        print(f"Mean Precision   : {valid['precision'].mean():.3f}")
        print(f"Mean Recall      : {valid['recall'].mean():.3f}")
        print()
        print("Per sub-type F1 (attr):")
        for st in SUBTYPES:
            col = valid[f"f1_{st}"].dropna()
            if len(col) > 0:
                print(f"  {st}: {col.mean():.3f}  (n={len(col)})")


if __name__ == "__main__":
    main()
