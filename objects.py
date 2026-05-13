#!/usr/bin/env python3
"""
objects.py — Hall/Van de Castle Objects Coder

Semi-automated classifier for the Objects category of the H/VdC coding system.
Codes all physical objects appearing in a dream using 2-letter category codes,
then evaluates against human-coded ground truth.

Usage:
    python objects.py                                        # 50 norms-f dreams (dev partition)
    python objects.py --n 20                                 # sample of 20 dreams
    python objects.py --all                                  # full dataset
    python objects.py --collection norms-f                   # one collection
    python objects.py --dream-id norms-f_0029               # single dream
    python objects.py --skip 50 --n 50                       # dreams 51–100 (validation partition)

Output:
    objects_results.csv — per-dream results with predicted and ground-truth
    object code lists, plus F1 for each dream.

Notes on evaluation:
    Codes are atomic — each code is matched directly (no decomposition).
    A Counter-based intersection gives credit for each correct code instance,
    including multiple instances of the same code (e.g., FO×3 vs FO×2 gives
    partial credit). Codes that appear multiple times in GT must appear that
    many times in the prediction to get full credit.

Methodology note:
    Objects are coded only in norms-f and norms-m. Development uses norms-f
    dreams 1–50; norms-m is reserved as the untouched final test set.
    IMPORTANT: The Nature code "NA" must not be silently converted to NaN —
    this file uses keep_default_na=False in all pd.read_csv() calls.
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
OUTPUT_CSV        = "objects_results.csv"
SAMPLE_SIZE       = 50
SAMPLE_COLLECTION = "norms-f"
DELAY_SECONDS     = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# CODEBOOK
# ─────────────────────────────────────────────────────────────────────────────
CODEBOOK = """
Hall/Van de Castle (H/VdC) Objects Coding Rules
================================================

## Overview

Code every distinct physical object that appears concretely in the dream scene
using the appropriate 2-letter category code. Codes can (and often do) repeat
within a single dream — each distinct object INSTANCE gets its own code.

A desk and a chair = HH + HH (not just HH once).
Two suitcases = TR + TR.
Three food/drink items = FO + FO + FO.

Average: about 5 object codes per dream. Most dreams have 3–10 objects.

─────────────────────────────────────────────────────────────────────────────
## THE 25 OBJECT CODES

### Architecture (A_)

**AR — Residential**
Private dwellings and their interior spaces: houses, apartments, hotels,
dormitories, bedrooms, living rooms, hallways, basements, bathrooms, kitchens.
  Examples: a house, an apartment, a hotel room, a dorm room, a bedroom,
  "went into the living room," a basement.
  NOTE: Code the building or room itself as an object only when it appears as
  a distinct physical thing encountered in the scene. If two different rooms
  are clearly separate locations, each gets its own AR code.
  NOTE: A vague or unfamiliar room where vocational activity is occurring
  (a music lesson, a medical demonstration, a work task) = AV, not AR.
  AR is for clearly residential contexts.

**AV — Vocational**
Buildings or rooms devoted mainly to business, manufacturing, employment,
or general education: stores, offices, factories, classrooms, laboratories,
banks, libraries.
  Examples: a biology lab, a classroom, a store, a shop, a factory, an office,
  "the bank," a library.
  KEY DISTINCTION from AI: Use AV for general educational or business contexts
  (classroom at school, office, store). Use AI when the building is specifically
  a medical, governmental, judicial, religious, or custodial institution.

**AE — Entertainment**
Buildings used for recreation and pleasure: restaurants, theaters, museums,
stadiums, bowling alleys, bars/clubs, auditoriums, recreation halls.
  Examples: a restaurant, a large dining room (as a public venue), a theater,
  a museum, a stadium, a backstage/performance area.
  NOTE: Backstage preparation areas that are clearly within a theater or
  performance venue = AE. Code each clearly distinct area that the dreamer
  inhabits as a separate AE.

**AI — Institutional**
Buildings maintained for collective societal action: hospitals, jails,
courthouses, government buildings, churches, prisons, medical facilities.
  Examples: a hospital ward, a church, a courthouse, a prison, a police
  station, a government building, an anatomy lecture hall or autopsy theater
  (medical/institutional context).
  KEY DISTINCTION from AV: An anatomy demonstration or autopsy at a medical
  facility = AI. A regular classroom at school = AV.

**AD — Architectural Details**
Parts of a building not usually regarded as separate rooms: doors, windows,
walls, ceilings, floors, roofs, chimneys, staircases, fireplaces.
  Examples: a door, a window, the wall, a staircase, stairs, a fireplace,
  the ceiling, a chimney.
  NOTE: Do NOT code vehicle doors, windows, or parts as AD. A car door or
  car window is part of the vehicle (TR), not a separate architectural detail.

**AM — Architectural Miscellaneous**
Building parts or structures not fitting other architecture categories:
towers, dams, fountains, monuments, fences, gates, construction scaffolding,
corridors, passageways, hallways, and connecting spaces within a building.
  Examples: a tower, a dam, a fountain, a monument, a fence, steel girders
  (construction), a winding corridor, a passageway through a building, a
  waiting room used as a hallway space, a tunnel.
  KEY DISTINCTION from AR: A corridor, passageway, hallway, or connecting
  space that is not a distinct residential/vocational/entertainment room
  = AM (not AR). A defined residential room (bedroom, living room) = AR.
  A defined waiting/lobby area that functions as a passageway = AM.

**AB — (rare) Structural/Building Elements**
Structural components of a building that don't fit other A codes.
Very rarely used. When in doubt, use AM.

─────────────────────────────────────────────────────────────────────────────
### Body (B_)

Code body parts ONLY when the dream explicitly describes them as objects of
attention or perception — not merely because a character is present.
"He was standing there" does NOT warrant any B code. "He had freakish limbs"
or "she had red hair" or "I saw her hand" DOES warrant a B code.

**BH — Head**
Visible parts of the head region: face, hair, eyes, nose, mouth, ears, teeth.
  Examples: her hair, his face, her eyes, "a beautiful smile," "long red hair."

**BE — Extremities**
Arms, legs, hands, feet, fingers, toes, and similar appendages.
  Examples: his arm, her hand, their feet, "a finger," "put his arm around me."

**BT — Torso**
Visible trunk areas: shoulders, chest, abdomen, back, hips, neck.
  Examples: his shoulder, her back, the chest wound, "broad shoulders."

**BA — Anatomy**
Internal body parts, organs, bones, tissues, and bodily substances/secretions:
heart, lungs, skull, bones, blood, saliva, mucus, a growth, a tumor, cancer.
  Examples: "could see his bones," a skull, organs on display, "2/10 blood,"
  "she had cancer," "a growth was detected," blood spatter.
  IMPORTANT: Code BA whenever any body substance, internal organ, or anatomical
  tissue is explicitly named — even in a single word or brief phrase. "Cancer,"
  "blood," "a growth" = BA. Do not skip BA because the mention seems incidental.
  However, when "cancer" and "growth" (or similar terms) describe the same
  condition in the same passage, code BA only once — they are the same object.

**BS — Sex**
Body parts and organs related to reproduction and excretion.
  Examples: sexual body parts, reproductive organs, excretory references.

─────────────────────────────────────────────────────────────────────────────
### Other Categories

**CL — Clothing**
Garments, accessories, jewelry, footwear, and worn items.
  Examples: a dress, shoes, a slip, earrings, a hat, "her housecoat," jewelry.

**CM — Communication**
Media, devices, and materials for communication: books, letters, phones,
newspapers, signs, computers, radios, televisions, intercoms.
  Examples: a book, a letter, a telephone, a radio, a newspaper, a sign.

**FO — Food**
All forms of food or drink, whether raw, prepared, or packaged.
  Examples: cookies, beer, a coke, a meal, mints, food on a plate, a snack.
  NOTE: Containers for food (bags, bottles) may be coded HH separately if
  the container itself is a distinct object of attention.

**HH — Household**
Furniture, appliances, utensils, and supplies commonly found in homes or
institutions: tables, chairs, desks, beds, lamps, appliances, bags/containers.
  Examples: a desk, a table, a dresser, a bed, a lamp, a drawer, furniture,
  a counter, an ironing board, bags/containers, curtains, carpeting.
  MUSICAL INSTRUMENTS AS HH: A piano, organ, or other instrument described as
  a household possession or home furnishing = HH (not IR). A piano "which is
  mine" or described as part of the home's furnishings = HH. Use IR only when
  an instrument is clearly being actively played for recreation in the dream.

**IR — Recreation Implements**
Sporting goods, games, toys, fishing equipment, and musical instruments
actively being used for recreation (not merely as home furnishings).
  Examples: a fishing rod, a ball, a toy, a game, a tennis racket, a bat.
  Distinction from HH: A piano as a household possession = HH. A guitar being
  strummed at a party = IR.

**IT — Tools**
Tools, machinery, apparatus, and machinery parts used in work, vocational,
or medical activities.
  Examples: a hammer, a wrench, machinery parts, an iron (for ironing clothes),
  scissors used as a tool, a drill, a medical apparatus, a piece of equipment
  used for a demonstration or procedure, a mouthpiece for a device.
  NOTE: When a dream mentions "apparatus," "equipment," "machine," or "device"
  in a work, medical, or scientific context, code it IT (not MS).
  Distinction: IT = tools/equipment used as tools; IW = weapons.

**IW — Weapons**
Objects whose primary purpose is harm: guns, swords, knives (as weapons),
bombs, tanks, grenades.
  Examples: a gun, a revolver, a sword, a bomb, a knife used as weapon.

**MO — Money**
Currency, coins, checks, financial records, credit cards, bank books.
  Examples: money, coins, a check, cash, "a lot of money," a bank book,
  a bank statement, a ledger.
  Distinction from CM: A bank book or financial document = MO (not CM).
  CM = books, letters, phones, newspapers, signs.

**MS — Miscellaneous Objects**
Objects that don't fit any other category.
  Examples: an unusual or unclassifiable object; use sparingly.

**NA — Nature**
Plants, terrain features, natural water bodies, weather phenomena, heavenly
bodies, minerals, and animals not otherwise coded.
  Examples: a tree, a lake, a river, a hill, a marsh, the sun, rocks, a field,
  an animal, flowers, a cloud, snow.
  IMPORTANT: "NA" is a valid 2-letter code — always include it as the string
  "NA" in your JSON output when you code a nature object.

**RG — Regions**
Bounded land areas, named or unnamed: cities, towns, states, countries,
parks, yards, farms, parking lots, named geographic areas — but ONLY when the
dreamer is physically present in that region in the dream scene.
  Examples: New York State (region the dreamer drives through), Rochester
  (city the dreamer is in), a park the dreamer visits, "the neighborhood."
  Distinction: RG = a bounded area or territory; NA = a specific natural object
  within a setting (a tree, a lake). A city = RG; a lake = NA.
  DO NOT CODE RG for: cities merely mentioned in narrative context without
  the dreamer physically being there ("he works in Chicago," "she came from
  Cleveland," "we might go to Europe"), background information about where
  someone lives, or travel plans that don't occur in the dream scene.

**ST — Streets**
Roadways and their infrastructure: streets, roads, highways, avenues, paths,
bridges, train tracks, sidewalks, driveways.
  Examples: Euclid Avenue (the street itself), a highway, a bridge, a sidewalk,
  a railroad track, a driveway.
  NOTE: A named street is ST; the area around it may be RG.

**TR — Travel**
Conveyances and travel-related objects: cars, planes, boats, bicycles,
trains, luggage, suitcases.
  Examples: a car, a plane, a boat, a suitcase, a bicycle, a train.
  Distinction: The vehicle = TR; the road it travels on = ST.

─────────────────────────────────────────────────────────────────────────────
## CODING RULES

### What to code
- Physical objects explicitly present in the dream scene
- Each distinct instance of the same type of object
- Objects mentioned as background (a car parked outside counts as TR)
- Buildings and rooms when they are explicitly entered or described as distinct
  physical spaces (AR, AV, AE, AI) — code each separately described room or
  area that the dreamer clearly inhabits

### What NOT to code
- Abstract concepts, emotions, plans
- Characters themselves (body parts only when explicitly described)
- Buildings or rooms that are only implied by context rather than explicitly
  described. "In to see a doctor" implies a doctor's office but does not
  explicitly describe a room — do not add AI just from this phrase. However,
  "entering a hospital ward" explicitly describes the space → code AI.
- Objects merely imagined, recalled, or discussed but NOT physically present
  in the dream scene
- Vehicle parts (car doors, windows, seats) as separate AD codes — they are
  part of the TR object

### Architecture objects: explicit vs implied
Only code an architecture object (AR, AV, AE, AI, AD, AM) when the building
or room is explicitly described as a physical space the dreamer enters or
encounters. When a dream implies a building type through activity context
("went to the doctor," "at work"), but never explicitly names or describes
the room, do not code an architecture object — look instead for explicit
physical objects in the scene (furniture, equipment, body anatomy, etc.).

### Dreams with no physical objects → output []
Some dreams focus entirely on interpersonal events, emotions, character
states, or abstract narrative with no distinct physical objects described.
For these dreams, output an empty "objects" list. DO NOT code objects just
because they are implied by context:
  - A friend visiting = don't code their workplace as AV
  - A birthday party = don't code the house as AR or the gifts as MS
  - A city mentioned as where someone lives or comes from = don't code as RG
  - A person's changed appearance (weight loss, haircut) = don't code as BT/BH
    UNLESS the body feature is explicitly described as a distinct object of
    description in the dream
If the only "objects" you can identify are inferred from context rather than
explicitly described, output [].

### Repeat the code for each instance
If the dream describes two suitcases, code TR twice. If a dream has three
food items, code FO three times. Do not collapse multiple instances of the
same object type into a single code.

### Threshold for body codes
Do NOT code body parts merely because characters are described as present.
Code B_ only when a body part is explicitly singled out as an object of
description or attention in the dream narrative.

─────────────────────────────────────────────────────────────────────────────
## OUTPUT FORMAT

Respond with ONLY valid JSON — no prose before or after — in this exact shape:

{
  "objects": ["AR", "HH", "FO", "FO", "TR"],
  "reasoning": {
    "AR (the dorm room)": "Residential building/room — entered as a physical space.",
    "HH (the desk)": "Household furniture.",
    "FO × 2 (cookies, mints)": "Two distinct food items.",
    "TR (the suitcase)": "Travel/luggage object.",
    "NOT CODED — 'the hallway'": "Same AR space as the room, not a new object."
  }
}

Field rules:
  - "objects": flat list of 2-letter codes, one per distinct object instance.
  - Codes can repeat (FO, FO, FO for three food items).
  - Valid codes: AB, AD, AE, AI, AM, AR, AV, BA, BE, BH, BS, BT, CL, CM,
    FO, HH, IR, IT, IW, MO, MS, NA, RG, ST, TR.
  - "NA" must be written as the string "NA" — it is a valid code for nature objects.
  - Include reasoning for each coded object and any notable judgment call.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES  (drawn from norms-f dev partition: dreams 1–50)
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT = [
    # Example 1: Nature objects, streets, regions — shows repeated NA and RG
    {
        "dream": (
            "My mother and I were driving through western New York State. We went through "
            "Rochester and could see lake Ontario in the distance. We then arrived in "
            "Freeport on our way to Erie. Somehow we had the idea that this was where the "
            "two lakes joined. We drove beside Lake Ontario through the little town and "
            "beside a very marshy section. A little ahead of us was a hill and we supposed "
            "that Lake Erie was on the other side. We saw a bridge going over the point "
            "where we entered the town. Just as we were about to turn around the hill, my "
            "alarm rang and I was very angry. I did want to see how the two lakes joined."
        ),
        "objects": ["NA", "NA", "NA", "ST", "RG", "RG", "RG", "RG"],
        "reasoning": {
            "NA (Lake Ontario)": "Natural water body — nature object.",
            "NA (Lake Erie)": "Natural water body — nature object.",
            "NA (marshy section)": "Natural terrain feature — nature object.",
            "ST (the road/bridge)": "Roadway/bridge infrastructure they are driving on and see.",
            "RG (New York State)": "Named geographic region.",
            "RG (Rochester)": "Named city — bounded land area.",
            "RG (Freeport)": "Named town — bounded land area.",
            "RG (Erie)": "Named city/destination — bounded land area.",
            "NOT CODED — 'the hill'": "A hill is a natural terrain feature = NA? No — the hill is vague terrain context, not a distinct object of attention. The lakes and marsh are explicitly named and attended to; the hill is briefly mentioned in passing. Judgment call: not coded.",
        },
    },
    # Example 2: Vocational building, household, food — short, clear dream
    {
        "dream": (
            "I was in biology lab and had a bag of cookies and a bag of after dinner "
            "mints. I put them on the desk and said everyone was welcome to help "
            "themselves. When I went to get some, they were all gone, so I went around "
            "to each person in the lab and made them put some back. I was perplexed "
            "because they had taken advantage of my generosity and disgusted because "
            "some of them had made such pigs of themselves."
        ),
        "objects": ["AV", "HH", "HH", "HH", "FO", "FO"],
        "reasoning": {
            "AV (biology lab)": "A classroom/laboratory — vocational/educational building.",
            "HH (bag of cookies — container)": "Bag as a household/container object.",
            "HH (bag of after dinner mints — container)": "Second distinct bag — container object.",
            "HH (the desk)": "Furniture — household object.",
            "FO (cookies)": "Food item.",
            "FO (after dinner mints)": "Second distinct food item.",
        },
    },
    # Example 3: Residential rooms, household, travel objects — variety
    {
        "dream": (
            "I dreamed that Norma (18), a girl who resides in the same dormitory as I, "
            "and I decided to change rooms. I was going to take the room I had last year "
            "and she was going to take my present room. I packed some of my things in a "
            "suitcase and went down to my old room. However, the present occupant of the "
            "room hadn't removed her things from the room. Since I had no place to put my "
            "things, I took my suitcase and went back to my present room. There I saw "
            "Norma had already moved in. She had changed some of the furniture around and "
            "was standing by the window ironing. I told her I couldn't get back my old "
            "room and she only laughed. I started to take her clothes out of the drawer "
            "and to put them in her suitcase. She had a lot of money lying on top of the "
            "dresser, which I also removed and put in the suitcase."
        ),
        "objects": ["AR", "AR", "HH", "AD", "CL", "HH", "HH", "MO", "TR", "TR"],
        "reasoning": {
            "AR (the dreamer's present room)": "A dorm room — residential space, coded as AR.",
            "AR (the old room)": "A second distinct dorm room — separate residential space.",
            "HH (furniture Norma rearranged)": "Household furniture.",
            "AD (the window)": "Architectural detail of the room.",
            "CL (Norma's clothes)": "Clothing items.",
            "HH (the drawer)": "Household furniture/fixture.",
            "HH (the dresser)": "Household furniture.",
            "MO (money on the dresser)": "Money — explicitly described.",
            "TR (the dreamer's suitcase)": "Luggage/travel object.",
            "TR (Norma's suitcase)": "Second distinct suitcase — luggage/travel object.",
            "NOT CODED — 'the dormitory'": "The dormitory building as a whole is the setting context, not a separately described object here. The rooms (AR × 2) are what appear as distinct physical spaces.",
        },
    },
    # Example 4: Entertainment venue, money, region, food — wedding/dining room
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
        "objects": ["AE", "MO", "RG", "FO", "FO", "FO"],
        "reasoning": {
            "AE (large dining room — restaurant/hall)": "A large dining room where the couple sits for a meal; coded as entertainment/dining venue.",
            "MO (money — he had very little money)": "Money explicitly mentioned.",
            "RG (Europe — the honeymoon destination discussed)": "Named geographic region mentioned as a destination.",
            "FO (beer)": "Food/drink item.",
            "FO (coke)": "Food/drink item.",
            "FO (double shot of alcohol)": "Food/drink item ordered.",
            "NOT CODED — 'home' at the start": "Home is the general context for the start of the dream but is not described as a distinct physical object encountered.",
        },
    },
    # Example 5: No physical objects — interpersonal dream → output []
    {
        "dream": (
            "I dreamed of a very good friend of mine. He is 32, works at Marshall Field "
            "and Co. in Chicago. He is very overweight, but with such a personality that "
            "everyone likes him. In the dream he came here to Cleveland to see me. When "
            "he arrived, I found he had lost a lot of weight, really looked swell. He had "
            "done it to surprise me and I was very happy and proud."
        ),
        "objects": [],
        "reasoning": {
            "NOT CODED — Marshall Field and Co.": (
                "The friend's workplace in Chicago is background information — he does "
                "not visit the store in the dream and the dreamer is not physically there."
            ),
            "NOT CODED — Chicago / Cleveland": (
                "City names mentioned as narrative context (where he is from, where the "
                "dreamer is). The dreamer is not physically traveling through or acting "
                "within these cities as distinct dream scenes. Do not code as RG."
            ),
            "NOT CODED — weight loss": (
                "The friend's changed appearance is noted but no body part is described "
                "as a distinct object of attention. 'Lost a lot of weight' is an "
                "interpersonal observation, not a description of a body part."
            ),
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    # keep_default_na=False is REQUIRED: without it, pandas converts the
    # Nature code "NA" to NaN, silently erasing all nature object codings.
    dreams  = pd.read_csv(INPUT_CSV,   keep_default_na=False)
    codings = pd.read_csv(CODINGS_CSV, keep_default_na=False)

    obj_df = codings[codings["coding_type"] == "obj"].copy()

    gt_by_dream = {}
    for dream_id, group in obj_df.groupby("dream_id"):
        codes = []
        for _, row in group.iterrows():
            code = str(row.get("code", "")).strip().upper()
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
        output = {"objects": ex["objects"], "reasoning": ex["reasoning"]}
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
def call_claude(client, dream_text, system_content):
    """Call Claude and return (objects_list, reasoning_dict)."""
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
                    "Code all objects in this dream report:\n\n"
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

    VALID_CODES = {
        "AB", "AD", "AE", "AI", "AM", "AR", "AV",
        "BA", "BE", "BH", "BS", "BT",
        "CL", "CM", "FO", "HH", "IR", "IT", "IW",
        "MO", "MS", "NA", "RG", "ST", "TR",
    }
    raw_objects = parsed.get("objects", [])
    if not isinstance(raw_objects, list):
        raw_objects = []
    objects = [
        c for o in raw_objects
        if (c := str(o).strip().upper()) in VALID_CODES
    ]
    return objects, parsed.get("reasoning", {})


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(pred_codes, gt_codes):
    """Counter-based F1 over a list of object codes (no decomposition)."""
    pred = Counter(str(c).strip().upper() for c in pred_codes)
    true = Counter(str(c).strip().upper() for c in gt_codes)
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
    parser = argparse.ArgumentParser(description="H/VdC Objects Coder")
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
                "dream_id":     dream_id,
                "pred_objects": "[]",
                "gt_objects":   json.dumps(gt),
                "f1":           None,
                "precision":    None,
                "recall":       None,
                "reasoning":    f"ERROR: {e}",
                "dream_report": str(dream_text)[:300],
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
            "dream_id":     dream_id,
            "pred_objects": json.dumps(pred, ensure_ascii=False),
            "gt_objects":   json.dumps(gt,   ensure_ascii=False),
            "f1":           m["f1"],
            "precision":    m["precision"],
            "recall":       m["recall"],
            "reasoning":    json.dumps(reasoning, ensure_ascii=False),
            "dream_report": str(dream_text)[:300],
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

        # Per-code-category F1 on dreams that contain that code in GT
        print(f"\nPer-category F1 (on dreams containing that code in GT, n≥3):")
        all_codes = sorted({
            c
            for row in valid["gt_objects"]
            for c in json.loads(row)
        })
        for code in all_codes:
            mask = valid["gt_objects"].apply(
                lambda s: code in json.loads(s)
            )
            subset = valid[mask]
            if len(subset) >= 3:
                print(f"  {code:3s} : {subset['f1'].mean():.3f}  (n={len(subset)})")


if __name__ == "__main__":
    main()
