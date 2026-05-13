#!/usr/bin/env python3
"""
settings.py — Hall/Van de Castle Settings Coder

Semi-automated classifier for the Settings category of the H/VdC coding system.
Codes the physical environment of each dream scene using 2-letter codes, then
evaluates against human-coded ground truth.

Usage:
    python settings.py                                        # 50 norms-f dreams (dev partition)
    python settings.py --n 20                                 # sample of 20 dreams
    python settings.py --all                                  # full dataset
    python settings.py --collection norms-f                   # one collection
    python settings.py --dream-id norms-f_0018               # single dream
    python settings.py --skip 50 --n 50                       # dreams 51–100 (validation partition)

Output:
    settings_results.csv — per-dream results with predicted and ground-truth
    settings code lists, plus F1 for each dream.

Notes on evaluation:
    Each settings code is decomposed into two attributes — location type
    (I/O/A/N) and familiarity (F/Q/U/D/G/S) — then scored via Counter
    intersection. This gives 50% partial credit when location is correct but
    familiarity is wrong (or vice versa), and 0% for a completely wrong code.

Methodology note:
    Settings are coded only in norms-f and norms-m. Every dream has at least
    one setting code. Development uses norms-f dreams 1–50; norms-m is
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
OUTPUT_CSV        = "settings_results.csv"
SAMPLE_SIZE       = 50
SAMPLE_COLLECTION = "norms-f"
DELAY_SECONDS     = 0.3

LOCATION_TYPES    = ["I", "O", "A", "N"]
FAMILIARITY_TYPES = ["F", "Q", "U", "D", "G"]

# ─────────────────────────────────────────────────────────────────────────────
# CODEBOOK
# ─────────────────────────────────────────────────────────────────────────────
CODEBOOK = """
Hall/Van de Castle (H/VdC) Settings Coding Rules
=================================================

## Overview

Code the physical environment of each dream scene using a 2-letter code:
[Location Type][Familiarity]. Assign one code per distinct location in the
dream. If the dream moves between different places, code each separately.

Every dream receives at least one settings code. Only use NS (No Setting) when
the dream has truly no physical environment whatsoever.

─────────────────────────────────────────────────────────────────────────────
## CODE STRUCTURE

Each code is two letters: [LOCATION][FAMILIARITY]

### Location Type (first letter)

**I — Indoor**
Inside any building, room, or enclosed structure.
  Examples: bedroom, classroom, laboratory, store, house, office, hallway,
  bathroom, dormitory, hospital, theater.
  NOTE: Cars and vehicles (parked or moving) = O (outdoor). A car on a road,
  whether the car is in motion or stationary, is coded as O because the relevant
  environment is the outdoor context. Do not code a car as I or A.

**O — Outdoor**
Outside in the open air: streets, roads, fields, bodies of water, parks,
wilderness, open landscapes, or any exterior environment.
  Examples: street, lake, field, forest, bridge, yard, open road, hillside,
  beach, canal, countryside.

**A — Ambiguous**
The setting exists but the text does not allow determining whether it is
indoors or outdoors. Use sparingly.
  Examples: a covered porch or open pavilion, a dream where ONLY characters
  and activities are described with NO physical environment whatsoever, a setting
  described only as "there" with no physical context, a camp or campground
  (could be cabin = indoor, or grounds = outdoor).
  KEY RULE: If no physical environment is described at all — only characters
  and activities (e.g., "a teacher and students talking about science") — code
  AQ (not IQ). Do NOT infer an indoor or outdoor setting from activity context
  alone. Social context (teacher + students) without ANY physical description
  = AQ, not IQ.
  TOWN/CITY/CAMPUS SETTINGS: A dream that describes being "in a town" or
  moving through campus abstractly (without specifying whether the scene is
  inside a building or outside) = A (ambiguous), not O (outdoor). Towns and
  campuses have both indoor and outdoor areas.
  AF (Ambiguous Familiar): When the dream takes place in a familiar context
  (the dreamer's own campus, home city, school area, a named place just
  recently visited) but the physical environment is ambiguous or not described,
  use AF. Examples: a dream set "on campus" moving between locations without
  specifying indoor/outdoor; a named camp the dreamer just returned from; an
  activity tied to the dreamer's familiar school routine without specific physical
  description. The dreamer is clearly in known territory even without a specific
  physical space described.
  AD (Ambiguous Distorted): A setting that is both ambiguous (can't tell
  indoor/outdoor) AND distorted (physically impossible or wrong). Example: "a
  small town but all the buildings were foreign to that area and it seemed to
  be in a different city" = AD.

**N — No Setting (NS only)**
The dream has NO physical environment at all — not even an implied or vague
location. Use NS only when the narrative could take place in absolutely any
place (or no place at all): pure abstract observation, a disembodied vision
(just seeing someone's face close up), purely conversational or conceptual
narrative with no spatial grounding.
  Examples:
  - "I had a spiritual rather than a visual dream"
  - Seeing only a person's face very close with no surrounding space
  - Overhearing people discuss topics with no scene or location
  NS vs AQ: If there is ANY spatial context — even vague — code AQ. NS is
  only for dreams with literally zero physical environment, zero sense of
  being somewhere.
  NS vs AF: If the dreamer implies they are in their own familiar territory
  (home area, campus, school) but no physical environment is described, code
  AF (not NS and not AQ). Familiar context without physical description = AF.

─────────────────────────────────────────────────────────────────────────────
### Familiarity (second letter)

**F — Familiar**
The dreamer uses a possessive pronoun for the place ("my bedroom," "my house,"
"our room," "my office," "our living room"), OR the place is explicitly named
as a specific recognized location.
  Examples: "my bedroom," "our house," "biology lab" (named lab), "Euclid
  Avenue" (named street the dreamer recognizes), "Quinn-Maah's" (named shop).
  DORMITORY RULE: When the dream is clearly set within the dreamer's own
  dormitory (established by phrases like "my dorm," "my dormitory," "girls
  from my dorm," "my ex-roommate," "girl living across the hall in my dorm"),
  ALL sub-spaces within that dormitory = F. A bathroom, hallway, or common room
  within the dreamer's own dorm = IF, not IQ.
  IMPORTANT — what does NOT make a place F:
  - Familiar PEOPLE or EVENTS in the setting: a named professor in a classroom
    does NOT make the classroom F; dormitory friends in a dining room do NOT
    make it F (unless clearly the dreamer's own dorm dining room).
  - Named EVENTS: the event being named is not the same as the VENUE being named.
  - General labels ("a classroom," "backstage") without possessives or context
    establishing them as the dreamer's own space.
  Key test: Is the PLACE ITSELF named or does the dreamer say "my/our [place]"?

**Q — Questionable**
The place exists but its familiarity is truly neutral — the text provides no
signal about whether the dreamer recognizes or does not recognize it.
  Examples: "a large room," "we were outside," "backstage at a performance,"
  "the dining room" (without possessive), "a classroom," "a building."
  Key test: No possessive, no name, no statement of familiarity OR unfamiliarity.

**U — Unfamiliar**
The dreamer uses explicit language indicating they do not recognize or know
the place. The signal must be present in the text — not just inferred from
the type of place.
  Explicit U signals:
  - "surroundings were unfamiliar," "I didn't know where we were," "it was
    not our house," "I'd never been there," "I don't recognize this place"
  - "an unidentified shop/building/room" — dreamer explicitly says unidentified
  - "a new home" / "new house" — explicitly new = not yet the dreamer's familiar
    space
  - "a strange town / strange street / strange place" — "strange" in reference
    to a location = dreamer does not recognize it → U
  - "secluded path," "secluded area," etc. — "secluded" as a description of a
    location = unfamiliar isolated place → U
  - Context that clearly implies the dreamer has NO prior relationship with the
    people OR the place: "I never went around with these girls" combined with
    the setting being their space.
  Generic place types do NOT by themselves make the code U:
  - A hotel room, doctor's office, hospital, classroom, or store that is not
    described as unfamiliar = Q (the generic type does not imply unfamiliarity).
  - Q is correct for any unnamed, non-possessive, non-explicitly-unfamiliar
    location regardless of what type of place it is.
  Distinction from Q: Q is neutral — no familiarity signal either way.
  U requires an explicit or near-explicit textual statement of non-recognition.

**D — Distorted**
The dreamer recognizes the PLACE TYPE as familiar (their room, their school,
etc.) but the setting is physically wrong, altered, or impossible — OR any
setting has physically impossible features (seeing through walls, rooms that
appear from nowhere, walls that vanish).
  Examples:
  - "my room, but it didn't look like mine" (recognizes type, wrong appearance)
  - "I could see right through the walls" in the living room (physical impossibility)
  - "a room that had not existed was left bare" (impossible architectural feature)
  - An outdoor setting with a wall that disappeared revealing an interior = OD
  IMPORTANT: The D code applies to the WHOLE setting — do not split a single
  distorted location into two codes. An outdoor area with a physically
  impossible feature = OD (one code, not OF + ID). A distorted ambiguous
  setting (e.g., a distorted town where you can't tell if you're inside or
  outside) = AD (one code).
  Key distinction from U: Distorted = the dreamer knows what place it should
  be but it is physically wrong. Unfamiliar = the dreamer does not recognize
  the place at all.
  Distorted outdoor examples:
  - A vessel/ship where the seating arrangement is physically impossible = OD
  - A town where the buildings are "foreign to that area" and it feels like
    it's in a different city = AD (ambiguous + distorted — the town setting is
    ambiguous between outdoor street and indoor buildings)
  IMPORTANT — "at home but it was not our house": This phrase = IU (Unfamiliar),
  NOT ID (Distorted). When the dreamer says "it was not our house," they are
  expressing non-recognition of the space — they do not claim it as their own
  familiar place at all. This is unfamiliar, not distorted.

**G — Geographic (outdoor only)**
A named real-world geographic location: a city, state, country, body of water,
or region. Only applies to outdoor settings (OG).
  Examples: "New York State," "Lake Ontario," "Rochester," "driving through
  France," "the Mississippi River."
  Note: Named indoor places (Quinn-Maah's, biology lab) = IF, not IG.

**S — (No Setting only)**
Used only in the code NS (No Setting). Not combined with any other location.

─────────────────────────────────────────────────────────────────────────────
## CODING RULES

### One code per distinct scene
Assign one setting code each time the dream moves to a clearly different
physical location. Code in the order scenes appear in the dream.

### Scene change threshold
Only code a new setting when the narrative explicitly relocates to a CLEARLY
DIFFERENT place. Err on the side of fewer codes.
  - Moving within the same building (hallway → room → bathroom) = NO new code.
  - Different rooms in the same house or building = NO new code.
  - Moving within the same campus, street block, or continuous outdoor area =
    NO new code.
  - Entering a completely different building on the same street = new code.
  - Moving from outdoors to a different indoor building = new code.

### Same location revisited
If the dream returns to a location already coded, do NOT add another code.
Code each distinct place once only.

### Setting transformations
If one setting transforms into a different setting in the same spot (e.g., a
lavatory that becomes a cafeteria, a room that changes appearance), this is
ONE code — not two.
  - D (Distorted) applies only when the dreamer RECOGNIZES THE TYPE as a
    specific familiar place but it looks physically wrong or impossible.
    Example: "my room, but it didn't look like mine" = ID.
    Example: "I could see right through the walls" in the living room = ID.
  - If two unrelated room-types transform into each other (lavatory → cafeteria)
    and neither is the dreamer's own recognized space, this is NOT D.
    Use IQ (or the appropriate familiarity code) for ONE setting — the one that
    best represents the primary or original location.
  Example: "I was in the lavatory. When I came out, it was no longer that but
  a cafeteria." → ONE code: IQ (not ID, not IQ + IQ).

### Do not add speculative codes
Only code settings where the DREAMER IS PHYSICALLY PRESENT in the scene.
  - A location mentioned in dialogue or reported speech (parents "were fishing
    on the lake," someone "was in Canada") but where the dreamer is NOT
    physically present = do NOT code it as a setting.
  - A location the dreamer thinks about, plans to visit, or hears about = do
    NOT code it.
  - A location the dreamer is clearly IN (standing, walking, sitting, acting
    in the scene) = code it.
Do not add an AQ code for the opening of a dream before a physical setting
is established — if the dream starts without a described setting and then
moves to one, code only the described setting.
Do not add a new code for an ambiguous transitional moment (e.g., a brief
"complete change of scene" with no physical description) — only code scenes
with an actual described physical environment.

### Entering an unfamiliar place = IU signal
When the dreamer is clearly entering a place they have no prior relationship
with — especially when guided there by strangers or through an introduction
("it was through them that I had been introduced to the people in charge") —
this context signals unfamiliarity: use U, not Q.

### Familiar people do NOT make the place familiar
The F code requires the PLACE ITSELF to be named or possessively owned —
not the people in it. Familiar companions, friends, or family members in
an unnamed outdoor or indoor setting do not make it F. Code Q if the place
has no name/possessive, regardless of who is present.
  Wrong: outdoor slope = OF because her boyfriend is there.
  Right: outdoor slope (unnamed) = OQ.

### Recurring campus events at known venues = IF
If the dreamer explicitly participates in a recurring campus event ("my part,"
"our turn," "since stunt night is approaching, I dreamed I was going") at a
campus venue, code the venue IF — the dreamer knows the place through repeated
participation, even if the specific building is not named.

### Familiarity is from the dreamer's perspective
Familiarity is determined by how the DREAMER experiences the place, not by
whether a reader would recognize it. A store named "Quinn-Maah's" = IF because
the dreamer is walking her familiar shopping street, even if the reader doesn't
know the store.

### G applies to named real-world outdoor geography only
City names, state/country names, lakes, rivers, and geographic regions = OG.
Named indoor locations (a specific store, a named school lab) = IF.
CRITICAL: "Seemed like [named place]" or "it reminded me of [place]" or
"which seemed like [named geographic location]" is NOT OG — use OF instead.
OG requires the dream to clearly and definitively take place IN the named
location. Resemblance or comparison ("which seemed like Forest Lawn Cemetery")
= OF (outdoor familiar through association), not OG.
Named landmarks with interior spaces (e.g., Washington Monument, a museum):
if the dream takes place INSIDE the named structure, code IF for the interior
and OF (or OG if definitively named outdoor area) for the exterior. Do not
collapse indoor and outdoor portions of a named landmark into a single OG.

### F for third-party possessives and shared living spaces
Use F when the context makes clear the dreamer knows and has visited the place,
even if it is not the dreamer's own:
  - "his frat house," "her apartment," "GS's living room" when the dreamer
    is a regular visitor = IF (context implies familiarity).
  - Dreamer's shared room (with a roommate) = IF even without "my room" — the
    shared living space is the dreamer's own familiar space.
  - A campus building the dreamer names and attends events at regularly = IF.
When in doubt about third-party spaces, use Q — unless context strongly
implies the dreamer is a regular visitor.

─────────────────────────────────────────────────────────────────────────────
## DO NOT CODE

  – Multiple codes for sub-locations within one scene: moving from one room
    to another in the same house is NOT a new scene. Only code when the dream
    clearly transitions to a different place.

  – Re-coding a revisited setting: if the dreamer leaves a location and returns
    to it, do not add a second code for the same place.

  – NS when any physical environment is described: use NS only when there is
    truly no physical setting. A vague or minimal setting still gets a code.

  – IG (indoor geographic): named indoor locations are IF, not IG. G applies
    only to outdoor named geographic locations.

  – AQ for the beginning of a dream before a setting is described: if the
    dream starts with body sensations, thoughts, or actions with no physical
    environment, do NOT add AQ. Only code the physically described settings
    that follow. A brief undescribed opening is not a separate code.

  – AQ or any code for a transitional "complete change of scene" with no
    physical description: if the dream text says "then there was a complete
    change" or similar without describing a new physical environment, do not
    add any code for that transition moment.

  – Camp/outdoor-event settings as O: a named camp or campground (e.g., "Camp
    Wise") is A (Ambiguous), not O (Outdoor), because camps include both indoor
    cabins and outdoor grounds. If the dream does not specify whether the scene
    is in a cabin (indoor) or at the grounds (outdoor), use A.

─────────────────────────────────────────────────────────────────────────────
## OUTPUT FORMAT

Respond with ONLY valid JSON — no prose before or after — in this exact shape:

{
  "settings": ["IF", "OQ"],
  "reasoning": {
    "IF (biology lab)": "Indoor familiar — named, specific location the dreamer recognizes.",
    "OQ (outdoors near railroad)": "Outdoor; location not named or described as familiar/unfamiliar."
  }
}

Field rules:
  - "settings": ordered list of 2-letter codes, one per distinct scene.
  - List codes in the order scenes appear in the dream.
  - Valid codes: IF, IQ, IU, ID, OF, OQ, OU, OD, OG, AF, AQ, AU, AD, NS.
  - NS is used alone when there is no physical environment.
  - Include reasoning for each coded setting and for any judgment call.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES  (drawn from norms-f dev partition: dreams 1–50)
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT = [
    # Example 1: Single familiar indoor (named location)
    {
        "dream": (
            "I was in biology lab and had a bag of cookies and a bag of after dinner "
            "mints. I put them on the desk and said everyone was welcome to help "
            "themselves. When I went to get some, they were all gone, so I went around "
            "to each person in the lab and made them put some back. I was perplexed "
            "because they had taken advantage of my generosity and disgusted because "
            "some of them had made such pigs of themselves."
        ),
        "settings": ["IF"],
        "reasoning": {
            "IF (biology lab)": (
                "The entire dream takes place in 'biology lab' — an indoor setting. "
                "The place is named and clearly a familiar academic location the "
                "dreamer attends. Indoor + Familiar = IF."
            ),
        },
    },
    # Example 2: Single outdoor unfamiliar (lake)
    {
        "dream": (
            "I was in water like a lake up to my waist and my sister (16) was with me. "
            "We were striving desperately to get away from a large Negro man, but the "
            "water was holding us back. I feared that he was going to rape us, but then "
            "I realized he would not."
        ),
        "settings": ["OU"],
        "reasoning": {
            "OU (lake/water)": (
                "The setting is a body of water described as 'like a lake' — outdoor. "
                "No name is given and no familiarity is expressed; 'like a lake' "
                "suggests it is not a recognized specific place. Outdoor + Unfamiliar = OU."
            ),
        },
    },
    # Example 3: Single indoor distorted (recognizable type, physically wrong)
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
        "settings": ["ID"],
        "reasoning": {
            "ID (my room / the stairs — same building)": (
                "The dream takes place inside the dreamer's room and building. The "
                "dreamer recognizes it as 'my room' (familiar type) but it 'did not "
                "look like mine — rather like Doris's room.' This is the defining "
                "feature of Distorted: the dreamer expects a familiar place but it is "
                "physically wrong. Indoor + Distorted = ID. Running down the stairs "
                "is within the same building — not a new scene."
            ),
        },
    },
    # Example 4: Single outdoor geographic (named real-world locations)
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
        "settings": ["OG"],
        "reasoning": {
            "OG (western New York State / Rochester / Lake Ontario / Erie)": (
                "The entire dream is an outdoor drive through named, real-world geographic "
                "locations: New York State, Rochester, Lake Ontario, Freeport, Lake Erie. "
                "Outdoor + Geographic = OG. All scenes are part of the same continuous "
                "drive through this region — one code."
            ),
        },
    },
    # Example 5: Three settings — scene changes across one dream
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
        "settings": ["OF", "IF", "IU"],
        "reasoning": {
            "OF (Euclid Avenue)": (
                "The dream opens on Euclid Avenue — a named outdoor street the dreamer "
                "is walking along. Outdoor + Familiar (named street the dreamer knows) = OF."
            ),
            "IF (Quinn-Maah's, Milgrims — named shops)": (
                "The dreamer enters the named specialty shops. These are indoor settings "
                "the dreamer recognizes by name (familiar shopping street). "
                "Indoor + Familiar = IF. Multiple named shops on the same street are "
                "the same scene type — one code."
            ),
            "IU (unidentified shop at the end)": (
                "The final shop is explicitly 'unidentified' — the dreamer enters a shop "
                "with no name. Indoor + Unfamiliar = IU. This is a distinct scene: a "
                "new, different shop that the dreamer does not recognize."
            ),
        },
    },
    # Example 6: No setting — abstract, non-visual dream
    {
        "dream": (
            "The only visible person was a man who had freakish limbs. At the end of "
            "each arm and leg there was either a stump or only one or two fingers or "
            "toes. I had the feeling he had acquired one of them only recently. I seemed "
            "to know that this man had a loving wife who would not leave him because of "
            "his horrible condition. I also had the feeling that both my fiance and I "
            "knew him and we seemed to be looking at him (but we were not actually in "
            "the dream). Later I had a spiritual rather than a visual dream in which I "
            "seemed to be training children."
        ),
        "settings": ["NS"],
        "reasoning": {
            "NS (no physical setting)": (
                "The dreamer explicitly notes they were 'not actually in the dream' and "
                "later describes it as 'a spiritual rather than a visual dream.' There "
                "is no physical environment — no room, no street, no outdoor space. "
                "The dream has no location whatsoever. No Setting = NS."
            ),
        },
    },
    # Example 7: Three settings including AF — ambiguous familiar
    {
        "dream": (
            "This dream was sort of mixed up. At first the large room we were in was "
            "all dark. A mock wedding was going on. Later it turned out to be a weekend "
            "out at Camp Wise (I had just returned from there). Later it seemed as if I "
            "were in a dorm. I spoke with several counselors. On the back of my mind I "
            "kept thinking about my purse."
        ),
        "settings": ["IQ", "AF", "IQ"],
        "reasoning": {
            "IQ (large dark room)": (
                "The dream opens in 'a large room' where a mock wedding is occurring. "
                "No possessive, no name, no familiarity signal. Indoor + Questionable = IQ."
            ),
            "AF (Camp Wise)": (
                "The dream transitions to Camp Wise — a named camp the dreamer just "
                "returned from (very familiar). However, camps include both indoor "
                "facilities and outdoor grounds, so the specific physical environment "
                "(cabin? outdoor field?) is ambiguous. Familiar because named and just "
                "visited, but Ambiguous because indoor vs outdoor cannot be determined. "
                "Ambiguous + Familiar = AF."
            ),
            "IQ (dorm — 'a dorm')": (
                "'Later it seemed as if I were in a dorm.' The dreamer says 'a dorm,' "
                "not 'my dorm' — no possessive, no specific name. An unnamed dorm is "
                "indoor but not clearly the dreamer's own familiar dorm. "
                "Indoor + Questionable = IQ."
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

    set_df = codings[codings["coding_type"] == "set"].copy()

    gt_by_dream = {}
    for dream_id, group in set_df.groupby("dream_id"):
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
        output = {"settings": ex["settings"], "reasoning": ex["reasoning"]}
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
    """Call Claude and return (settings_list, reasoning_dict)."""
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
                    "Code all settings in this dream report:\n\n"
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
    settings = [str(s).strip().upper() for s in parsed.get("settings", [])]
    return settings, parsed.get("reasoning", {})


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def decompose_code(code):
    """Decompose a 2-letter settings code into (slot, value) tuples."""
    code = str(code).strip().upper()
    if len(code) == 2:
        return [("location", code[0]), ("familiarity", code[1])]
    return [("code", code)]


def evaluate(pred_codes, gt_codes):
    """Attribute-level F1 over a list of settings codes."""
    pred = Counter(item for c in pred_codes for item in decompose_code(c))
    true = Counter(item for c in gt_codes  for item in decompose_code(c))
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
    parser = argparse.ArgumentParser(description="H/VdC Settings Coder")
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

        gt = gt_by_dream.get(dream_id, [])

        try:
            pred, reasoning = call_claude(client, str(dream_text), system_content)
        except Exception as e:
            print(f"  ERROR {dream_id}: {e}")
            results.append({
                "dream_id":      dream_id,
                "pred_settings": "[]",
                "gt_settings":   json.dumps(gt),
                "f1":            None,
                "precision":     None,
                "recall":        None,
                "reasoning":     f"ERROR: {e}",
                "dream_report":  str(dream_text)[:300],
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
            "dream_id":      dream_id,
            "pred_settings": json.dumps(pred, ensure_ascii=False),
            "gt_settings":   json.dumps(gt,   ensure_ascii=False),
            "f1":            m["f1"],
            "precision":     m["precision"],
            "recall":        m["recall"],
            "reasoning":     json.dumps(reasoning, ensure_ascii=False),
            "dream_report":  str(dream_text)[:300],
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

        # Per-location-type breakdown
        print(f"\nPer-location-type F1 (on dreams containing that type in GT):")
        location_labels = {"I": "Indoor", "O": "Outdoor", "A": "Ambiguous", "N": "No Setting"}
        for loc, label in location_labels.items():
            mask = valid["gt_settings"].apply(
                lambda s: any(c.startswith(loc) for c in json.loads(s))
            )
            subset = valid[mask]
            if len(subset) > 0:
                print(f"  {label:12s}: {subset['f1'].mean():.3f}  (n={len(subset)})")

        # Per-familiarity-type breakdown
        print(f"\nPer-familiarity-type F1 (on dreams containing that type in GT):")
        fam_labels = {"F": "Familiar", "Q": "Questionable", "U": "Unfamiliar",
                      "D": "Distorted", "G": "Geographic"}
        for fam, label in fam_labels.items():
            mask = valid["gt_settings"].apply(
                lambda s: any(c.endswith(fam) for c in json.loads(s))
            )
            subset = valid[mask]
            if len(subset) > 0:
                print(f"  {label:13s}: {subset['f1'].mean():.3f}  (n={len(subset)})")


if __name__ == "__main__":
    main()
