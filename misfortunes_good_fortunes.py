#!/usr/bin/env python3
"""
misfortunes_good_fortunes.py — Hall/Van de Castle Misfortunes & Good Fortunes Coder

Semi-automated classifier for the Misfortunes (MF) and Good Fortunes (GF) categories
of the H/VdC coding system. Codes negative events that happen TO characters (misfortunes,
sub-typed 1–6) and unexpectedly positive events that happen TO characters (good fortunes,
no sub-type), then evaluates against human-coded ground truth.

Usage:
    python misfortunes_good_fortunes.py                              # all b-baseline dreams (dev partition)
    python misfortunes_good_fortunes.py --n 50                       # first 50 b-baseline dreams
    python misfortunes_good_fortunes.py --collection norms-f         # validation partition
    python misfortunes_good_fortunes.py --collection norms-f --n 50  # first 50 norms-f dreams
    python misfortunes_good_fortunes.py --dream-id b-baseline_0062   # single dream
    python misfortunes_good_fortunes.py --all                        # full dataset

Output:
    mf_gf_results.csv — per-dream results with predicted and ground-truth MF pairs
    and GF character lists, plus F1 for each.

Notes on evaluation:
    MF F1 is computed at the attribute level: each (character, sub-type) pair is
    decomposed into (slot, value) tuples — number, gender, identity, age, sub-type —
    then scored via Counter intersection. This gives partial credit when the character
    is correct but the sub-type is wrong (or vice versa).

    GF F1 is computed the same way over character attributes only (no sub-type).

Methodology note:
    Misfortunes and Good Fortunes have ground-truth codings in b-baseline, norms-f,
    and norms-m. b-baseline (250 dreams, 107 with MF codings, 5 with GF codings)
    is the development partition. norms-f is used for validation. norms-m is reserved
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
OUTPUT_CSV        = "mf_gf_results.csv"
SAMPLE_SIZE       = 250
SAMPLE_COLLECTION = "b-baseline"
DELAY_SECONDS     = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# CODEBOOK
# ─────────────────────────────────────────────────────────────────────────────
CODEBOOK = """
Hall/Van de Castle (H/VdC) Misfortunes and Good Fortunes Coding Rules
======================================================================

## Overview

### Misfortunes (MF)
A Misfortune is a negative event that HAPPENS TO a character from outside — not
something they chose to do, but something undesirable that befalls them. Code WHO
experienced the misfortune and WHAT TYPE (1–6).

Misfortune coding is INDEPENDENT of Success and Failure coding. Accidents,
threats, illness, and death are coded as misfortunes even when the character
also tried to avoid them (the effort to avoid is Activities/Striving; the
negative event itself is the Misfortune).

### Good Fortunes (GF)
A Good Fortune is an unexpectedly positive event that happens TO a character —
not through their direct goal-directed effort (that would be a Success), but as
a stroke of luck, a rescue, an unexpected gift, or a happy windfall. Code WHO
received the good fortune. There are no sub-types for Good Fortune.

Good fortune and misfortune can co-occur in the same dream for the same
character (e.g., a boat crashes = Misfortune type 4, but the crash
accidentally improves the boat = Good Fortune for the same character).

─────────────────────────────────────────────────────────────────────────────
## The Dreamer and Character Codes

Use "D" for the dreamer.
Use standard H/VdC character codes: NUMBER + GENDER + IDENTITY + AGE
  e.g., 1MKA (one known male adult), 1FKA (one known female adult),
        1FSA (one female stranger adult), 2MKA (two known male adults)
  Relative codes (identity slot R): 1FRA (one female relative adult),
    1MRA (one male relative adult), etc. — use these for aunts, uncles, cousins,
    and other relatives when the specific relationship is mentioned.
  Animals: 1ANI / 2ANI

IMPORTANT — DO NOT use single-letter family codes:
  Never use M (mother), F (father), H (husband), W (wife), or B (brother)
  as standalone character codes. Even when the dream says "my mother" or
  "my father," use the descriptive format based on observable attributes:
  "1FMA" for one female middle-aged adult (mother), "1MKA" for one known
  male adult (father), etc. Single-letter family codes are NOT used in
  ground-truth codings.

─────────────────────────────────────────────────────────────────────────────
## MF Sub-types

The primary question for each sub-type: "What is the DOMINANT narrative element
of this misfortune scene — the character's emotional state (→ type 1), the
objective situation they are in (→ type 3), or the physical danger they face
from an external source (→ type 4)?"

### Type 1 — Apprehension (MOST COMMON)
The PRIMARY NARRATIVE ELEMENT is the character's WORRY, ANXIETY, FEAR, or
DREAD. Code type 1 when the dream focuses on HOW DISTRESSED the character feels
— including scenes where the character is actively managing a difficult
situation but the EMOTIONAL APPREHENSION is the central experience.
  - "I got worried," "I feared," "I was concerned," "I was anxious," "I dread"
  - Anticipating a bad outcome: "I know I'll flunk," "I was afraid"
  - Struggling with something difficult or threatening where the WORRY is the
    primary element, even if the character is actively coping
  Examples:
  - Dreamer must walk a tightrope to reach her seat (worry about the task)
  - Dreamer worried about a malfunctioning cash register and growing line
  - Dreamer must hold on to a dangerous animal (fear/struggle is the focus)
  - Dreamer feared she forgot to attend class and will flunk

### Type 2 — Physical accident / mishap
The character suffers an unintentional physical mishap or is accidentally harmed.
The harm is caused by accident, not by another character's deliberate act.
  - Slipping, falling, tripping, bumping into something
  - Equipment or structural failure that directly harms the character

### Type 3 — Adverse situation
The PRIMARY NARRATIVE ELEMENT is the SITUATION the character is OBJECTIVELY IN
— a seriously unpleasant or uncomfortable external circumstance that does NOT
rise to active physical danger or threat of serious bodily harm.
Code type 3 when the dream's focus is on the nature of the BAD SITUATION, not
primarily on the character's emotional apprehension, and NOT when the character
faces immediate physical danger (use type 4 for that).
  - Being formally scrutinized, questioned, accused, or interrogated by an
    authority figure — a socially threatening institutional interaction
  - Being in genuinely unpleasant environmental conditions (heavy rain on a
    barren road, displacement from one's home or room, being stranded)
  - Unfair, unjust, or humiliating circumstances the character is stuck in
  NOT code 3:
  - Routine embarrassment or mild social awkwardness — threshold not met
  - Situations where the character's worry (type 1) is the dominant element
  - Fire, building collapse, or other situations where the character faces
    IMMEDIATE RISK OF SERIOUS PHYSICAL HARM — use type 4 for those

### Type 4 — Physical jeopardy / passive victim of danger
The character is a PASSIVE VICTIM of acute physical danger that could cause
serious injury or death — from an external source such as another person, a
dangerous vehicle, fire, or a physically threatening condition.
  - Being in a vehicle driven recklessly or dangerously by another person
  - Being directly threatened or menaced by another person or creature
  - Being caught in a fire, building collapse, or other disaster with risk
    of serious physical harm (even if the character also tries to escape)
  - Being told that one will be placed in a physically dangerous situation
    (e.g., dreamer knows she'll be assigned to a room with a dangerous fire
    escape — the anticipatory physical jeopardy IS code 4)
  - Being physically endangered by conditions imposed from outside
  - An animal being caught, trapped, or physically endangered by any cause
    (including being accidentally caught by a human action) — code type 4 for
    the ANIMAL, not type 2 for the human who accidentally caused it
  NOT code 4:
  - Situations where the character is the active agent coping with danger
    (those are typically type 1 — apprehension during active management)
  - Unpleasant situations that are uncomfortable but not physically dangerous
    (use type 3 for those)

### Type 5 — Physical suffering
The character actually suffers physical illness, injury, pain, deprivation, or
disability. The harm has materialized — not feared, but present.
  - Illness: disease, needing medical treatment, being unwell
  - Injury: bruises, wounds, physical pain (described as existing, not feared)
  - Deprivation: starving, severely hungry, physically exhausted
  - Disability: physical or cognitive impairment limiting the character

### Type 6 — Death
A character LITERALLY dies in the dream narrative — the dreamer is told of a
death, or a character dies or is killed during the dream events.
  - "He died," "she was killed," "they found the body," being told of a death
  - A character visibly dying or being described as dead within the dream
  - Applies to humans AND animals
  NOT code 6:
  - Metaphorical or symbolic death ("something dying so the next phase can grow")
  - Dream imagery of coffins if the death itself is not narrated as an event
  - ANTICIPATED or FEARED death in an apocalyptic scenario: "the end of the world
    is coming and we will all die" — no one actually dies; this is type-3 (adverse
    situation), not type-6. Only code type 6 when death ACTUALLY OCCURS in the dream

─────────────────────────────────────────────────────────────────────────────
## Good Fortunes (GF) — no sub-type

Code a Good Fortune ONLY when something UNAMBIGUOUSLY and UNEXPECTEDLY positive
happens TO a character — not through their own deliberate goal-directed effort.
  - Being rescued or saved from a dangerous situation by luck or another character
  - Receiving an unexpected gift, money, or valuable object
  - Something that seemed bad turning out to be surprisingly good (windfall)
  - An unexpected positive physical or social change the character didn't earn

DO NOT code as GF:
  - Successes through the character's own explicit practice or effort
    (those are Success codes in the S&F module)
  - Ordinary pleasant activities (dancing, socializing, enjoyable meals)
  - Ambivalent outcomes ("pleased and not pleased at the same time")
  - A person showing up or returning — unless this is a clear windfall
  - Positive experiences that simply complete a scene without being surprising

─────────────────────────────────────────────────────────────────────────────
## CODING THRESHOLD — DO NOT OVER-CODE

MF coding is SELECTIVE. Human coders do not code every negative element in a
dream. About half of all dreams have NO misfortune codes at all. Only code
misfortunes that are SALIENT NARRATIVE ELEMENTS — events the dream devotes
meaningful attention to as a negative experience.

DO NOT code as MF:
  - Brief or passing social awkwardness, mild embarrassment, or minor criticism
    (e.g., "He said my hair was messed up," "I was a little embarrassed")
  - Emotional states (guilt, sadness, grief, disappointment) — these belong to
    the Emotions module, not MF
  - Routine inconveniences or frustrations that the character handles easily
  - Threatening dream imagery that does not materialize into a real threat
    (e.g., a scary scene that resolves without incident)
  - Institutional or systemic threats framed as unfair/unjust without direct
    physical harm (e.g., being sentenced for a minor offense in an anxiety
    dream where the injustice is the focus)
  - IMMEDIATELY RESOLVED apprehension: if the character expresses worry or
    fear that is quickly and fully resolved in the same scene (e.g., "I was
    suspicious but then a thought occurred that it was OK," or "I said I was
    afraid but a friend helped me and I felt proud"), do NOT code type 1 — the
    apprehension did not persist as a misfortune
  - A character COMPETENTLY MANAGING a difficult situation without expressing
    apprehension: if someone is in a tough spot but actively and confidently
    handles it (no worry language, no fear expression), do NOT code type 1

CRITICAL — NO DOUBLE-CODING RULE:
  - When a character experiences type 4 (physical jeopardy), type 5 (physical
    suffering), or type 6 (death) in the SAME EPISODE, do NOT also code type 1
    for that character's emotional reaction to it. The higher-numbered type
    captures the full event. Example: dreamer's teeth fall out and she panics
    = type 5 ONLY, not type 5 + type 1 for the panic. Dreamer is dying and
    worried about it = type 5 ONLY. Golf ball injures dreamer = type 5 ONLY,
    not type 4 + type 5.
  - When a character's type 4 RESULTS IN actual injury (type 5), code only
    type 5. Type 4 applies when the character is in danger but NOT yet hurt.
  - Do NOT code type 1 (apprehension) for a character's grief or dread about
    ANOTHER character's death or suffering. If someone learns their mother died
    (type 6 for mother), the dreamer's fear about the funeral is NOT also coded
    as type 1 for the dreamer.
  - Exception: type 1 and type 4/5/6 CAN co-occur for the same character ONLY
    if they refer to COMPLETELY SEPARATE, UNRELATED situations within the dream
    (e.g., the character both worries about an exam AND is separately in a
    dangerous vehicle — two distinct events).

Code misfortunes for ALL characters who experience them, not just the dreamer.
If a side character is explicitly described as suffering, ill, threatened,
dying, or in physical jeopardy, code THAT character's misfortune with the
appropriate type. When multiple characters are in the SAME dangerous situation
together, code each character separately (e.g., if a dreamer and a companion
are both in a dangerous vehicle, code type-4 for EACH character present).

─────────────────────────────────────────────────────────────────────────────
## OUTPUT FORMAT

Respond with ONLY valid JSON — no prose before or after — in this exact shape:

{
  "misfortunes": [
    {"char": "D", "type": 1},
    {"char": "D", "type": 4},
    {"char": "1MRA", "type": 6}
  ],
  "good_fortunes": ["D", "1MBA"],
  "reasoning": {
    "D type-1 (apprehension)": "Dreamer worried about failing the class — clear apprehension.",
    "D type-4 (physical jeopardy)": "Dreamer was in the car with a reckless driver — direct threat.",
    "1MRA type-6 (death)": "The male relative died during the dream.",
    "D gf": "The crash unexpectedly improved the boat — good fortune for the dreamer.",
    "NOT CODED — dreamer felt relieved": "Relief is not a misfortune; no code warranted."
  }
}

Field rules:
  - "misfortunes": list of objects, each with "char" (character code) and "type" (integer 1–6)
  - "good_fortunes": list of character codes (strings), no type needed
  - If a misfortune or good fortune applies to multiple characters as a group,
    list each character separately: [{"char": "D", "type": 4}, {"char": "1MBA", "type": 4}]
  - Empty lists when nothing is coded: "misfortunes": [], "good_fortunes": []
  - Include reasoning for each coded event AND for near-misses you decided NOT to code
"""

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES  (all from b-baseline dev partition)
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT = [
    # Example 1: Type 1 — Apprehension (dreamer worries about forgotten class)
    {
        "dream": (
            'As I was "waking up" from a nap, Ginny and Rosemary were perched on my bed. '
            "Ginny was doing Math. I watched her count columns, carry and write numbers as she "
            'figured and I said, "You do math like me." She wasn\'t doing well. Then I got '
            'worried. I said, "Oh Lord, I\'m sure that I\'m signed up for an intro to German '
            "class, but I forgot to go to any classes and it's late in the term. I'll flunk.\" "
            "I started looking through my filing cabinet for the register form. I saw some "
            "B'day cards scrunched in there. Rosemary said, \"Take those out.\" I had to see a "
            "boy standing on his head trying to say something but no one would listen. It was a "
            "B'day card to the devil. He had a grotesque red face, with huge reddish teeth. He "
            "was standing very large in a formal striped suit with a tie and carnation and a "
            "huge overcoat on. His coarse, snaky tail was draped around him. No fear."
        ),
        "misfortunes": [{"char": "D", "type": 1}],
        "good_fortunes": [],
        "reasoning": {
            "D type-1 (apprehension)": (
                "The dreamer suddenly worries she signed up for a German class and forgot to "
                "attend, fearing she will flunk — clear, explicit apprehension about a bad "
                "outcome. All other events (watching Ginny do math, birthday card, devil image) "
                "contain no MF or GF coding — they are activities or setting details."
            ),
            "NOT CODED — Ginny not doing well at math": (
                "Ginny's difficulty is an activity (cognitive), not coded as a misfortune for "
                "Ginny — there is no indication she is suffering, threatened, or in jeopardy."
            ),
            "NOT CODED — grotesque devil image": (
                "A strange visual detail; 'No fear' is explicitly stated. No apprehension or "
                "misfortune warranted."
            ),
        },
    },
    # Example 2: Type 3 — Adverse situation (social scrutiny)
    {
        "dream": (
            "A vocational rehab counselor was questioning me about my cousin Sonja and her "
            "capabilities as a VR counselor in the coast office. I told him that Sonja and I "
            "go back as far as birth. She had a few things to learn yet, but she was pretty "
            "good. He said he wanted to know more. We got into a car and drove to the M City "
            "house. There was some snow on the ground. I suggested that he park in the driveway. "
            'I say, "Turn right into the driveway." He obstinately turned left into a bank of '
            "dirt and snow. He and I agree that that won't do. He turned the car around and "
            'parked on the street, facing the "bad corner." I was concerned but saw another car '
            "was parked right on the corner so that we were protected from collision. We go into "
            "the house. I took him the full length of the house to the far end. I say, \"Let's "
            'start at the oldest part." It was a freshly remodeled room, empty of furniture, '
            "and freshly painted. I told him how this used to be a half-finished garage. I "
            "pointed out where the grape vines used to be. It was now a lovely wall with "
            "windows. We moved to the next room, the rec room. I pointed to the door. I said, "
            '"It used to be somewhere else. I still can\'t get used to it being there and it '
            'startles me when it opens. It too has been freshly remodeled." As he and I had '
            "driven up to the house, I had pointed out the rising creek that was in flood stage. "
            "I told him that sometimes the creek came right up to the door and sometimes came "
            "in and started to flood the house."
        ),
        "misfortunes": [{"char": "D", "type": 3}],
        "good_fortunes": [],
        "reasoning": {
            "D type-3 (adverse situation — social scrutiny)": (
                "The dreamer is being formally questioned by an authority figure (vocational rehab "
                "counselor) about a professional matter — a socially uncomfortable and scrutinizing "
                "situation that the dreamer did not initiate. This is type-3 adverse situation "
                "(social adversity: being questioned/evaluated by an authority)."
            ),
            "NOT CODED — creek in flood stage": (
                "The flood-prone creek is mentioned as background information the dreamer shares "
                "with the counselor; it is not actively threatening or affecting the characters "
                "during the dream scene itself. No MF code for situational information."
            ),
            "NOT CODED — concern about parking at 'bad corner'": (
                "Momentary concern that resolved immediately when the dreamer saw a protective "
                "car. This is a brief apprehensive reaction, not a sustained misfortune — the "
                "situation was resolved before it warranted coding."
            ),
        },
    },
    # Example 3: Type 4 — Physical jeopardy (dangerous driving)
    {
        "dream": (
            "Howard is driving a beat up VW bus. His wife Karen, the kids and I are in the bus. "
            "I am in the passenger seat. I see Howard going to sleep as he drives. I am angry. "
            "I yank the driver's seat from him. We argue. He is on a bended knee next to my "
            "right side, trying to get close to me. I feel very closed in, rigid. Karen pleads "
            "with me to see his need to drive and to be close. I'm angry and I argue that it's "
            "best for me to drive and that I don't like him anyway, so there! The brakes are "
            "weak and I have trouble stopping or slowing us down when it is needed."
        ),
        "misfortunes": [{"char": "D", "type": 4}],
        "good_fortunes": [],
        "reasoning": {
            "D type-4 (physical jeopardy — dangerous vehicle)": (
                "The dreamer is in a vehicle with a driver who falls asleep at the wheel, then "
                "takes over but finds the brakes are weak and can't stop — direct, ongoing "
                "physical danger from a dangerous driving situation. Type-4 physical jeopardy."
            ),
            "NOT CODED — 'I am angry' / 'I feel very closed in, rigid'": (
                "Emotional states (anger, feeling closed in). These belong to the Emotions "
                "module, not Misfortunes."
            ),
            "NOT CODED — Howard, Karen, or the kids": (
                "The other characters are present in the same dangerous vehicle, but only the "
                "dreamer is coded by the human coders for this dream. Code what the ground truth "
                "reflects — not every occupant need be independently coded."
            ),
        },
    },
    # Example 4: Type 6 — Death of relative; no GF
    {
        "dream": (
            "My mother tells me that Aunt Muriel died. I am surprised because I had just written "
            "to her because I'd dreamed about her last week. I started looking for the dream book "
            "to show my mother what I'd written. I can't find it. I pick up a child's schoolbook "
            "and read it until I realize it's not the dream book. I had been packing to go visit "
            "my mother and I find the dream book. I open it to a dream where a beautifully drawn "
            "picture is on the page and the dream was written over it. I am very impressed. I had "
            "no idea I was capable of such excellent drawing skills. It is a vague cousin V, her "
            "husband and their baby."
        ),
        "misfortunes": [{"char": "1FRA", "type": 6}],
        "good_fortunes": [],
        "reasoning": {
            "1FRA type-6 (death)": (
                "Aunt Muriel (a female relative adult, coded 1FRA) is reported as having died. "
                "Death of a character = type-6 misfortune."
            ),
            "NOT CODED — dreamer can't find the dream book": (
                "This is a frustration/failure, not a misfortune. The dreamer's own inability "
                "to locate an object belongs to the Success and Failure module (fail), not MF."
            ),
            "NOT CODED — dreamer discovers drawing ability": (
                "The dreamer is surprised and impressed by her own drawing skill — this looks "
                "like a good fortune. However, it is a retrospective discovery, not an "
                "unexpected event that happens during the active dream scene. The human coders "
                "did not code a GF here. When in doubt about GF, do not code it."
            ),
        },
    },
    # Example 5: Type 4 + GF (co-occurring misfortune and good fortune)
    {
        "dream": (
            "I go to a house. It is supposedly Jake's house. He's having a party. I take the "
            "opportunity to do a tour of the rooms: living room, a family room, two bedrooms. "
            "Jake comes in. He suggests we join the others in the boat. He and I get in a row "
            "boat with a small motor on it. I ask, \"Are we going further out?\" He says, \"No. "
            "We're just going around the corner.\" We go around and head for a narrow cement "
            "chute that was filled with sand, not water. We slide down fast. I'm in the front. "
            "The front crashes on the cement barrier and curls the front of the boat up. This "
            "has improved the boat's capabilities. We get out and return to the house. Lots of "
            "people are there. Patricia is there with the Mexican Tony. They are kissing. I look "
            "around for Hector and can't find him."
        ),
        "misfortunes": [
            {"char": "D", "type": 4},
            {"char": "1MBA", "type": 4},
        ],
        "good_fortunes": ["D", "1MBA"],
        "reasoning": {
            "D type-4 + 1MBA type-4 (physical jeopardy — crashing boat)": (
                "The dreamer and Jake (her male brother, 1MBA) slide into a cement chute in a "
                "rowboat and crash into the cement barrier — direct physical jeopardy for both. "
                "Type-4 for each character."
            ),
            "D gf + 1MBA gf": (
                "The crash, despite being dangerous, unexpectedly improves the boat's "
                "capabilities — a wholly unearned positive outcome that neither character "
                "intended or worked toward. Good Fortune for both."
            ),
            "NOT CODED — 'I can't find Hector'": (
                "The dreamer looks for Hector and can't find him. This might seem like a "
                "Failure, but there is no explicit goal ('I tried to find Hector') or "
                "explicit outcome language — it's a passing observation. Not coded."
            ),
        },
    },
    # Example 6: NO coding — brief embarrassment and passing comments do not meet threshold
    {
        "dream": (
            "For the second night in a row, I dreamed of Jon. He was trying to make an "
            "impression on me. He embarrassed me! Then I dreamed that Reta and I saw Blake "
            "in a restaurant and we went in and had a coke. Some kids were dancing and I was "
            "wishing Blake would ask me. He did. We bopped all over. When it was done, we sat "
            "down. He said, \"Your hair's messed up.\" I said, \"Oh, dear, not again\" (He had "
            "his arm around me, though.)"
        ),
        "misfortunes": [],
        "good_fortunes": [],
        "reasoning": {
            "NOT CODED — 'He embarrassed me'": (
                "A brief, passing embarrassing moment. This does not meet the threshold for "
                "type-3 coding — the embarrassment is not sustained, not caused by formal "
                "authority, and is not the central narrative focus. Mild social awkwardness "
                "is not a codeable misfortune."
            ),
            "NOT CODED — 'I was wishing Blake would ask me'": (
                "Wishful thinking or desire is not apprehension (type 1). There is no "
                "expressed fear or dread here — only a wish."
            ),
            "NOT CODED — 'Your hair's messed up'": (
                "A passing social comment. Not formal scrutiny or criticism warranting "
                "type-3. The dreamer's mild reaction ('Oh, dear') confirms it's trivial."
            ),
            "NOT CODED GF — Blake asking her to dance": (
                "Blake asking her to dance is a pleasant social event, not a windfall or "
                "unexpected stroke of luck that rises to the Good Fortune threshold."
            ),
        },
    },
    # Example 7b: Type 1 + GF — creative blockage is apprehension; unexpected inspiration is GF
    {
        "dream": (
            "Someone shot an arrow at a helicopter and it crashed in flames. A narrator said, "
            '"Woody sort of threw this thing and it made a noise and kind of hurt it." I '
            'thought, "What a cover up! Disgusting." I saw a car with huge baskets on the top '
            "of it driving away. I was writing a play. It called for a new character. I got "
            "stuck and then inspiration struck and I wrote one in."
        ),
        "misfortunes": [{"char": "D", "type": 1}],
        "good_fortunes": ["D"],
        "reasoning": {
            "D type-1 (apprehension — creative block)": (
                "The dreamer was writing a play, needed a new character, and GOT STUCK — being "
                "unable to produce or create something important is a form of apprehension/distress "
                "that rises to the MF type-1 threshold."
            ),
            "D gf (unexpected inspiration)": (
                "'Inspiration struck and I wrote one in' — the creative solution arrived "
                "unexpectedly, without deliberate effort. This is a windfall/unexpected positive "
                "event = Good Fortune."
            ),
            "NOT CODED — helicopter crash and flames": (
                "The dreamer witnesses a helicopter crash but is not part of it — no MF coding "
                "for witnessed events unless the character is personally affected. The dreamer's "
                "reaction is moral indignation ('disgusting'), not personal apprehension."
            ),
        },
    },
    # Example 8: Mixed codes — Type 1, 5, and Type 6 for different characters
    {
        "dream": (
            "The stray kittens plead for food. They are very hungry. I feel badly for them. "
            "I look in the refrigerator. I find some sugar cakes and some cheese. I am dressing "
            "up to go out on a date. It's a conservative outfit, but as soon as I'm out the "
            "door, I'll readjust the front and it will be very sexy. There's a bruise on my "
            "neck. I have a belt around my neck and then I put it around my waist. My kids "
            "watch. I go outside, stairs. I say, \"Where's the elevator?\" They say, \"No "
            "problem. You can go up the stairs.\" I am annoyed. It will be painful. I go up "
            "one flight. Then I sit down and say, \"No, I'll go in the elevator. These stairs "
            "don't go where I want anyway.\" (they went back doors). I see penguins that live "
            "on the stairs. If radio activity gets bad, they die and then we know. Some lie "
            "down dead."
        ),
        "misfortunes": [
            {"char": "D", "type": 5},
            {"char": "D", "type": 1},
            {"char": "2ANI", "type": 6},
        ],
        "good_fortunes": [],
        "reasoning": {
            "D type-5 (physical suffering — injury)": (
                "The dreamer has a bruise on her neck and anticipates that climbing the stairs "
                "will be painful — actual physical injury and suffering already present."
            ),
            "D type-1 (apprehension — distress for kittens)": (
                "The dreamer feels badly for the hungry kittens — an apprehensive, distressed "
                "emotional response to their suffering."
            ),
            "2ANI type-6 (death — penguins / animals die)": (
                "Some of the penguins lie down dead at the end of the dream — death of animals "
                "(coded 2ANI for two or more animals) = type-6 misfortune."
            ),
            "NOT CODED — kittens being hungry": (
                "The kittens' hunger could warrant a type-5 (deprivation) for the animals, "
                "but the human coders did not code it separately — only the dreamer's apprehensive "
                "response (type-1) was coded. Follow the conservative threshold."
            ),
            "NOT CODED — going on a date, dressing up": (
                "Routine activities; no misfortune or good fortune involved."
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

    mf_df = codings[codings["coding_type"] == "mf"].copy()
    gf_df = codings[codings["coding_type"] == "gf"].copy()

    def rows_to_mf_pairs(g):
        pairs = []
        for _, r in g.iterrows():
            code    = r.get("code", "")
            char_raw = r.get("char", "")
            if pd.isna(code) or pd.isna(char_raw):
                continue
            subtype = str(code).strip()
            for c in str(char_raw).split("+"):
                c = c.strip()
                if c:
                    pairs.append((c, subtype))
        return pairs

    def rows_to_chars(g):
        chars = []
        for _, r in g.iterrows():
            val = r.get("char", "")
            if pd.notna(val) and str(val).strip():
                for c in str(val).split("+"):
                    c = c.strip()
                    if c:
                        chars.append(c)
        return chars

    mf_gt = mf_df.groupby("dream_id").apply(rows_to_mf_pairs).to_dict()
    gf_gt = gf_df.groupby("dream_id").apply(rows_to_chars).to_dict()
    return dreams, mf_gt, gf_gt


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────────────────
def build_system_content():
    ex_text = ""
    for ex in FEW_SHOT:
        output = {
            "misfortunes":   ex["misfortunes"],
            "good_fortunes": ex["good_fortunes"],
            "reasoning":     ex["reasoning"],
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
        + "If neither misfortunes nor good fortunes occur, return: "
        + '{"misfortunes": [], "good_fortunes": [], "reasoning": {}}\n'
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(client, dream_text, system_content):
    """Call Claude and return (mf_pairs, gf_chars, reasoning)."""
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
                    "Code all misfortunes and good fortunes in this dream report:\n\n"
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

    mf_raw = parsed.get("misfortunes", [])
    mf_pairs = [(item["char"], str(item["type"])) for item in mf_raw
                if isinstance(item, dict) and "char" in item and "type" in item]

    gf_chars = parsed.get("good_fortunes", [])
    reasoning = parsed.get("reasoning", {})
    return mf_pairs, gf_chars, reasoning


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


def decompose_mf_pair(char, subtype):
    """Decompose (character, MF-subtype) into (slot, value) tuples for F1 scoring."""
    attrs = decompose_character(char)
    attrs.append(("subtype", str(subtype)))
    return attrs


def evaluate_mf(pred_pairs, gt_pairs):
    """Attribute-level F1 over MF (char, subtype) pairs via Counter intersection."""
    pred = Counter(item for (c, t) in pred_pairs for item in decompose_mf_pair(c, t))
    true = Counter(item for (c, t) in gt_pairs  for item in decompose_mf_pair(c, t))
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


def evaluate_gf(pred_chars, gt_chars):
    """Attribute-level F1 over GF character codes (no sub-type)."""
    pred = Counter(item for c in pred_chars for item in decompose_character(c))
    true = Counter(item for c in gt_chars   for item in decompose_character(c))
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
    parser = argparse.ArgumentParser(
        description="H/VdC Misfortunes and Good Fortunes Coder"
    )
    parser.add_argument("--all",        action="store_true",
                        help="Run on the full dataset")
    parser.add_argument("--collection", type=str, default=None,
                        help="Restrict to one collection (e.g. b-baseline, norms-f)")
    parser.add_argument("--dream-id",   type=str, default=None,
                        help="Run on a single dream by ID")
    parser.add_argument("--n",          type=int, default=None,
                        help="Override sample size")
    parser.add_argument("--skip",       type=int, default=0,
                        help="Skip the first N dreams from the selected collection")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)
    system_content = build_system_content()

    dreams, mf_gt, gf_gt = load_data()

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

        gt_mf = mf_gt.get(dream_id, [])
        gt_gf = gf_gt.get(dream_id, [])

        try:
            pred_mf, pred_gf, reasoning = call_claude(
                client, str(dream_text), system_content
            )
        except Exception as e:
            print(f"  ERROR {dream_id}: {e}")
            results.append({
                "dream_id":     dream_id,
                "pred_mf":      "[]",
                "gt_mf":        json.dumps(gt_mf),
                "pred_gf":      "[]",
                "gt_gf":        json.dumps(gt_gf),
                "f1_mf":        None,
                "f1_gf":        None,
                "mean_f1":      None,
                "precision_mf": None,
                "recall_mf":    None,
                "precision_gf": None,
                "recall_gf":    None,
                "reasoning":    f"ERROR: {e}",
                "dream_report": str(dream_text)[:300],
            })
            time.sleep(DELAY_SECONDS)
            continue

        m_mf   = evaluate_mf(pred_mf, gt_mf)
        m_gf   = evaluate_gf(pred_gf, gt_gf)
        mean_f1 = round((m_mf["f1"] + m_gf["f1"]) / 2, 3)

        has_gt  = len(gt_mf) > 0 or len(gt_gf) > 0
        marker  = "" if has_gt else "  [no gt]"

        print(
            f"  {'✓' if mean_f1 == 1.0 else '✗'}  {dream_id}"
            f"  mf_pred={pred_mf}  mf_gt={gt_mf}"
            f"  gf_pred={pred_gf}  gf_gt={gt_gf}"
            f"  F1_mf={m_mf['f1']:.2f}  F1_gf={m_gf['f1']:.2f}"
            f"  mean={mean_f1:.2f}{marker}"
        )

        results.append({
            "dream_id":     dream_id,
            "pred_mf":      json.dumps(pred_mf,  ensure_ascii=False),
            "gt_mf":        json.dumps(gt_mf,    ensure_ascii=False),
            "pred_gf":      json.dumps(pred_gf,  ensure_ascii=False),
            "gt_gf":        json.dumps(gt_gf,    ensure_ascii=False),
            "f1_mf":        m_mf["f1"],
            "f1_gf":        m_gf["f1"],
            "mean_f1":      mean_f1,
            "precision_mf": m_mf["precision"],
            "recall_mf":    m_mf["recall"],
            "precision_gf": m_gf["precision"],
            "recall_gf":    m_gf["recall"],
            "reasoning":    json.dumps(reasoning, ensure_ascii=False),
            "dream_report": str(dream_text)[:300],
        })

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
        print(f"Mean F1 (MF)     : {valid['f1_mf'].mean():.3f}")
        print(f"  Precision      : {valid['precision_mf'].mean():.3f}")
        print(f"  Recall         : {valid['recall_mf'].mean():.3f}")
        print(f"Mean F1 (GF)     : {valid['f1_gf'].mean():.3f}")
        print(f"  Precision      : {valid['precision_gf'].mean():.3f}")
        print(f"  Recall         : {valid['recall_gf'].mean():.3f}")
        print(f"Mean F1 (overall): {valid['mean_f1'].mean():.3f}")

        has_gt_mf = valid[valid["gt_mf"] != "[]"]
        has_gt_gf = valid[valid["gt_gf"] != "[]"]
        if len(has_gt_mf) > 0:
            print(f"\nDreams with MF GT codings: {len(has_gt_mf)}")
            print(f"  F1 MF   : {has_gt_mf['f1_mf'].mean():.3f}")
        if len(has_gt_gf) > 0:
            print(f"Dreams with GF GT codings: {len(has_gt_gf)}")
            print(f"  F1 GF   : {has_gt_gf['f1_gf'].mean():.3f}")


if __name__ == "__main__":
    main()
