#!/usr/bin/env python3
"""
social_interactions.py — Hall/Van de Castle Social Interactions Coder

Semi-automated classifier for the Social Interactions category of the H/VdC coding system.
Codes all aggressive (agg), friendly (fri), and sexual (sex) interactions in dream reports
using Claude, then evaluates against human-coded ground truth.

Usage:
    python social_interactions.py                               # sample of 50 b-baseline dreams
    python social_interactions.py --n 15                        # sample of 15 dreams
    python social_interactions.py --all                         # full dataset
    python social_interactions.py --collection norms-f          # one collection
    python social_interactions.py --dream-id b-baseline_0003    # single dream

Output:
    social_interactions_results.csv — per-dream results with predicted and ground-truth
    interaction tuples, plus F1 overall and broken out by sub-type (agg / fri / sex).

Notes on evaluation:
    Each interaction is a tuple (init, rec, type, code). F1 is computed via Counter
    intersection, the same approach used in characters.py. Because the LLM assigns its
    own character codes, evaluation is conservative: credit is only given when the
    predicted tuple exactly matches ground truth including the character codes.
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
OUTPUT_CSV        = "social_interactions_results.csv"
SAMPLE_SIZE       = 50
SAMPLE_COLLECTION = "b-baseline"
DELAY_SECONDS     = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# CODEBOOK
# ─────────────────────────────────────────────────────────────────────────────
CODEBOOK = """
Hall/Van de Castle (H/VdC) Social Interactions Coding Rules
===========================================================

## Overview

Code all deliberate social interactions between characters and/or the dreamer.
Three types of interaction are coded: Aggressive (agg), Friendly (fri), Sexual (sex).

## The Dreamer
Always use "D" to represent the dreamer. The dreamer CAN be an initiator or recipient.
D is NOT a character code — it is a fixed symbol for the person having the dream.

## Character Codes
Other characters use H/VdC character codes in the format: NUMBER + GENDER + IDENTITY + AGE
  Examples: 1MKA (individual male known adult), 2IEA (group indeterminate ethnic adult),
            1FMA (individual female, mother, adult), 1ANI (individual animal)
  When multiple characters act together as a unit: join with " + " (e.g., "1MPA + 2IUA")

## What is Codeable

CRITICAL THRESHOLD — when in doubt, DO NOT code. H/VdC only codes
DELIBERATE, CLEARLY-DIRECTED social acts. Human coders are conservative;
they do NOT code the following:

  - INFORMATION DELIVERY AS PART OF A ROLE: a doctor diagnosing a patient,
    a teacher demonstrating equipment, a professional explaining procedure,
    a salesperson describing a product. The role-based exchange is not a
    friendly social act.

  - STATIC STATES OR POSITIONS: "she was sitting next to him," "they stood
    together," "she was on his lap," "he was holding the letter." States
    and positions are not interactions — interactions require an ACTION.

  - NEGATIVE-SOUNDING SPEECH WITH NON-HOSTILE INTENT: concern, advice,
    warnings, suggestions. Examples that are NOT A2 verbal aggression:
      • "I told him he was foolish to drive in the snow" (concern/advice)
      • "I warned her not to do that" (caution, not hostility)
      • "I said it would be a mistake" (advice, not insult)
    A2 requires HOSTILE INTENT to harm, insult, or intimidate.

  - INTERNAL FEELINGS THE DREAMER HAS ABOUT A CHARACTER: anxiety, worry,
    hope, uncertainty, longing. "I was anxious that he should not be seen"
    is a dreamer state, NOT a covert hostility (A1). A1 requires hostility
    DIRECTED AT a target, not just any negative feeling.

  - PASSIVE OR IMPERSONAL EVENTS: "I was introduced to them," "we met up,"
    "we ran into each other." Code only when a specific actor performs a
    deliberate act with clear initiative.

  - ROUTINE LOGISTICAL EXCHANGES: paying for things, asking directions,
    ordering food, swapping rooms. Not coded unless emotionally charged
    with deliberate friendly or hostile intent.

When a dream is short or primarily descriptive, lean toward FEWER codes,
not more. Missing a borderline interaction costs less than inventing one.

─────────────────────────────────────────────────────────────────────────────

Code an interaction when it is:
  - Physically enacted in the dream (seen, heard, directly experienced)
  - Witnessed by the dreamer even if the dreamer does not participate
  - For aggression: a DELIBERATE, INTENTIONAL act meant to harm or annoy
  - For friendliness: a PURPOSEFUL act of support, help, or positive social contact
  - For sexuality: any explicitly sexual or romantic (non-greeting) content

Do NOT code:
  - Accidental harm (e.g., a car crash — misfortune, not aggression)
  - Vague hostility with no specific act directed at anyone
  - Interactions only recalled, imagined, or mentioned in passing — not enacted in the dream
  - Greeting or farewell kisses (code as F2 friendly, NOT sexual)
  - Internal feelings the dreamer experiences ("I felt afraid," "I felt
    embarrassed," "a feeling of being chased," "I felt pursued"). These are
    NOT interactions unless the dream explicitly identifies an actor performing
    a deliberate act that caused the feeling. A vague feeling without a
    specific aggressor or actor is not coded.

─────────────────────────────────────────────────────────────────────────────
## ONE EVENT = ONE CODE (CRITICAL — STRICT RULE)

Each distinct interaction event gets exactly ONE code per (init, rec, type,
code) tuple. If a single act (one kiss, one yell, one chase) is described
across multiple sentences or from multiple angles, do NOT emit the same
tuple twice.

DECISION TEST before emitting any tuple that would duplicate another:
  Ask: "Do these describe the SAME physical event from different angles,
        OR TWO clearly separate moments in time?"

  → SAME event described over multiple sentences (one continuous chase,
    one kiss spread across "he leaned in... and kissed me... the kiss
    was real")  →  ONE code total, NOT two.

  → TWO temporally separate events  →  separate codes are appropriate
    ONLY when the text contains explicit time-separation cues:
      • "she came back later"
      • "he kissed me again"
      • "then much later"
      • a clear scene change between the two acts
    Without such explicit cues, default to ONE code.

Before adding any tuple to your output, scan the existing list. If an
identical (init, rec, type, code) tuple is already present, do NOT add
another unless the dream text EXPLICITLY describes a second, separate
occurrence with a time gap.

─────────────────────────────────────────────────────────────────────────────
## DIRECTION CODES  (appended to every sub-type number)

  >  = One-directional: initiator acts toward recipient
  =  = Mutual: both parties engage equally, no clear single initiator
  R  = Rejected: the recipient actively resists or counterattacks the initiator's act
  *  = Self-directed: the character acts toward themselves (no recipient; rec = null)

─────────────────────────────────────────────────────────────────────────────
## AGGRESSIVE INTERACTIONS  (type = "agg")

An aggressive act is DELIBERATE or INTENTIONAL, meant to harm or annoy another.
Accidental harm is NOT aggression — it is a misfortune.

Sub-types (1–8, ordered by severity):
  1  Covert/internal hostility — felt but not overtly expressed (e.g., silent anger)
  2  Verbal aggression — yelling, swearing, criticizing, insulting, taunting
  3  Rejection or verbal coercion — refusals, dismissals, demands, exploitation
  4  Serious verbal threats — explicit threat to physically harm another
  5  Theft or destruction of property
  6  Physical coercion — chasing, capturing, confining, kidnapping
  7  Physical violence — hitting, slapping, shooting, stabbing, weapon use
  8  Lethal violence — murder or attempted murder

AGGRESSION RULES:
  - Code each distinct aggressive act separately (different sub-type OR separate moment).
  - Counterattack: if the victim fights back, add a second row with victim as initiator
    and the direction code R (e.g., 1ANI→D: 7> then D→1ANI: 6R).
  - Simultaneous mutual fighting from the start: use = direction.
  - Self-criticism, self-harm, or self-blame: use * with rec = null.
  - Witnessed aggression (dreamer observes without participating) IS codeable
    with the actual aggressor and victim, not D.
  - "He embarrassed me" alone is NOT aggression unless the text clearly indicates
    deliberate intent to humiliate.
  - NOT aggression by itself: "looking for," "seeking," "approaching,"
    "going to find," "calling for," "moving toward." These describe movement
    or attention, not deliberate harm. Aggression requires an intentional
    act DIRECTED AT a target with intent to harm or annoy. A character
    simply moving toward another is not enough; there must be an explicit
    hostile act (chasing to capture, threatening, attacking, demanding).

─────────────────────────────────────────────────────────────────────────────
## FRIENDLY INTERACTIONS  (type = "fri")

A friendly act is purposeful support, help, kindness, or positive social contact.

Sub-types (1–7):
  1  Friendliness felt internally but not overtly expressed (e.g., wishing someone well)
  2  Verbal/gestural expressions — greeting, waving, smiling, phoning/writing for
       friendly purpose, introducing one person to another
  3  Offering gifts or loaning possessions
  4  Extending assistance or protection — helping, rescuing, protecting, guiding
  5  Initiative in requesting shared social activities — invitations, dates, visits
  6  Socially acceptable physical contact — hugging, cuddling, dancing,
       non-sexual kissing, shaking hands, arm around someone
  7  Long-term commitment — falling in love, engagement, marriage proposal

FRIENDLINESS RULES:
  - Code each different sub-type separately when multiple occur between the same pair.
  - Repeated identical acts in the same temporal block = one code. Time-separated
    identical acts = separate codes.
  - A greeting kiss (peck hello/goodbye) is F2, NOT sexual.
  - Professional help (doctor treating, firefighter rescuing) still counts as F4.
  - The same physical act (e.g., a romantic kiss) can be coded BOTH as F6 (physical
    contact) AND S3 (romantic kiss) — they are not mutually exclusive.
  - For mutual acts (e.g., dancing together), use = direction with the person who
    initiated or list them in any consistent order.
  - NOT friendliness: "I see X," "X is there," "I noticed Y," "I was with Z,"
    "Y was nearby," "we were in the same room." Visual recognition or shared
    presence is NOT a friendly interaction. F2 (verbal/gestural friendliness)
    requires an actual ACT directed at the recipient — a greeting, smile,
    wave, conversation, or signal. Mere co-presence is not coded.
  - Just naming a character ("my friend Sarah came in") does not require
    coding any friendly interaction unless an actual friendly act follows.
  - NEUTRAL CONVERSATION IS NOT FRIENDLINESS. Casual or neutral verbal
    exchanges — observations, descriptions, statements of fact, neutral
    questions — do NOT qualify as F2. F2 requires the verbal act to be
    SUPPORTIVE, ENCOURAGING, WELCOMING, or POSITIVE: a greeting ("Hi!"),
    compliment ("You look great"), expression of care ("I'm glad you're
    here"), kind question ("How are you doing?"). Things like:
      • "I said, 'X is happening'" (statement of fact)
      • "I commented on her work"
      • "We chatted for a moment" (without specific positive content)
      • "She said something about the weather"
    are NOT friendly. They are neutral conversation and should not be coded.

─────────────────────────────────────────────────────────────────────────────
## SEXUAL INTERACTIONS  (type = "sex")

Any explicitly sexual or romantic (non-greeting) contact, proposition, or fantasy.

Sub-types (1–5):
  1  Sexual fantasies or thoughts about a specific character
  2  Sexual propositions or advances
  3  Romantic/passionate kissing (NOT a greeting or farewell kiss)
  4  Non-intercourse sexual activities — fondling, petting, masturbation
  5  Sexual intercourse or attempted intercourse

SEXUAL RULES:
  - A romantic/passionate kiss = S3. A greeting/farewell peck = F2 (not sexual).
  - Witnessed sexuality (dreamer watches) IS codeable with the actual participants.
  - Self-directed sexuality (solitary act) uses * with rec = null.
  - DEFAULT TO SEX ALONE for clearly romantic/sexual acts. When an interaction is
    clearly romantic — kissing passionately, embracing romantically, falling in
    love, sexual intercourse — code it ONLY as sex. Do NOT also add a parallel
    friendly code (fri 6 physical contact, fri 7 long-term commitment) unless
    the friendly dimension is clearly SEPARATE from the romantic act (e.g., a
    distinct hug occurring outside the romantic context). When in doubt
    between "sex only" vs. "sex + fri," pick sex alone. Human coders are
    conservative on this — they rarely double-code.

─────────────────────────────────────────────────────────────────────────────
## ANIMAL INTERACTIONS

Routine human-animal interactions (calling pets, feeding them, playing with
them, holding/cuddling pets, pet care, watching pets play) are generally
NOT coded as friendly interactions in H/VdC.

The fri/agg/sex codes for animals are reserved for:
  - Explicit AGGRESSION by an animal toward a person or another animal
    (attacks, threats, biting, chasing to harm)
  - Explicit AGGRESSION by a person toward an animal (hitting, killing,
    capturing the animal as a hostile act)
  - Aggression between animals that is clearly hostile (predator-prey,
    territorial fights)

Affectionate handling, ordinary care, and routine companionship with pets
(petting a dog, calling a cat over, feeding kittens, training a horse,
walking a pet) are NOT coded as friendly social interactions.

─────────────────────────────────────────────────────────────────────────────
## OUTPUT FORMAT

Respond with ONLY valid JSON — no prose before or after — in this exact shape:

{
  "interactions": [
    {"init": "D",    "rec": "1MKA", "type": "agg", "code": "2>"},
    {"init": "1MKA", "rec": "D",    "type": "fri", "code": "6>"},
    {"init": "D",    "rec": null,   "type": "agg", "code": "1*"}
  ],
  "reasoning": {
    "agg D→1MKA 2>": "Dreamer yells at character — verbal aggression.",
    "fri 1MKA→D 6>": "Character hugs dreamer — physical friendly contact.",
    "agg D→SELF 1*": "Dreamer silently blames herself."
  }
}

Field rules:
  - "init": character code or "D". Groups: "1MKA + 2IUA".
  - "rec": character code, "D", or null (self-directed only).
  - "type": exactly one of "agg", "fri", "sex".
  - "code": sub-type digit + direction symbol, e.g. "2>", "6=", "3R", "1*".
  - If NO social interactions occur in the dream, return:
    {"interactions": [], "reasoning": {}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT = [
    {
        "dream": (
            'For the second night in a row, I dreamed of Jon. He was trying to make an impression '
            'on me. He embarrassed me! Then I dreamed that Reta and I saw Blake in a restaurant '
            'and we went in and had a coke. Some kids were dancing and I was wishing Blake would '
            'ask me. He did. We bopped all over. When it was done, we sat down. He said, "Your '
            'hair\'s messed up." I said, "Oh, dear, not again" (He had his arm around me). He '
            'bent to me and said, "Yes-let\'s mess up a mustache" and then he kissed me. The kiss '
            'was so real! Then I dreamed that Nate came home. I met him in S City at the bus '
            'depot. He was 6 feet tall and all muscles. He grabbed my hands and then threw his '
            'arms around me and kissed me.'
        ),
        "interactions": [
            {"init": "D",    "rec": "1MKA", "type": "fri", "code": "1>"},
            {"init": "1MKA", "rec": "D",    "type": "fri", "code": "5>"},
            {"init": "D",    "rec": "1MKA", "type": "fri", "code": "6="},
            {"init": "1MKA", "rec": "D",    "type": "fri", "code": "6>"},
            {"init": "1MKA", "rec": "D",    "type": "fri", "code": "6>"},
            {"init": "1MKA", "rec": "D",    "type": "sex", "code": "3>"},
            {"init": "1MKA", "rec": "D",    "type": "sex", "code": "3>"},
        ],
        "reasoning": {
            "fri D→1MKA 1>": (
                "Dreamer internally wishes Blake would ask her to dance — friendly feeling "
                "not yet expressed (F1). Blake = 1MKA (individual male known adult)."
            ),
            "fri 1MKA→D 5>": (
                "Blake asks the dreamer to dance — initiative in requesting a shared activity (F5)."
            ),
            "fri D↔1MKA 6=": (
                "They dance together — mutual physical contact between equals (F6, direction = "
                "because both participate equally)."
            ),
            "fri 1MKA→D 6> (arm around)": (
                "Blake puts his arm around the dreamer while seated — physical friendly contact (F6)."
            ),
            "fri 1MKA→D 6> (bending close)": (
                "Blake bends toward the dreamer — another instance of F6 physical contact, "
                "temporally separated from the arm-around moment."
            ),
            "sex 1MKA→D 3> (first kiss)": (
                "Blake kisses the dreamer — romantic/passionate kiss (S3). Distinct from "
                "the F6 friendly physical contact because it is overtly romantic."
            ),
            "sex 1MKA→D 3> (Nate's kiss)": (
                "Nate (also 1MKA — same attribute code) throws his arms around and kisses "
                "the dreamer — romantic kiss (S3)."
            ),
            "NOT CODED — Jon embarrassing dreamer": (
                "Jon tried to make an impression and the dreamer felt embarrassed, but "
                "H/VdC aggression requires deliberate intent to harm or annoy. Jon's goal "
                "was to impress, not to embarrass — the embarrassment was incidental. "
                "Not coded as aggression."
            ),
        },
    },
    {
        "dream": (
            "I had a horrible dream. Howard was in a coffin. I yelled and screamed at his mom "
            "that it was all her fault. I kicked myself that I hadn't waited to become a widow "
            "rather than a divorcee in order to get the insurance. I woke up feeling miserable, "
            "the dream was so icky."
        ),
        "interactions": [
            {"init": "D",    "rec": "1FKA", "type": "agg", "code": "2>"},
            {"init": "D",    "rec": None,   "type": "agg", "code": "1*"},
        ],
        "reasoning": {
            "agg D→1FKA 2>": (
                "Dreamer yells and screams at Howard's mom — verbal aggression (A2), "
                "one-directional. Howard's mom = 1FKA (individual female known adult)."
            ),
            "agg D→SELF 1*": (
                "'I kicked myself' — self-directed internal hostility/self-blame (A1 covert, "
                "self-directed). No recipient; rec = null."
            ),
            "NOT CODED — Howard in coffin": (
                "Howard is dead in the dream (3MRA); he does not initiate or receive any "
                "interaction. Being in a coffin is a state, not a social interaction."
            ),
        },
    },
    {
        "dream": (
            'Paul McCartney and my "friends" were on one side of a room, and the Arabs were '
            'on the other. A chain link fence was between us but there were gaps in it. Paul '
            'started yelling insults and taunts at the Arabs. I said, "Stop it." I stood up '
            'and walked to him. The Arabs started yelling back and coming into "our" side of '
            'the room to fight. I was in the middle; I made them go back. Paul and the others '
            'started up again. I again tried to stop them. One Arab thanked me. Then, they all '
            'started again and I gave up.'
        ),
        "interactions": [
            {"init": "1MPA",          "rec": "2IEA",          "type": "agg", "code": "2>"},
            {"init": "D",             "rec": "1MPA",           "type": "agg", "code": "3>"},
            {"init": "2IEA",          "rec": "1MPA",           "type": "agg", "code": "2R"},
            {"init": "2IEA",          "rec": "1MPA",           "type": "agg", "code": "6>"},
            {"init": "D",             "rec": "2IEA",           "type": "agg", "code": "3>"},
            {"init": "1MPA + 2IUA",   "rec": "2IEA",           "type": "agg", "code": "2>"},
            {"init": "D",             "rec": "1MPA + 2IUA",    "type": "agg", "code": "3>"},
            {"init": "1MPA + 2IUA",   "rec": "2IEA",           "type": "agg", "code": "2="},
            {"init": "1IEA",          "rec": "D",              "type": "fri", "code": "2>"},
            {"init": "1MPA",          "rec": "D",              "type": "fri", "code": "5>"},
        ],
        "reasoning": {
            "agg 1MPA→2IEA 2>": (
                "Paul yells insults at the Arabs — verbal aggression (A2), one-directional. "
                "Paul McCartney = 1MPA (prominent). Arabs = 2IEA (group, ethnic identity)."
            ),
            "agg D→1MPA 3>": (
                "Dreamer says 'Stop it' to Paul — verbal coercion/demand directed at a "
                "specific character (A3), one-directional."
            ),
            "agg 2IEA→1MPA 2R": (
                "Arabs yell back — verbal aggression (A2) as a rejection/counterattack "
                "against Paul's prior aggression, coded R."
            ),
            "agg 2IEA→1MPA 6>": (
                "Arabs advance physically into Paul's side of the room to fight — "
                "physical coercion/encroachment (A6), one-directional."
            ),
            "agg D→2IEA 3>": (
                "Dreamer makes the Arabs go back — verbal coercion (A3), one-directional."
            ),
            "agg 1MPA+2IUA→2IEA 2>": (
                "Paul and friends start up verbal aggression again (A2); "
                "grouped as 1MPA + 2IUA acting together."
            ),
            "agg D→1MPA+2IUA 3>": (
                "Dreamer again tries to stop them — verbal coercion (A3) directed at "
                "Paul and friends grouped."
            ),
            "agg 1MPA+2IUA↔2IEA 2=": (
                "Everyone starts up again simultaneously — mutual verbal aggression (A2, =)."
            ),
            "fri 1IEA→D 2>": (
                "One Arab thanks the dreamer — verbal friendly expression (F2). "
                "Coded as 1IEA (individual, since one Arab acted alone)."
            ),
            "fri 1MPA→D 5>": (
                "Paul and friends restart aggression, but note: Paul earlier implicitly "
                "positioned the dreamer as an ally on his 'side' — coded F5 (initiative "
                "in shared social activity, being on the same side)."
            ),
        },
    },
    {
        "dream": (
            "A dinosaur. I am on its back. A large beautiful snake, green and shiny, with gold "
            "threads, is its driver. I must hold on to her head just behind the jaw to keep "
            "from getting bit and to control where to go. She tries to bite me. I have to be "
            "very strong to keep her under control. I am struggling hard and I almost can't do "
            "it. She turns to look at me and says, \"'Little girls' shouldn't (or couldn't) "
            "do this.\""
        ),
        "interactions": [
            {"init": "1ANI", "rec": "D",    "type": "agg", "code": "7>"},
            {"init": "D",    "rec": "1ANI", "type": "agg", "code": "6R"},
        ],
        "reasoning": {
            "agg 1ANI→D 7>": (
                "The snake tries to bite the dreamer — physical violence (A7), one-directional. "
                "Snake = 1ANI (individual animal; animals use ANI notation regardless of gender "
                "attributes shown in the dream)."
            ),
            "agg D→1ANI 6R": (
                "Dreamer grabs the snake's jaw and physically restrains it to prevent being bitten "
                "— physical coercion (A6) as a counterattack/rejection of the snake's aggression. "
                "Coded R because it is a response to 1ANI's prior attack, not an independent act."
            ),
            "NOT CODED — snake's verbal taunt": (
                "The snake says 'Little girls shouldn't do this' — this could be A2 verbal "
                "aggression, but the remark is coded as part of the overall confrontation "
                "already captured above. If coding conservatively, add: "
                "{init: 1ANI, rec: D, type: agg, code: 2>}."
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

    social = codings[codings["coding_type"].isin(["agg", "fri", "sex"])].copy()

    def rows_to_interactions(g):
        result = []
        for _, r in g.iterrows():
            rec = r["rec"]
            result.append({
                "init": str(r["init"]) if pd.notna(r["init"]) else None,
                "rec":  str(rec)        if pd.notna(rec)       else None,
                "type": r["coding_type"],
                "code": str(r["code"])  if pd.notna(r["code"]) else None,
            })
        return result

    ground_truth = (
        social.groupby("dream_id")
        .apply(rows_to_interactions)
        .to_dict()
    )
    return dreams, ground_truth


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────────────────
def build_system_content():
    ex_text = ""
    for ex in FEW_SHOT:
        output = {"interactions": ex["interactions"], "reasoning": ex["reasoning"]}
        ex_text += f"\nDream:\n{ex['dream']}\n\n"
        ex_text += f"Output:\n{json.dumps(output, indent=2, ensure_ascii=False)}\n"
        ex_text += "\n---\n"

    return (
        CODEBOOK
        + "\n\n"
        + "## Worked Examples\n"
        + ex_text
        + "\n\nRespond with ONLY valid JSON matching the output format above.\n"
        + "If no social interactions occur in the dream, return: "
        + '{"interactions": [], "reasoning": {}}\n'
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(client, dream_text, system_content):
    """Call Claude and return (interactions_list, reasoning_dict)."""
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
                    "Code all social interactions in this dream report:\n\n"
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
    return parsed.get("interactions", []), parsed.get("reasoning", {})


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def _to_tuple(interaction):
    """Normalize an interaction dict to a hashable tuple for whole-tuple comparison."""
    rec = interaction.get("rec")
    return (
        str(interaction.get("init", "")).strip(),
        str(rec).strip() if rec is not None else "SELF",
        str(interaction.get("type", "")).strip(),
        str(interaction.get("code", "")).strip(),
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


def decompose_interaction(interaction):
    """Flatten an interaction dict into a list of (slot, value) tuples.

    Splits init/rec character codes into their attribute slots, and the
    interaction code (e.g. '5>') into separate subtype + direction.
    """
    items = []
    items.append(("type", str(interaction.get("type", "")).strip()))

    code = str(interaction.get("code", "")).strip()
    if code and code[-1] in (">", "=", "R", "*"):
        items.append(("subtype",   code[:-1]))
        items.append(("direction", code[-1]))
    elif code:
        items.append(("code_raw", code))

    for role in ("init", "rec"):
        val = interaction.get(role)
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
    """Counter F1 over decomposed (slot, value) tuples — gives partial credit."""
    pred = Counter(item for i in predicted    for item in decompose_interaction(i))
    true = Counter(item for i in ground_truth for item in decompose_interaction(i))
    tp = sum((pred & true).values())
    precision = tp / sum(pred.values()) if pred else (1.0 if not true else 0.0)
    recall    = tp / sum(true.values()) if true else (1.0 if not pred else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    pred_tuples = Counter(_to_tuple(i) for i in predicted)
    true_tuples = Counter(_to_tuple(i) for i in ground_truth)

    return {
        "exact_match": pred_tuples == true_tuples,
        "precision":   round(precision, 3),
        "recall":      round(recall,    3),
        "f1":          round(f1,        3),
    }


def evaluate_by_type(predicted, ground_truth, itype):
    pred_t = [i for i in predicted    if i.get("type") == itype]
    true_t = [i for i in ground_truth if i.get("type") == itype]
    return evaluate_attribute(pred_t, true_t)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="H/VdC Social Interactions Coder")
    parser.add_argument("--all",        action="store_true",
                        help="Run on the full dataset")
    parser.add_argument("--collection", type=str, default=None,
                        help="Restrict to one collection (e.g. b-baseline, norms-f)")
    parser.add_argument("--dream-id",   type=str, default=None,
                        help="Run on a single dream by ID")
    parser.add_argument("--n",          type=int, default=None,
                        help="Override sample size (default: SAMPLE_SIZE setting)")
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
        sample = dreams[dreams["collection_id"] == args.collection].head(n)
    else:
        sample = dreams[dreams["collection_id"] == SAMPLE_COLLECTION].head(n)

    print(f"Processing {len(sample)} dream(s) with model {MODEL}...\n")

    results = []

    for _, row in sample.iterrows():
        dream_id   = row["dream_id"]
        dream_text = row["dream_report"]

        if pd.isna(dream_text):
            print(f"  SKIP  {dream_id}  (missing report text)")
            continue

        gt_interactions = ground_truth.get(dream_id, [])

        try:
            pred_interactions, reasoning = call_claude(
                client, str(dream_text), system_content
            )
        except Exception as e:
            print(f"  ERROR {dream_id}: {e}")
            results.append({
                "dream_id":     dream_id,
                "predicted":    "[]",
                "ground_truth": json.dumps(gt_interactions),
                "exact_match":  False,
                "precision":    None,
                "recall":       None,
                "f1_attr":      None,
                "f1_agg":       None,
                "f1_fri":       None,
                "f1_sex":       None,
                "reasoning":    f"ERROR: {e}",
                "dream_report": str(dream_text)[:300],
            })
            time.sleep(DELAY_SECONDS)
            continue

        metrics     = evaluate_attribute(pred_interactions, gt_interactions)
        metrics_agg = evaluate_by_type(pred_interactions, gt_interactions, "agg")
        metrics_fri = evaluate_by_type(pred_interactions, gt_interactions, "fri")
        metrics_sex = evaluate_by_type(pred_interactions, gt_interactions, "sex")

        has_gt = len(gt_interactions) > 0
        status = "✓" if metrics["exact_match"] else "✗"
        marker = "" if has_gt else "  [no gt]"
        print(
            f"  {status}  {dream_id}"
            f"  pred={len(pred_interactions)}  gt={len(gt_interactions)}"
            f"  F1={metrics['f1']:.2f}{marker}"
        )

        results.append({
            "dream_id":     dream_id,
            "predicted":    json.dumps(pred_interactions, ensure_ascii=False),
            "ground_truth": json.dumps(gt_interactions,  ensure_ascii=False),
            "exact_match":  metrics["exact_match"],
            "precision":    metrics["precision"],
            "recall":       metrics["recall"],
            "f1_attr":      metrics["f1"],
            "f1_agg":       metrics_agg["f1"],
            "f1_fri":       metrics_fri["f1"],
            "f1_sex":       metrics_sex["f1"],
            "reasoning":    json.dumps(reasoning, ensure_ascii=False),
            "dream_report": str(dream_text)[:300],
        })

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
        print(f"  F1 agg (attr)  : {valid['f1_agg'].mean():.3f}")
        print(f"  F1 fri (attr)  : {valid['f1_fri'].mean():.3f}")
        print(f"  F1 sex (attr)  : {valid['f1_sex'].mean():.3f}")

        mismatches = valid[~valid["exact_match"]]
        if len(mismatches) > 0:
            print(f"\n{'─'*60}")
            print(f"Mismatches ({len(mismatches)}) — first 10 shown:\n")
            for _, r in mismatches.head(10).iterrows():
                pred_list = json.loads(r["predicted"])
                gt_list   = json.loads(r["ground_truth"])
                print(f"  {r['dream_id']}")
                print(f"    Predicted ({len(pred_list)})   : "
                      + ", ".join(f"{i['type']} {i['init']}→{i['rec']} {i['code']}"
                                  for i in pred_list[:5])
                      + ("..." if len(pred_list) > 5 else ""))
                print(f"    Ground truth ({len(gt_list)}): "
                      + ", ".join(f"{i['type']} {i['init']}→{i['rec']} {i['code']}"
                                  for i in gt_list[:5])
                      + ("..." if len(gt_list) > 5 else ""))
                print(f"    F1={r['f1_attr']:.2f}  P={r['precision']:.2f}  R={r['recall']:.2f}")
                print()


if __name__ == "__main__":
    main()
