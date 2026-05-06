#!/usr/bin/env python3
"""
striving.py — Hall/Van de Castle Striving Coder

Semi-automated classifier for the Striving category of the H/VdC coding system.
Codes goal-directed behavior outcomes — which characters achieved a success and
which experienced a failure — in dream reports using Claude, then evaluates
against human-coded ground truth.

Usage:
    python striving.py                                       # sample of 50 norms-f dreams (dev partition)
    python striving.py --n 20                                # sample of 20 dreams
    python striving.py --all                                 # full dataset
    python striving.py --collection norms-f                  # one collection
    python striving.py --dream-id norms-f_0003               # single dream
    python striving.py --skip 50 --n 50                      # dreams 51–100 (validation partition)

Output:
    striving_results.csv — per-dream results with predicted and ground-truth
    character lists for successes and failures, plus F1 for each.

Notes on evaluation:
    Each striving code is a character (D, 1FKA, etc.). F1 is computed at the
    attribute level: each character code is decomposed into (slot, value) tuples
    — number, gender, identity, age — then scored via Counter intersection.
    This gives partial credit when the model identifies the right character type
    but mis-codes one attribute (e.g., known vs. stranger).

Methodology note:
    Striving has ground-truth codings in norms-f and norms-m (and sparsely in
    b-baseline and emma). We use norms-f dreams 1–50 as the development partition
    for iteration. norms-m is reserved as the untouched final test set.
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
OUTPUT_CSV        = "striving_results.csv"
SAMPLE_SIZE       = 50
SAMPLE_COLLECTION = "norms-f"
DELAY_SECONDS     = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# CODEBOOK
# ─────────────────────────────────────────────────────────────────────────────
CODEBOOK = """
Hall/Van de Castle (H/VdC) Striving Coding Rules
=================================================

## Overview

Code every instance in which a character (including the dreamer) engages in
goal-directed behavior with a definite outcome — either Success or Failure.

A striving event requires THREE elements:
  1. A GOAL: the character wants to accomplish something specific
  2. An EFFORT: the character actively tries to achieve that goal
  3. An OUTCOME: the goal was either achieved (Success) or not (Failure)

Code WHO was involved in each outcome: the dreamer (D) or other characters.
Striving is less common than activities. Most dreams have 0–2 striving events.

## The Dreamer
Use "D" for the dreamer.

## Character Codes
Use standard H/VdC character codes: NUMBER + GENDER + IDENTITY + AGE
  e.g., 1MKA (one known male adult), 1FKA (one known female adult),
        1FSA (one female stranger adult), 2MKA (two known male adults)
  Family codes where the relationship is explicitly stated: M (mother), F (father),
  H (husband), W (wife), B (brother/sibling), etc.
  Animals: 1ANI / 2ANI

─────────────────────────────────────────────────────────────────────────────
## SUCCESS

Code a Success for a character when all three elements are present and the
outcome is explicitly positive. Key language:
  - "I finally found it," "I managed to open it," "I got through," "I found what I
    was looking for," "I succeeded in," "we made it," "we got away"
  - An obstacle is overcome after explicitly-stated effort AND explicit success outcome
  - A desired outcome is explicitly attained: "I got the job," "she won," "he convinced her"

DO NOT code success just because a character arrived somewhere, completed a
routine activity, or things worked out in the end with no stated goal or effort.

─────────────────────────────────────────────────────────────────────────────
## FAILURE

Code a Failure for a character when all three elements are present and the
outcome is explicitly negative. Key language:
  - "I couldn't," "I tried but failed," "I couldn't find my way out," "it didn't work,"
    "I was unable to," "I failed to," "I gave up," "no matter how hard I tried"
  - Blocked, stopped, or prevented from reaching a clearly-stated goal
  - Dream paralysis: unable to move, run, speak, or call out when EXPLICITLY trying to

DO NOT code failure just because something bad happened (accident, injury) or
because a plan changed — the explicit TRY + explicit FAIL language must be present.

─────────────────────────────────────────────────────────────────────────────
## DO NOT CODE AS STRIVING

  – Goals not stated explicitly: if the dream does not contain explicit goal
    language ("I tried to," "I was trying to," "I wanted to," "I attempted to,"
    "I was striving to find/reach/do X"), do NOT infer a goal. Only code
    striving when the character is explicitly said to be trying.

  – Outcomes not stated explicitly: if the dream does not explicitly state the
    outcome ("I finally found it," "I managed to," "I couldn't," "I failed to,"
    "it didn't work," "I gave up"), do NOT code success or failure.
    Arrival at a destination, completion of a journey, or an activity with
    effort does NOT automatically imply striving.

  – Fleeing, chasing, running from a threat: being chased by a person, animal,
    or threat and running away is locomotion on foot (Activity M), NOT striving,
    UNLESS the dream explicitly says the character was "trying to escape/get away"
    AND gives an explicit outcome ("I finally got away" / "I couldn't escape").
    Simply being pursued and running, without explicit try-outcome language, is
    not a failure or success — it is activity.

  – Arriving somewhere after a difficult journey: getting to a destination, even
    via a slippery slope, long walk, or difficult path, is locomotion (M or L),
    NOT a striving success. The difficulty of a journey does not make it striving.
    Only code success when goal + effort + outcome are all explicitly stated.

  – Routine activities with no specific stated goal: walking, eating, riding,
    talking — these belong to the Activities module even if they involve effort.

  – Plans that change due to circumstances: "we realized there was no place for
    bikes, so we returned" — adjusting plans in response to circumstances is not
    a failure. No explicit goal was stated and explicitly failed.

  – Background mentions of convincing, achieving, or trying: "after finally
    convincing me," "he had been working to get the job" — these are background
    mentions of striving, not actively-enacted scenes. Only code when the
    striving event unfolds in the dream scene with explicit goal and outcome.

  – Accidents and events that just happen: "I fell into the water," "the bridge
    opened," "she was shot" — events happening to a character are not striving
    failures, as the character did not TRY and fail.

  – Reactive survival acts: "I swam to the surface after falling in" — reactive
    motor behavior in response to an accident is locomotion, not striving.

  – Emotional states: "I was afraid," "I was panicky," "I felt lost." These are
    emotions, not goal-directed behavior.

  – Partial completion coded as success: if a character makes progress toward a
    goal but does NOT fully achieve it, do NOT code success. The stated goal must
    be fully achieved. Example: searching for an object, finding it but then not
    completing the task = still a failure.

  – Defensive resistance coded as success: if another character tries to make the
    dreamer do something and the dreamer successfully resists, code the other
    character's FAILURE — do NOT separately code the dreamer's success. Only the
    initiating character's failed attempt is coded as striving.

  – Multiple attempts at the same goal coded as multiple failures: if a character
    makes two or more attempts at the same goal within one dream, code it as ONE
    failure (or success) — not separate entries for each attempt. Example: trying
    to explain oneself twice in the same argument = one fail=[D], not fail=[D, D].

  – Ambiguous outcomes: when you cannot clearly determine success or failure from
    the text, do NOT code striving for that event.

─────────────────────────────────────────────────────────────────────────────
## FREQUENCY GUIDANCE

Most dreams have 0–2 striving events. If you produce more than 3–4 striving
codings for a single dream, re-check whether each event truly has all three
elements: a clear GOAL + a stated EFFORT + an explicit OUTCOME.

─────────────────────────────────────────────────────────────────────────────
## OUTPUT FORMAT

Respond with ONLY valid JSON — no prose before or after — in this exact shape:

{
  "successes": ["D", "1FKA"],
  "failures": ["D"],
  "reasoning": {
    "D succ (found exit)":       "Dreamer searched for and found the exit — clear success.",
    "1FKA succ (completed task)": "Known female adult achieved her stated goal — success.",
    "D fail (couldn't call)":    "Dreamer tried to call but the phone wouldn't work — failure.",
    "NOT CODED — walking to store": "Routine locomotion, no specific goal or outcome stated."
  }
}

Field rules:
  - "successes": list of character codes who achieved a success. Use "D" for the
    dreamer; standard H/VdC codes for others. If multiple characters jointly
    achieved a success, list each separately (e.g., ["D", "1FTT"]).
  - "failures": list of character codes who experienced a failure. Same format.
  - Empty lists when that type has no codings: "successes": [], "failures": []
  - If there are no striving events at all: {"successes": [], "failures": [], "reasoning": {}}
  - Include reasoning for each coded event AND for any near-miss you explicitly
    decided NOT to code.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES  (drawn from norms-f dev partition: dreams 1–50)
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT = [
    # Example 1: Success — dreamer achieves shopping goal after searching
    {
        "dream": (
            "I dreamed that I was walking up Euclid Avenue on a shopping trip. I was "
            "going into the small specialty shops such as Quinn-Maah's, Milgrims, etc. "
            "looking for bargains in lingerie. In each store there was a counter with a "
            "large pile of pink slips with fancy lace trimming and in each store, after "
            "looking through the pile and disarranging it, I walked out. Finally I came "
            "to an unidentified shop operated by two elderly women. There I found what I "
            "was looking for at a much lower price and selected several yellow slips and "
            "a pair of gold hanging earrings. I was quite satisfied."
        ),
        "successes": ["D"],
        "failures":  [],
        "reasoning": {
            "D succ (found what she was looking for)": (
                "The dreamer had a specific goal — finding bargain lingerie — tried multiple "
                "stores (explicit effort), and eventually found it at a lower price (explicit "
                "positive outcome: 'I found what I was looking for'). All three elements present: "
                "goal + effort + success outcome."
            ),
            "NOT CODED — walking, looking through piles, walking out of stores": (
                "These are routine activities (locomotion, visual inspection, physical browsing). "
                "They belong to the Activities module. The striving code is for the overall "
                "shopping goal and its outcome, not each sub-step."
            ),
        },
    },
    # Example 2: Failure — dreamer tries but cannot perform in play
    {
        "dream": (
            "I had a part in a play, but put off learning the lines. I kept thinking about "
            "the part and kept putting it out of my mind. Finally I was backstage, surrounded "
            "by black velvet curtains. I couldn't find the opening in the curtains to get to "
            "the stage, and I was also panicky because I had not learned my part. Just at the "
            "moment when I was supposed to go onstage, my older sister went on instead and "
            "took my part."
        ),
        "successes": [],
        "failures":  ["D"],
        "reasoning": {
            "D fail (couldn't get to the stage / couldn't perform)": (
                "The dreamer has a clear goal (perform in the play), made effort to reach the "
                "stage (searching for the opening in the curtains), and explicitly failed: "
                "couldn't find the opening, hadn't learned the lines, and ultimately could not "
                "go on. The sister taking her part is the explicit negative outcome."
            ),
            "NOT CODED — 'I kept thinking about the part'": (
                "This is a cognitive activity (thinking, putting off), not a striving event. "
                "Only the backstage episode — with clear goal + effort + failure outcome — is "
                "coded as striving."
            ),
            "NOT CODED — 'I was panicky'": (
                "Emotional state. Not a goal-directed act; belongs to Emotions module."
            ),
        },
    },
    # Example 3: Joint success — dreamer and another character escape together
    {
        "dream": (
            "I was in water like a lake up to my waist and my sister (16) was with me. "
            "We were striving desperately to get away from a large Negro man, but the water "
            "was holding us back. I feared that he was going to rape us, but then I realized "
            "he would not."
        ),
        "successes": ["D", "1FTT"],
        "failures":  [],
        "reasoning": {
            "D succ + 1FTT succ (escaped the man)": (
                "Both the dreamer and her 16-year-old sister (coded 1FTT) have a clear shared "
                "goal (escape from the threatening man), stated effort ('striving desperately'), "
                "and a positive outcome — the threat is ultimately avoided ('I realized he would "
                "not'). Each character who participated in the joint success is listed separately."
            ),
            "NOT CODED — 'I feared'": (
                "Emotional state. Not a striving event."
            ),
        },
    },
    # Example 4: Failure — multiple attempts, all fail; plus a partial act that is NOT coded
    {
        "dream": (
            "My kid sister (15) and I were getting ready for bed in the room next to ours. "
            "I went into our room to get my pajamas and I noticed that the window shades were "
            "up. Our double room looked out on the porch roof and as I walked to the window, "
            "I had the feeling that someone was on the roof. I pulled the blind down, but a "
            "hand reached in the window and pushed it back up. A man was out on the roof "
            "grinning at me. I was angry and tried to call my father, but I couldn't. I went "
            "into his room and woke him up. I told him very incoherently what happened and he "
            "got up. I didn't want him to go in our room because he might get hurt, but he "
            "insisted. As he walked down the hall, I forced myself to wake up."
        ),
        "successes": [],
        "failures":  ["D"],
        "reasoning": {
            "D fail (tried to call father but couldn't)": (
                "The dreamer explicitly tried to call her father and explicitly failed: "
                "'tried to call my father, but I couldn't.' Clear goal + effort + failure outcome."
            ),
            "NOT CODED — 'I went into his room and woke him up'": (
                "The dreamer DID wake the father — this looks like a success. However, the "
                "human coders coded only fail=[D] for this dream. Walking to the father's room "
                "and waking him was incidental locomotion + a brief physical act, not a "
                "goal-directed striving event with its own narrative focus. The dreamer's "
                "PRIMARY stated goal was to call — which failed. Follow the same conservative "
                "standard: only code striving when the goal + effort + outcome are the clear "
                "narrative focus."
            ),
            "NOT CODED — 'I pulled the blind down' (blind pushed back up)": (
                "A minor reactive act (pulling the blind) with an immediate trivial reversal. "
                "This is a physical activity, not a striving event with a stated goal."
            ),
            "NOT CODED — 'I was angry'": (
                "Emotional state. Not a striving event."
            ),
        },
    },
    # Example 5: No striving — bike ride, accident, no goal-directed outcomes
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
        "successes": [],
        "failures":  [],
        "reasoning": {
            "NOT CODED — returning when no bike parking": (
                "They realized there was nowhere to put the bikes and turned around. This is "
                "a passive adjustment to circumstances, not a striving failure. No specific "
                "goal was stated and explicitly failed — they simply changed plans."
            ),
            "NOT CODED — falling into the water": (
                "An accident caused by the drawbridge opening. The characters did not TRY to "
                "fall — it just happened. Accidents are not striving failures."
            ),
            "NOT CODED — 'it took a lot of effort getting to the surface'": (
                "Swimming to the surface after accidentally falling in is a reactive physical "
                "act, not a stated goal that was set and pursued. There is no explicit 'I tried "
                "to reach the surface' + outcome language. The effort involved does not make "
                "it striving — there must be an explicit GOAL stated before the attempt."
            ),
        },
    },
    # Example 6: No striving — chase/escape with active fleeing, but no explicit try/outcome
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
        "successes": [],
        "failures":  [],
        "reasoning": {
            "NOT CODED — fleeing / running from the old man": (
                "The dreamer and the other girl run from the old man — this is locomotion on "
                "foot (Activity M), not striving. The dream never says 'I tried to escape' or "
                "uses any explicit goal + outcome language for their flight. Simply fleeing a "
                "threat, even actively and urgently, is not striving. The fact that they ended "
                "up safe does NOT make it a success — there was no explicitly stated escape goal."
            ),
            "NOT CODED — girl tripping the old man": (
                "The other girl extends her foot and trips the man — this is a physical act "
                "(Activity P). It looks like an intentional act, but the dream does not frame "
                "it as a goal ('she tried to trip him') with an explicit outcome. It is a "
                "single spontaneous physical act in the context of fleeing, coded as Activity P."
            ),
            "NOT CODED — 'I asked a girl to check the closet'": (
                "The dreamer asks the girl to check the closet — this is a directed verbal "
                "activity (V>), not a striving event. The checking is a brief act, not a "
                "prolonged goal-directed effort with a stated outcome."
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

    succ_df = codings[codings["coding_type"] == "succ"].copy()
    fail_df = codings[codings["coding_type"] == "fail"].copy()

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

    succ_gt = succ_df.groupby("dream_id").apply(rows_to_chars).to_dict()
    fail_gt  = fail_df.groupby("dream_id").apply(rows_to_chars).to_dict()
    return dreams, succ_gt, fail_gt


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────────────────
def build_system_content():
    ex_text = ""
    for ex in FEW_SHOT:
        output = {
            "successes": ex["successes"],
            "failures":  ex["failures"],
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
        + "If no striving events occur in the dream, return: "
        + '{"successes": [], "failures": [], "reasoning": {}}\n'
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(client, dream_text, system_content):
    """Call Claude and return (successes_list, failures_list, reasoning_dict)."""
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
                    "Code all striving events in this dream report:\n\n"
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
    return (
        parsed.get("successes", []),
        parsed.get("failures",  []),
        parsed.get("reasoning", {}),
    )


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
    # Family codes (H, W, M, F, B, etc.) and other short codes
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="H/VdC Striving Coder")
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

    dreams, succ_gt, fail_gt = load_data()

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

        gt_succ = succ_gt.get(dream_id, [])
        gt_fail = fail_gt.get(dream_id, [])

        try:
            pred_succ, pred_fail, reasoning = call_claude(
                client, str(dream_text), system_content
            )
        except Exception as e:
            print(f"  ERROR {dream_id}: {e}")
            results.append({
                "dream_id":        dream_id,
                "pred_succ":       "[]",
                "gt_succ":         json.dumps(gt_succ),
                "pred_fail":       "[]",
                "gt_fail":         json.dumps(gt_fail),
                "f1_succ":         None,
                "f1_fail":         None,
                "mean_f1":         None,
                "precision_succ":  None,
                "recall_succ":     None,
                "precision_fail":  None,
                "recall_fail":     None,
                "reasoning":       f"ERROR: {e}",
                "dream_report":    str(dream_text)[:300],
            })
            time.sleep(DELAY_SECONDS)
            continue

        m_succ = evaluate_attribute(pred_succ, gt_succ)
        m_fail = evaluate_attribute(pred_fail, gt_fail)
        mean_f1 = round((m_succ["f1"] + m_fail["f1"]) / 2, 3)

        has_gt  = len(gt_succ) > 0 or len(gt_fail) > 0
        marker  = "" if has_gt else "  [no gt]"
        print(
            f"  {'✓' if mean_f1 == 1.0 else '✗'}  {dream_id}"
            f"  succ_pred={pred_succ}  succ_gt={gt_succ}"
            f"  fail_pred={pred_fail}  fail_gt={gt_fail}"
            f"  F1_succ={m_succ['f1']:.2f}  F1_fail={m_fail['f1']:.2f}"
            f"  mean={mean_f1:.2f}{marker}"
        )

        results.append({
            "dream_id":        dream_id,
            "pred_succ":       json.dumps(pred_succ, ensure_ascii=False),
            "gt_succ":         json.dumps(gt_succ,   ensure_ascii=False),
            "pred_fail":       json.dumps(pred_fail, ensure_ascii=False),
            "gt_fail":         json.dumps(gt_fail,   ensure_ascii=False),
            "f1_succ":         m_succ["f1"],
            "f1_fail":         m_fail["f1"],
            "mean_f1":         mean_f1,
            "precision_succ":  m_succ["precision"],
            "recall_succ":     m_succ["recall"],
            "precision_fail":  m_fail["precision"],
            "recall_fail":     m_fail["recall"],
            "reasoning":       json.dumps(reasoning, ensure_ascii=False),
            "dream_report":    str(dream_text)[:300],
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
        print(f"Mean F1 (succ)   : {valid['f1_succ'].mean():.3f}")
        print(f"  Precision      : {valid['precision_succ'].mean():.3f}")
        print(f"  Recall         : {valid['recall_succ'].mean():.3f}")
        print(f"Mean F1 (fail)   : {valid['f1_fail'].mean():.3f}")
        print(f"  Precision      : {valid['precision_fail'].mean():.3f}")
        print(f"  Recall         : {valid['recall_fail'].mean():.3f}")
        print(f"Mean F1 (overall): {valid['mean_f1'].mean():.3f}")

        # Dreams with any GT striving coding
        has_gt = valid[(valid["gt_succ"] != "[]") | (valid["gt_fail"] != "[]")]
        if len(has_gt) > 0:
            print(f"\nDreams with GT codings: {len(has_gt)}")
            print(f"  F1 succ  : {has_gt['f1_succ'].mean():.3f}")
            print(f"  F1 fail  : {has_gt['f1_fail'].mean():.3f}")
            print(f"  Mean F1  : {has_gt['mean_f1'].mean():.3f}")


if __name__ == "__main__":
    main()
