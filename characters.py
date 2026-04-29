#!/usr/bin/env python3
"""
characters.py — Hall/Van de Castle Character Coder

Semi-automated classifier for the Characters category of the H/VdC coding system.
Uses Claude to identify and code all characters in dream reports, then evaluates
against human-coded ground truth.

Usage:
    python characters.py                               # sample of 50 b-baseline dreams
    python characters.py --n 15                        # sample of 15 dreams
    python characters.py --all                         # full dataset
    python characters.py --collection emma             # one collection
    python characters.py --collection emma --series-mode  # series mode: builds character registry
    python characters.py --dream-id b-baseline_0003    # single dream

Notes on evaluation:
    Two F1 scores are reported:
    - Overall F1: all character codes
    - Non-family F1: excludes family/relative codes (H,W,D,B,T,M,F,X,A,Y,C,I,R)
      which require biographical context not present in single dream texts.
    The non-family F1 is the primary metric for a generalizable single-dream tool.

Series mode:
    With --series-mode, the classifier maintains a character registry across dreams
    in a collection. The registry is populated ONLY from explicit relationship
    statements in the text (e.g., "my husband Frank"). Dreams are processed in
    order; the registry grows incrementally and is never inferred or assumed.
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
OUTPUT_CSV        = "characters_results.csv"
SAMPLE_SIZE       = 50           # dreams to process in default mode
SAMPLE_COLLECTION = "b-baseline" # collection to draw the default sample from
DELAY_SECONDS     = 0.3          # pause between API calls

# ─────────────────────────────────────────────────────────────────────────────
# CODEBOOK
# ─────────────────────────────────────────────────────────────────────────────
CODEBOOK = """
Hall/Van de Castle (H/VdC) Character Coding Rules
==================================================

## Who is Codeable
Code a character when they are:
  - Physically present in the dream
  - Heard or seen via communication (phone, radio, TV, pictures)
  - Mentioned anywhere in the report
  - Referenced to establish ownership or relationship (e.g., "my BOYFRIEND's car" → code boyfriend)
  - Partially present (e.g., "I held his HAND" where the hand's owner is clear)

Do NOT code:
  - Generic mass references: "everyone," "a crowd," "some people," "dogs in general"
  - Characters mentioned only to contrast ("not my boyfriend," "unlike my mother")
  - Implied but unnamed characters (a gun firing does not mean a shooter is coded)
  - Characters referenced only as a goal, destination, or future need
    (e.g., "we need to get him to a doctor" — the doctor is not present and
    is not codeable; "I thought about my mother" alone does not code the mother
    unless she appears or is described as present)

## The Dreamer (D)
The dreamer is NEVER included in the character code list. Use the symbol D when
describing interactions, but D must not appear as a character code in your output.

─────────────────────────────────────────────────────────────────────────────
## CODE FORMAT:  NUMBER  +  GENDER  +  IDENTITY  +  AGE
   Exception: animals use 1ANI / 2ANI (and 3ANI–8ANI for special cases)
─────────────────────────────────────────────────────────────────────────────

## NUMBER
  1 = Individual — a single, individually described character
  2 = Group — two or more not distinguished from one another
  3 = Individual dead — a single character who is dead in the dream
  4 = Group dead
  5 = Individual imaginary / fictional / supernatural / dream-within-dream
  6 = Group imaginary / fictional / supernatural
  7 = Metamorphosis — original form (before transformation)
  8 = Metamorphosis — changed form (after transformation)

## GENDER
  M = Male (explicitly male, or masculine pronouns he/him/his)
  F = Female (explicitly female, or feminine pronouns she/her/hers)
  I = Indeterminate — use for:
        • a single character whose gender is not stated and no pronouns are given
        • a small group whose gender composition is truly unknown
        • a character identified only by an occupational role
  J = Joint gender — use for groups that contain or are likely to contain
      both male and female members. Apply J to:
        • large social gatherings (crowds, party guests, wedding guests, audiences)
        • any group explicitly described as mixed-gender
        • any unnamed group of 3 or more people in a social, academic,
          or workplace setting — presume mixed gender unless stated otherwise
        • groups in a social setting large enough to presume mixed gender
      Do NOT use J for individual characters.
      I is for SMALL groups (typically 2 people) of genuinely unknown gender.

  CRITICAL: Do NOT use F or M for a group just because most members appear to
  be one gender. A bus, party, classroom, audience, or other social gathering
  with 3+ people is J unless every member is explicitly stated to be the same
  gender. "Mostly women," "mainly men," "a group of women and one man" → all J.

  GENDER RULES:
  - Pronouns always take priority. Use he/him → M, she/her → F.
  - For names without pronouns: use M or F only when the name is
    unambiguously gendered in contemporary English usage
    (e.g., Mabel, Susan, Mary, Karen, Ellie → F; Howard, Bob, Nate → M).
    For names that could be either gender (Reta, Shea, Kim, Pat), use I.
  - For groups: ask "Is this a large social gathering?" → J.
    "Is this a small group of unknown composition?" → I.
    "Several women / a group of men" → F or M respectively.

## IDENTITY (ordered from most to least familiar to the dreamer)

  STEP 1 — Check for family/relative first:
  Immediate family (use the SPECIFIC code, not K):
    F = Father
    M = Mother
    X = Both parents together (coded as one unit)
    B = Brother
    T = Sister
    H = Husband  ← ONLY for an actual spouse explicitly called "husband"
    W = Wife     ← ONLY for an actual spouse explicitly called "wife"
    D = Daughter
    C = Son
    I = Own infant
    A = Aunt or Uncle
    Y = Other immediate family member (e.g., dreamer's grandchild)

  CRITICAL for H and W: Use H/W ONLY when the text explicitly states
  a MARITAL relationship ("my husband," "her husband," "his wife").
  Do NOT use H/W for:
    • Fiancés or fiancées ("my fiance," "my fiancée") → use K
    • Boyfriends or girlfriends → use K
    • Romantic partners without explicit marriage → use K
    • Characters who seem romantic but aren't called husband/wife → use K
  If the dream says "my husband" but the person is actually a fiance
  or the marriage is only imaginary, follow what the ACTUAL relationship
  is — use K for unmarried partners.
  Other relatives:
    R = Other relative — grandparents, in-laws, cousins, step-relations,
        ex-spouses (coded R not H/W), and anyone described as a relation
        but not in the immediate list above.

  CRITICAL for R: Use R for ANY of the following relationships, even when
  the character is named or feels personally familiar to the dreamer:
    • ex-spouses ("my ex-husband," "Howard, who I divorced," a former
      spouse the dreamer is no longer married to)
    • parents-in-law, siblings-in-law ("my mother-in-law")
    • grandparents, great-grandparents, grandchildren
    • cousins, nieces, nephews
    • step-relatives, half-siblings
    • any character explicitly described as a relative but not in the
      immediate list above (F/M/X/B/T/H/W/D/C/I/A/Y)
  Do NOT code these as K — the family/relative relationship always takes
  priority over personal acquaintance.

  STEP 2 — If not family, work through this decision tree:

    O = Occupational — character is identified primarily by their
        professional vocation or service role, with no other familiarity.
        Professional vocations: "a doctor," "the minister/priest," "a nurse,"
        "a soldier," "the teacher," "a police officer," "a cashier,"
        "a shopkeeper," "a bagger," "customers in a store."
        → Use O even if the character is also a stranger.
        → If the occupational figure is ALSO a named acquaintance the
          dreamer personally knows, use K. ("My friend Tyler who is a
          doctor" → K, not O.)
        IMPORTANT: "The bride" and "the groom" at a wedding are NOT O —
        they are ceremony participants (strangers = S), not professionals.
        Clergy officiating the ceremony (minister, priest) ARE O.

    E = Ethnic/Racial/National — the character's racial, ethnic,
        national, or cultural-historical identity is the primary
        identifying feature. Use E whenever a character's group identity
        is their main descriptor, including:
        • named ethnic/national groups: "some Arabs," "Chinese girls,"
          "a Mexican man," "the French soldiers"
        • historical or indigenous peoples encountered in a time-travel
          or period dream: "the natives," "the clansmen," "cavemen,"
          "pre-historic people" — these represent a distinct cultural
          group and should be coded E, not S or U.

    P = Prominent — real people famous beyond the dreamer's personal
        circle: celebrities, politicians, historical figures, athletes.
        IMPORTANT: Real famous people (Paul McCartney, the President,
        Einstein) are ALWAYS P with number 1 or 2 — they are NOT
        imaginary (5/6) even when appearing in a dream.

    K = Known — a character with whom the dreamer has a PRIOR PERSONAL
        RELATIONSHIP. K does NOT require a name. Use K when:
          • Named acquaintances: "Jim," "Sarah," "my friend Bob"
          • Unnamed romantic partners: "my boyfriend," "the girl I go with,"
            "my fiance," "the guy I've been seeing"
          • Classmates or study-group members: "a boy in my class,"
            "the students in my lab," "a girl from school"
          • Coworkers the dreamer personally knows: "co-worker Josh,"
            "my old boss," "a man I had worked for"
          • Anyone described as personally known: "a neighbor," "a girl
            I know," "someone from my building"
        IMPORTANT: If an occupational figure is also described as a
        personal acquaintance, use K instead of O (e.g., "a doctor I
        know," "a man I had worked for," "co-worker [Name]").
        IMPORTANT: K is NOT limited to named characters — a prior
        personal relationship is sufficient.

    S = Stranger — use when context makes clear the dreamer has NO prior
        relationship with this character. Does NOT require the word
        "stranger." Examples: "a man sent to me," "a woman I met [for
        the first time]," unnamed participants at a public event the
        dreamer is attending (a wedding, a party), people in a foreign
        or unfamiliar setting where no prior connection is implied.
        Key test: is there ANY indication of a prior relationship?
        If yes → K. If possibly → U. If clearly no → S.

    U = Uncertain — use ONLY when you genuinely cannot determine whether
        the dreamer knows this character. Examples: "some people" with no
        context, an unnamed figure in a fully ambiguous setting.
        U is a last resort — prefer K (prior relationship implied),
        S (stranger context clear), or O (occupational role clear).
        Tiebreaker: prefer S over U when there is a clear TRANSACTIONAL or
        PUBLIC context — a store, wedding, restaurant, bus, party, public
        event, or someone else's home. Use U for genuinely vague mentions
        where no setting or context tells you anything: "some people,"
        "kids" mentioned without scene context, "a person" in an ambiguous
        space. The presence of a setting that implies stranger relationships
        → S; absence of any informative context → U.

  CRITICAL IDENTITY RULES:
  - Never use K for family members. Family always gets the specific code.
  - O takes priority over S/U when a role/profession is mentioned.
  - P is for real famous people — never use 5 (imaginary) for celebrities.
  - S = unnamed + no prior relationship implied. U = truly ambiguous.

## AGE
  A = Adult (default when no age cue is present)
  T = Teenager (ages 13–17; high school student; peer of teenage dreamer)
  C = Child (ages 1–12; referred to as a child; elementary school pupil)
  B = Baby (under 1 year; called infant or baby — not as a term of endearment)

─────────────────────────────────────────────────────────────────────────────
## ANIMALS
  1ANI = individual animal
  2ANI = group of animals
  Apply the same number modifiers for dead/imaginary/metamorphosis animals:
    e.g., 3ANI = individual dead animal, 5ANI = individual imaginary animal

  IMPORTANT: When an animal is described with ANY human role, kinship term,
  family label, or relationship word (e.g., "my father was a gorilla,"
  "Jay cat, a cousin of mine," "my sister dog," "the cat that's like
  family," "Mama cat"), code it as ANI regardless of the label. The
  character's physical form in the dream ALWAYS takes priority over any
  human role, kinship metaphor, or relationship word applied to it. An
  animal is never coded with a human identity slot (F/M/B/T/H/W/D/C/I/A/Y/R/K/S/U/O/E/P).

## CREATURES
  CZZ = creature that cannot be identified as either human or animal
  (Append number prefix: 1CZZ, 2CZZ, etc.)

─────────────────────────────────────────────────────────────────────────────
## SPECIAL RULES
  1. Code each character only ONCE per dream (no duplicates for same character).
  2. If a group has some individually named members, code those separately AND
     code the remaining unnamed members as a group.
  3. A character who dies during the dream is coded with number 3/4
     (dead), NOT as metamorphosis (7/8).
  4. Metamorphosis (7/8): use ONLY when a single character unambiguously
     transforms from one distinct form to another mid-dream (e.g., a man
     physically becomes a wolf). When in doubt, code as TWO SEPARATE
     CHARACTERS instead — one before and one after. Narrative phrases like
     "she changes into a well-dressed woman" or "he became someone else"
     should be coded as two separate individuals (e.g., 1FSA + 1FPA),
     NOT as 7+8. Most dreams do not contain true metamorphosis.
  5. Imaginary (5/6): use ONLY for characters who are clearly supernatural,
     fictional, or from an explicitly stated dream-within-a-dream. Strange
     or unusual characters in otherwise realistic dreams are NOT imaginary —
     code them with number 1 or 2 as normal. Most characters are 1 or 2.
     NEVER use 5/6 for real famous people — they are always P with number 1/2.
  6. Dreamer metamorphosis: if the DREAMER transforms (e.g., "I was a lion"),
     do NOT code that transformation. Only code metamorphosis (7/8) when
     a character OTHER than the dreamer explicitly transforms.
  7. When uncertain whether two mentions refer to the same character,
     code the first mentioned unless the dream clarifies they are different.
  8. Dreams that contain ONLY the dreamer have an empty character list [].
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
        "codes": ["1MKA", "1IKA", "1MKA", "2ISC", "1MKA"],
        "reasoning": {
            "1MKA (Jon)": "Individual (1), male pronoun 'He' (M), known acquaintance (K), adult (A).",
            "1IKA (Reta)": "Individual (1), no pronoun used — name alone is insufficient for gender assignment so I (I), known (K), adult (A).",
            "1MKA (Blake)": "Individual (1), male pronoun 'He' (M), known (K), adult (A).",
            "2ISC (some kids)": "Group of children not individually described (2), no gender stated (I), strangers (S), children (C).",
            "1MKA (Nate)": "Individual (1), male pronoun 'him/his' (M), known (K), adult (A)."
        }
    },
    {
        "dream": (
            "I had a horrible dream. Howard was in a coffin. I yelled and screamed at his mom "
            "that it was all her fault. I kicked myself that I hadn't waited to become a widow "
            "rather than a divorcee in order to get the insurance. I woke up feeling miserable, "
            "the dream was so icky."
        ),
        "codes": ["3MRA", "1FKA"],
        "reasoning": {
            "3MRA (Howard)": "Dead individual (3) — explicitly in a coffin. Male pronoun 'his' (M). "
                             "Other relative (R) — ex-husband coded as relative rather than H because "
                             "the dreamer is now a divorcee. Adult (A).",
            "1FKA (his mom)": "Individual (1), female pronoun 'her' (F), known acquaintance (K), adult (A)."
        }
    },
    {
        "dream": (
            "A man, a friend is doing laundry. Some electrical problem starts. He goes to the "
            "circuit breaker. As he touches it, he gets zapped. We pull him off. Now we need to "
            "get him to a doctor, but someone brings to our attention that he needs to have a "
            "check-up and update his insurance or they won't pay! We carry him there to get it "
            "worked out."
        ),
        "codes": ["1MUA", "1IUA"],
        "reasoning": {
            "1MUA (a man, a friend)": "Individual (1), male pronoun 'He/him' (M), uncertain (U) "
                                     "— described as 'a friend' but never named. K requires a "
                                     "NAME; 'a friend' alone → U. Adult (A).",
            "1IUA (someone)": "Individual (1), no gender stated → I, uncertain (U) — 'someone' "
                             "is vague and unidentifiable. Adult (A).",
            "NOT CODED — the doctor": "'We need to get him to a doctor' — the doctor is "
                                      "referenced only as a future destination/need. They are "
                                      "not physically present in the dream and are not codeable.",
        }
    },
    {
        "dream": (
            "A wedding. The guests were dressed up like for Halloween, Witches, etc. The bride "
            "came and stood at the alter. A male in drag, crying (fake), and exaggerating a "
            "coquettish female, came down the aisle. He was the 'groom.' Then after he got to "
            "the aisle, he took off his mask and was an usher. The real groom (male), came down "
            "the steps. The female minister in costume was replaced by a male, 'real' minister."
        ),
        "codes": ["2JUA", "1FSA", "1MSA", "1MSA", "1FOA", "1MOA"],
        "reasoning": {
            "2JUA (the guests)": "Group (2), large social gathering at a wedding presumed to "
                                 "contain both genders → J, uncertain because the dreamer "
                                 "doesn't know who they are → U, adult (A).",
            "1FSA (the bride)": "Individual (1), female (F), stranger (S) — the bride is a "
                                "ceremony PARTICIPANT, not a professional; use S not O. Adult (A).",
            "1MSA (male in drag / fake groom)": "Individual (1), male pronoun 'He' (M), "
                                               "stranger (S) — ceremony participant, not a "
                                               "professional. Adult (A).",
            "1MSA (the real groom)": "Individual (1), male (M), stranger (S) — same reasoning "
                                    "as above. Adult (A).",
            "1FOA (female minister)": "Individual (1), female (F), occupational (O) — clergy "
                                     "officiating a ceremony ARE professionals → O, not S. Adult (A).",
            "1MOA (male minister)": "Individual (1), male (M), occupational (O) — same "
                                   "reasoning. Adult (A).",
        }
    },
    {
        "dream": (
            'Paul McCartney and my "friends" were on one side of a room, and the Arabs were '
            'on the other. A chain link fence was between us but there were gaps in it. Paul '
            'started yelling insults and taunts at the Arabs. I said, "Stop it." I stood up '
            'and walked to him. The Arabs started yelling back and coming into "our" side of '
            'the room to fight. I was in the middle; I made them go back. Paul and the others '
            'started up again. I again tried to stop them. One Arab thanked me. Then, they all '
            'started again and I gave up. It was too big. My father said, "You can go somewhere '
            'remote, if you can handle traveling over the Texas desert."'
        ),
        "codes": ["1MPA", "2IUA", "2IEA", "1IEA", "1MFA"],
        "reasoning": {
            "1MPA (Paul McCartney)": "Individual (1), male pronoun 'he' (M), prominent real "
                                     "famous person (P) — real celebrities are ALWAYS P with "
                                     "number 1, never imaginary (5), adult (A).",
            "2IUA (my friends)": "Group (2), gender not specified → I, 'my friends' as a "
                                 "vague group gives no info about which individuals → U "
                                 "(uncertain), adult (A). Note: even though they're the "
                                 "dreamer's friends, the group as a whole is U because we "
                                 "can't identify specific individuals.",
            "2IEA (the Arabs)": "Group (2), gender not specified → I, identified by "
                                "national/ethnic identity → E, adult (A).",
            "1IEA (one Arab)": "Individual (1), gender not stated → I, ethnic identity → E, "
                               "adult (A). Coded separately because he acted individually.",
            "1MFA (my father)": "Individual (1), male (M), father → use immediate family "
                                "code F (not K), adult (A). Code = 1MFA.",
        }
    },
    {
        "dream": (
            "As I was 'waking up' from a nap, Ginny and Rosemary were perched on my bed. "
            "Ginny was doing Math. I watched her count columns, carry and write numbers as she "
            "figured and I said, 'You do math like me.' She wasn't doing well. Then I got "
            "worried. I said, 'Oh Lord, I'm sure that I'm signed up for an intro to German "
            "class, but I forgot to go to any classes and it's late in the term. I'll flunk.' "
            "I started looking through my filing cabinet for the register form. I saw some B'day "
            "cards scrunched in there. Rosemary said, 'Take those out.' I had to see a boy "
            "standing on his head trying to say something but no one would listen. It was a B'day "
            "card to the devil. He had a grotesque red face, with huge reddish teeth. He was "
            "standing very large in a formal striped suit with a tie and carnation and a huge "
            "overcoat on. His coarse, snaky tail was draped around him. No fear."
        ),
        "codes": ["1FKA", "1FKA", "1MSC", "5MPA"],
        "reasoning": {
            "1FKA (Ginny)": "Individual (1), female pronoun 'her/she' (F), known (K), adult (A).",
            "1FKA (Rosemary)": "Individual (1), female (F — contextually female name used alongside Ginny "
                               "and consistent with the dreamer's social circle), known (K), adult (A).",
            "1MSC (a boy)": "Individual (1), male (M — 'a boy'), stranger (S), child (C).",
            "5MPA (the devil)": "Imaginary/supernatural individual (5), male pronoun 'He/His' (M), "
                                "prominent supernatural figure (P), adult (A)."
        }
    },
    {
        "dream": (
            "I'm in a bus with other people, mostly women. One woman says, \"Let's go live at "
            "the Kendell.\" I said, \"Oh, I've lived there before. I guess it would be O.K. as "
            "long as I don't have to live on the 4th floor. People who apply late usually get "
            "the 4th floor, but if there's an elevator, then it would be O.K.\" My friend then "
            "goes to the center of the bus and announces that I will play a song. I'm "
            "embarrassed. I get in front of everyone and need to readjust my instrument. I glue "
            "some cardboard together and cut some down to size. I use cottage cheese as a glue. "
            "I hide my work from the audience because they will have to guess the answer to the "
            "game. Then I try to play my banjo. It doesn't sound right. Finally, I notice that "
            "the cap is on incorrectly and the instrument is backwards and looks like a guitar. "
            "I look reprovingly at my friend and tell her she put the 4 on backwards for the "
            "caps. I fix it and then I can play the banjo."
        ),
        "codes": ["2JSA", "1FUA"],
        "reasoning": {
            "2JSA (other people on the bus)": (
                "Group (2), large social gathering on a bus with 3+ people → J even though "
                "'mostly women.' Do NOT use F just because most members appear female — the "
                "group is J unless every member is explicitly stated to be the same gender. "
                "No prior relationship with the dreamer implied → S (strangers). Adult (A)."
            ),
            "1FUA (one woman / my friend — same character)": (
                "Individual (1), female pronoun 'her/she' (F). "
                "SAME CHARACTER as 'one woman': the woman who speaks up on the bus ('Let's "
                "go live at the Kendell') is the same person the dreamer later calls 'my "
                "friend.' When two mentions could plausibly refer to the same character, "
                "code them as one — do not split into separate codes. "
                "Identity = U, not K: 'my friend' without a name is insufficient for K. "
                "K requires a named individual or a clearly described prior relationship. "
                "An unnamed friend in a dream defaults to U. Adult (A)."
            ),
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    dreams = pd.read_csv(INPUT_CSV)
    codings = pd.read_csv(CODINGS_CSV)
    chars = codings[codings["coding_type"] == "char"]
    ground_truth = chars.groupby("dream_id")["code"].apply(list).to_dict()
    return dreams, ground_truth


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────────────────
def build_system_content():
    """Return the system prompt as a string (cached at API level)."""
    ex_text = ""
    for ex in FEW_SHOT:
        ex_text += f"\nDream:\n{ex['dream']}\n\n"
        ex_text += f"Output:\n{json.dumps({'codes': ex['codes'], 'reasoning': ex['reasoning']}, indent=2)}\n"
        ex_text += "\n---\n"

    return (
        CODEBOOK
        + "\n\n"
        + "## Worked Examples\n"
        + ex_text
        + "\n\n"
        + "## Output Format\n"
        + "Respond with ONLY valid JSON — no prose before or after — in this exact shape:\n"
        + '{\n'
        + '  "codes": ["1MKA", "2ISC", ...],\n'
        + '  "reasoning": {"1MKA (character name)": "explanation", ...}\n'
        + '}\n\n'
        + "If the dream contains no codeable characters (dreamer only), return:\n"
        + '{"codes": [], "reasoning": {}}\n\n'
        + "Never include D (the dreamer) in the codes list.\n"
        + "\n"
        + "## Series Mode (when a Character Registry is provided)\n"
        + "If a 'Character Registry' section appears at the start of the user message:\n"
        + "- Use the listed name→code mappings when you recognize a character by name.\n"
        + "- Add a 'registry_updates' key to your JSON with any NEW explicit relationships\n"
        + "  found in this dream (e.g., 'my husband Frank' → {\"Frank\": \"1MHA\"}).\n"
        + "  Only include entries where the relationship is directly stated in the text.\n"
        + "  Do NOT infer, assume, or carry forward relationships.\n"
        + "- If no new explicit relationships appear, omit registry_updates or set it to {}.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(client, dream_text, system_content, registry=None):
    """Call Claude and return (codes_list, reasoning_dict, registry_updates)."""
    user_text = ""
    if registry:
        user_text = format_registry(registry) + "\n\n"
    user_text += f"Code the characters in this dream report:\n\n{dream_text}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": system_content,
                "cache_control": {"type": "ephemeral"},  # cache the large system prompt
            }
        ],
        messages=[
            {
                "role": "user",
                "content": user_text,
            }
        ],
    )

    raw = response.content[0].text.strip()

    # Extract JSON robustly — Opus sometimes adds prose before/after the block
    # Strategy 1: pull content from a ```json ... ``` fence
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            candidate = part.lstrip("json").strip()
            if candidate.startswith("{"):
                raw = candidate
                break

    # Strategy 2: find the outermost { ... } if no fence was found
    if not raw.startswith("{"):
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]

    parsed = json.loads(raw)
    return (
        parsed.get("codes", []),
        parsed.get("reasoning", {}),
        parsed.get("registry_updates", {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(predicted, ground_truth):
    """
    Compare predicted vs ground-truth character code lists using Counter
    intersection. Returns precision, recall, F1, and exact_match flag.
    """
    pred = Counter(predicted)
    true = Counter(ground_truth)

    tp = sum((pred & true).values())

    precision = tp / sum(pred.values()) if pred else (1.0 if not true else 0.0)
    recall    = tp / sum(true.values()) if true else (1.0 if not pred else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "exact_match": pred == true,
        "precision":   round(precision, 3),
        "recall":      round(recall,    3),
        "f1":          round(f1,        3),
    }


def attribute_accuracy(predicted, ground_truth):
    """
    Rough per-slot accuracy for valid 4-char codes (ignores ANI/CZZ).
    Matches predicted vs ground-truth positionally after sorting both lists.
    Meant as a diagnostic, not a primary metric.
    """
    def parse(code):
        code = str(code).strip()
        if len(code) == 4 and "ANI" not in code and "ZZ" not in code:
            return {"number": code[0], "gender": code[1],
                    "identity": code[2], "age": code[3]}
        return None

    pred_parsed = [p for p in (parse(c) for c in sorted(predicted)) if p]
    true_parsed = [p for p in (parse(c) for c in sorted(ground_truth)) if p]

    slots = ["number", "gender", "identity", "age"]
    counts = {s: {"correct": 0, "total": 0} for s in slots}

    for p, t in zip(pred_parsed, true_parsed):
        for s in slots:
            counts[s]["total"] += 1
            if p[s] == t[s]:
                counts[s]["correct"] += 1

    return {
        s: (counts[s]["correct"] / counts[s]["total"]
            if counts[s]["total"] > 0 else None)
        for s in slots
    }


# ─────────────────────────────────────────────────────────────────────────────
# FAMILY / NON-FAMILY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
FAMILY_IDENTITY_CODES = set("HWDBMTXFAYCIR")


def is_family_code(code):
    code = str(code).strip()
    return len(code) == 4 and code[2] in FAMILY_IDENTITY_CODES


def evaluate_nonfamily(predicted, ground_truth):
    pred_nf = [c for c in predicted    if not is_family_code(c)]
    true_nf = [c for c in ground_truth if not is_family_code(c)]
    return evaluate(pred_nf, true_nf)


def decompose_code(code):
    code = str(code).strip()
    if "ANI" in code:
        return [("num", code.replace("ANI", "")), ("type", "ANI")]
    if "CZZ" in code:
        return [("num", code.replace("CZZ", "")), ("type", "CZZ")]
    if len(code) == 4:
        return [("num", code[0]), ("gen", code[1]), ("idt", code[2]), ("age", code[3])]
    return [("raw", code)]


def evaluate_attribute(predicted, ground_truth):
    pred = Counter(attr for code in predicted    for attr in decompose_code(code))
    true = Counter(attr for code in ground_truth for attr in decompose_code(code))
    tp = sum((pred & true).values())
    precision = tp / sum(pred.values()) if pred else (1.0 if not true else 0.0)
    recall    = tp / sum(true.values()) if true else (1.0 if not pred else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "exact_match": Counter(predicted) == Counter(ground_truth),
        "precision":   round(precision, 3),
        "recall":      round(recall,    3),
        "f1":          round(f1,        3),
    }


def evaluate_nonfamily_attribute(predicted, ground_truth):
    pred_nf = [c for c in predicted    if not is_family_code(c)]
    true_nf = [c for c in ground_truth if not is_family_code(c)]
    return evaluate_attribute(pred_nf, true_nf)


def format_registry(registry):
    """Format the character registry dict for inclusion in the user message."""
    if not registry:
        return ""
    lines = ["Character Registry (explicit relationships from earlier dreams in this series):"]
    for name, code in registry.items():
        lines.append(f"  {name} → {code}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="H/VdC Character Coder")
    parser.add_argument("--all",        action="store_true",
                        help="Run on the full dataset")
    parser.add_argument("--collection", type=str, default=None,
                        help="Restrict to one collection (e.g. b-baseline, emma, ed)")
    parser.add_argument("--dream-id",   type=str, default=None,
                        help="Run on a single dream by ID")
    parser.add_argument("--n",          type=int, default=None,
                        help="Override sample size (default: SAMPLE_SIZE setting)")
    parser.add_argument("--series-mode", action="store_true",
                        help="Build character registry from explicit relationships across dreams")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)
    system_content = build_system_content()

    dreams, ground_truth = load_data()

    # ── Select dreams to process ──────────────────────────────────────────────
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
        # Default: first n dreams from SAMPLE_COLLECTION
        sample = (dreams[dreams["collection_id"] == SAMPLE_COLLECTION]
                  .head(n))

    print(f"Processing {len(sample)} dream(s) with model {MODEL}...\n")

    registry = {}
    results = []

    for _, row in sample.iterrows():
        dream_id   = row["dream_id"]
        dream_text = row["dream_report"]

        if pd.isna(dream_text):
            print(f"  SKIP  {dream_id}  (missing report text)")
            continue

        gt_codes = ground_truth.get(dream_id, [])

        try:
            registry_arg = registry if args.series_mode else None
            pred_codes, reasoning, registry_updates = call_claude(
                client, str(dream_text), system_content, registry_arg
            )
            if args.series_mode and registry_updates:
                registry.update(registry_updates)
        except Exception as e:
            print(f"  ERROR {dream_id}: {e}")
            results.append({
                "dream_id":              dream_id,
                "predicted":             "[]",
                "ground_truth":          json.dumps(gt_codes),
                "exact_match":           False,
                "precision":             None,
                "recall":                None,
                "f1_attr":               None,
                "f1_nonfamily_attr":     None,
                "precision_nonfamily":   None,
                "recall_nonfamily":      None,
                "exact_match_nonfamily": None,
                "reasoning":             f"ERROR: {e}",
                "dream_report":          str(dream_text)[:300],
            })
            time.sleep(DELAY_SECONDS)
            continue

        metrics    = evaluate_attribute(pred_codes, gt_codes)
        nf_metrics = evaluate_nonfamily_attribute(pred_codes, gt_codes)
        attr_acc   = attribute_accuracy(pred_codes, gt_codes)

        status = "✓" if metrics["exact_match"] else "✗"
        print(
            f"  {status}  {dream_id}"
            f"  pred={pred_codes}"
            f"  gt={gt_codes}"
            f"  F1={metrics['f1']:.2f}"
        )

        results.append({
            "dream_id":              dream_id,
            "predicted":             json.dumps(pred_codes),
            "ground_truth":          json.dumps(gt_codes),
            "exact_match":           metrics["exact_match"],
            "precision":             metrics["precision"],
            "recall":                metrics["recall"],
            "f1_attr":               metrics["f1"],
            "f1_nonfamily_attr":     nf_metrics["f1"],
            "precision_nonfamily":   nf_metrics["precision"],
            "recall_nonfamily":      nf_metrics["recall"],
            "exact_match_nonfamily": nf_metrics["exact_match"],
            "attr_number":           attr_acc["number"],
            "attr_gender":           attr_acc["gender"],
            "attr_identity":         attr_acc["identity"],
            "attr_age":              attr_acc["age"],
            "reasoning":             json.dumps(reasoning),
            "dream_report":          str(dream_text)[:300],
        })

        time.sleep(DELAY_SECONDS)

    # ── Save results ──────────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = df_out[df_out["f1_attr"].notna()]
    n = len(valid)

    print(f"\n{'─'*60}")
    print(f"Results saved → {OUTPUT_CSV}")
    print(f"Dreams evaluated : {n}")

    if n > 0:
        exact = valid["exact_match"].sum()
        print(f"Exact match      : {exact}/{n}  ({exact/n:.1%})")
        print(f"Mean F1 (attr)   : {valid['f1_attr'].mean():.3f}")
        print(f"Mean Precision   : {valid['precision'].mean():.3f}")
        print(f"Mean Recall      : {valid['recall'].mean():.3f}")

        nf_valid = valid[valid["f1_nonfamily_attr"].notna()]
        if len(nf_valid) > 0:
            nf_exact = nf_valid["exact_match_nonfamily"].sum()
            print(f"Non-family F1    : {nf_valid['f1_nonfamily_attr'].mean():.3f}  (excl. H/W/D/B/T/M/F/X/A/Y/C/I/R)")
            print(f"Non-family Exact : {nf_exact}/{len(nf_valid)}  ({nf_exact/len(nf_valid):.1%})")

        # Per-attribute accuracy (where computable)
        for attr in ["attr_number", "attr_gender", "attr_identity", "attr_age"]:
            col = valid[attr].dropna()
            if len(col) > 0:
                print(f"Attr {attr[5:]:10s}: {col.mean():.3f}  (n={len(col)})")

        # ── Mismatches for human review ───────────────────────────────────────
        mismatches = valid[~valid["exact_match"]]
        if len(mismatches) > 0:
            print(f"\n{'─'*60}")
            print(f"Mismatches ({len(mismatches)}) — first 10 shown for review:\n")
            for _, r in mismatches.head(10).iterrows():
                print(f"  {r['dream_id']}")
                print(f"    Predicted   : {r['predicted']}")
                print(f"    Ground truth: {r['ground_truth']}")
                print(f"    F1={r['f1_attr']:.2f}  P={r['precision']:.2f}  R={r['recall']:.2f}")
                print()


if __name__ == "__main__":
    main()
