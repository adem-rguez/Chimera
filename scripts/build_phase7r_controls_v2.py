"""Phase 7R control-set v2: a much larger, much more diverse set of control (non-decision) prompts,
to fix the false-trigger problem from oracle/p7r-run4-gen.jsonl (the Stage-A model triggers on 24/100
natural-eval control prompts, mostly ones with invented type names like "joke_type"/"tone"/
"conversation_style" -- a symptom of the training set having only 114 controls drawn from 30 topics x
12 near-identical templates, scripts/build_phase7r_hybrid_prompts.py).

Produces oracle/phase7r-controls-v2-prompts.jsonl: ~600 rows, shaped for scripts/gen_think_traces.py
input ({id, user, kind:"control", type:"control", gold_label:None, ...}; no "system" key).

Coverage, >=60 distinct templates across these families:
  - creative writing: jokes, poems, slogans/taglines, story openers
  - brainstorming: ideas, names, titles
  - rewrite/edit/tone-change
  - summarizing a short embedded passage
  - explaining a concept
  - how-to / step plans
  - comparisons / pros-and-cons
  - email/message drafting
  - translation
  - simple math word problems / arithmetic
  - trivia / factual questions (none of which are any of the five typed decisions)
  - coding snippets
  - open advice
  - ~25% "hard near-miss" controls: surface-level choose/classify/rank language that is NOT one of
    the five typed decisions (claim_handling, order_outcome, answer_type, kb_category, news_topic) --
    e.g. picking a tone/title/colour, ranking options, sentiment/language classification, "which
    phrase is better", yes/no opinions. None of these are answerable by the five typed decisions:
    no expense-claim policy approvals, no order-quantity-limit checks, no "what kind of answer does
    this question need", no encyclopedia-entity category, no news-section routing.

Disjointness, asserted programmatically:
  - zero overlap of exact `user` text with oracle/phase7r-natural-eval.jsonl (kind=="control") and
    with oracle/phase7r-hybrid-prompts.jsonl
  - the templates here are new paraphrasings, not reuses, of the eval's "Write a short joke about..."/
    "Brainstorm two ideas for handling..." wording (checked via substring heuristics below)

Usage:
    python scripts/build_phase7r_controls_v2.py
    python scripts/build_phase7r_controls_v2.py --selftest
"""
import argparse
import collections
import itertools
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_FILE = os.path.join(ROOT, "oracle", "phase7r-natural-eval.jsonl")
HYBRID_FILE = os.path.join(ROOT, "oracle", "phase7r-hybrid-prompts.jsonl")
OUT_PATH = os.path.join(ROOT, "oracle", "phase7r-controls-v2-prompts.jsonl")

SEED = 72619031
N_TOTAL = 600
N_NEAR_MISS = 150  # ~25% of 600

BANNED_SUBSTRINGS = ["write a short joke about", "brainstorm two ideas for handling"]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# shared topic/fact pools, varied lengths and subjects (not the eval's/hybrid's topic pools)
# ---------------------------------------------------------------------------

TOPICS_SHORT = [
    "a leaky garden hose", "a jammed stapler", "a wobbling ceiling fan", "a dead car battery",
    "a snoring housemate", "a cracked phone screen", "an overflowing inbox", "a burnt toast smell",
    "a missing TV remote", "a squeaky bike chain", "a flat soda", "a traffic jam on the bridge",
    "a melted ice cream cone", "a locked bike rack", "a dripping faucet", "a torn umbrella",
    "a slow elevator", "a forgotten password", "a crowded subway car", "an empty coffee pot",
    "a broken guitar string", "a tangled garden hose", "a scratched record", "a flickering streetlight",
    "a mismatched sock drawer", "a chipped coffee mug", "a stubborn jar lid", "a wilting houseplant",
    "a late pizza delivery", "a cold cup of tea", "a jammed vending machine", "a noisy upstairs neighbor",
    "a short attention span during meetings", "a crowded parking garage", "a stuck elevator button",
    "a dusty bookshelf", "a leaking kayak", "a rusty bicycle lock", "a foggy bathroom mirror",
    "a half-eaten sandwich",
]

CONCEPTS = [
    "how a rainbow forms", "why bread rises", "how a refrigerator keeps things cold",
    "what causes jet lag", "how noise-cancelling headphones work", "why the sky is blue",
    "how compound interest works", "what a firewall does on a computer network",
    "how vaccines train the immune system", "why ice floats on water",
    "how a combustion engine turns fuel into motion", "what causes ocean tides",
    "how photosynthesis works", "why metal feels colder than wood at the same temperature",
    "how GPS determines your location", "what a credit score measures",
    "how a thermostat regulates temperature", "why some clouds bring rain and others don't",
    "how a search engine ranks results", "what causes seasons to change",
]

PASSAGES = [
    "The town council voted Tuesday to extend the public library's weekend hours after months of "
    "requests from local parents and students. The new hours begin next month and add four extra "
    "hours on Saturdays.",
    "A small bakery on Fifth Street has started donating its unsold bread to the community shelter "
    "every evening. The owner says the idea came from noticing how much was thrown out each night.",
    "Researchers tracking a local wetland found that the number of nesting herons tripled over the "
    "past five years, which they credit to a ban on motorboats in that section of the river.",
    "The city's bus system will add two new routes next spring connecting the east-side neighborhoods "
    "to the downtown transit hub, cutting the average commute by roughly fifteen minutes.",
    "A community garden that started with six raised beds in 2019 now covers half an acre and "
    "supplies produce to three different food pantries each week during the growing season.",
    "After the old pedestrian bridge was closed for repairs, local cyclists organized a volunteer "
    "crew to clear and mark a temporary detour trail through the adjacent park.",
]

PRODUCTS = [
    "a reusable coffee cup", "a folding bike", "a desk lamp", "a hiking backpack",
    "a noise-cancelling headset", "a ceramic plant pot", "a running shoe", "a travel pillow",
    "a kitchen knife set", "a notebook app", "a board game", "a rain jacket",
]

EVENTS = [
    "a neighborhood block party", "a weekend coding workshop", "a charity bake sale",
    "a small-town music festival", "a company picnic", "a book club meetup",
    "a local 5k fundraiser run", "a pop-up art exhibit",
]

LANGS = ["French", "Spanish", "German", "Japanese", "Italian", "Portuguese", "Dutch", "Swedish"]
PHRASES_TO_TRANSLATE = [
    "Could you pass the salt, please?", "The meeting has been moved to three o'clock.",
    "I would like to order a large coffee.", "Thank you for your help this morning.",
    "Where is the nearest train station?", "This package needs to arrive by Friday.",
    "Please remember to lock the door.", "The weather looks nice for a walk today.",
]

SKILLS = [
    "changing a bicycle tire", "making a pour-over coffee", "packing a suitcase efficiently",
    "setting up a tent", "writing a cover letter", "learning to juggle three balls",
    "organizing a small closet", "starting a vegetable garden from seed",
    "backing up a laptop's files", "tying a bow tie",
]

CODE_TASKS = [
    "reverses a string", "checks whether a number is prime", "finds the maximum value in a list",
    "counts the vowels in a sentence", "removes duplicates from a list",
    "converts a temperature from Celsius to Fahrenheit", "sums the digits of an integer",
    "checks if a word is a palindrome",
]

ADVICE_TOPICS = [
    "staying motivated to exercise in winter", "saving money on weekly groceries",
    "keeping houseplants alive while traveling", "reducing screen time before bed",
    "making small talk at a work conference", "staying organized with a busy freelance schedule",
    "getting better sleep before an early flight", "learning a new language as an adult",
]


def mk_math_word_problems():
    rows = []
    rng = random.Random(SEED + 1)
    templates = [
        "A baker makes {a} loaves of bread each morning and sells {b} by noon. How many loaves are left?",
        "A train travels {a} miles in {b} hours. What is its average speed in miles per hour?",
        "A classroom has {a} students, and {b} of them bring their own lunch. How many buy lunch instead?",
        "A recipe calls for {a} cups of flour to make {b} cookies. How many cups of flour are needed "
        "for one cookie, as a fraction?",
        "A parking lot has {a} spaces. If {b} cars are already parked, how many spaces remain?",
        "A runner jogs {a} laps around a track before breakfast and {b} more laps after work. How many "
        "laps total did they run?",
        "What is {a} plus {b}?",
        "What is {a} times {b}?",
        "What is {a} minus {b}?",
        "If a book costs ${a} and is discounted by ${b}, what's the sale price?",
    ]
    for i, tpl in enumerate(templates):
        a = rng.randint(3, 48)
        b = rng.randint(1, a)
        rows.append(tpl.format(a=a, b=b))
    return rows


def mk_trivia():
    return [
        "What is the capital of New Zealand?",
        "Who wrote the novel 'Pride and Prejudice'?",
        "How many continents are there on Earth?",
        "What is the chemical symbol for gold?",
        "In which year did the first commercial internet service providers appear?",
        "What is the tallest mountain in Africa?",
        "How many strings does a standard violin have?",
        "What gas do plants absorb from the air during photosynthesis?",
        "Who painted the ceiling of the Sistine Chapel?",
        "What is the smallest prime number?",
        "Which ocean is the largest by surface area?",
        "What is the boiling point of water in Fahrenheit at sea level?",
        "How many bones are in the adult human body?",
        "What year did the Berlin Wall fall?",
        "What is the currency used in Japan?",
    ]


# ---------------------------------------------------------------------------
# normal controls: family -> list of (template-id-stub, callable rng,pool -> text)
# Every template takes {t} (or specific named slots) and is registered once in TEMPLATES below,
# which is iterated with the topic pool to build the combo space.
# ---------------------------------------------------------------------------

JOKE_TEMPLATES = [
    "Come up with a quick one-liner joke about {t}.",
    "Tell me a silly joke involving {t}.",
    "I need a pun about {t} for a greeting card.",
    "Write a knock-knock joke themed around {t}.",
]

POEM_TEMPLATES = [
    "Write a four-line poem about {t}.",
    "Compose a short haiku inspired by {t}.",
    "Write a rhyming couplet about {t}.",
]

SLOGAN_TEMPLATES = [
    "Come up with a catchy slogan for {p}.",
    "Write a tagline for a company that sells {p}.",
    "Suggest a punchy ad headline for {p}.",
]

STORY_OPENER_TEMPLATES = [
    "Write an opening sentence for a short story involving {t}.",
    "Give me a dramatic first line for a story about {t}.",
    "Start a mystery story with a scene about {t}.",
]

BRAINSTORM_IDEAS_TEMPLATES = [
    "List three creative ways to deal with {t}.",
    "Suggest a few options for handling {t} next time it happens.",
    "What are some fun ways to turn {t} into a good story later?",
]

BRAINSTORM_NAME_TEMPLATES = [
    "Suggest five possible names for a pet hamster.",
    "Come up with three name ideas for a new neighborhood coffee shop.",
    "Brainstorm a handful of team names for a weekend trivia league.",
    "Suggest a few baby name ideas that mean 'bright' or 'light' in different languages.",
]

BRAINSTORM_TITLE_TEMPLATES = [
    "Suggest a title for a blog post about {t}.",
    "What would be a good chapter title for a memoir section about {t}?",
]

REWRITE_TEMPLATES = [
    "Rewrite this sentence to sound more formal: \"I can't make it to the thing tonight, sorry.\"",
    "Make this sentence more concise: \"I was thinking that maybe we could possibly consider "
    "rescheduling the meeting to sometime later next week if that works for everyone.\"",
    "Rewrite this so it sounds friendlier: \"Your report was late again and that's not acceptable.\"",
    "Turn this casual note into a professional email opener: \"hey, quick thing, can we talk later?\"",
    "Simplify this sentence for a younger reader: \"The organization's quarterly fiscal performance "
    "exceeded analyst projections by a considerable margin.\"",
    "Rewrite this in a more enthusiastic tone: \"The event starts at 6pm. There will be food.\"",
]

SUMMARY_TEMPLATES = [
    "Summarize this passage in one sentence:\n\n\"{t}\"",
    "Give me a two-sentence summary of this:\n\n\"{t}\"",
    "What's the main point of this paragraph?\n\n\"{t}\"",
]

EXPLAIN_TEMPLATES = [
    "Explain {t} in simple terms a ten-year-old could understand.",
    "Give a short explanation of {t}.",
    "Can you explain {t} in two or three sentences?",
]

HOWTO_TEMPLATES = [
    "Give me a short step-by-step plan for {t}.",
    "What are the basic steps involved in {t}?",
    "Walk me through the process of {t} in a few steps.",
]

COMPARE_TEMPLATES = [
    "Compare the pros and cons of working from home versus working in an office.",
    "What are the tradeoffs between renting an apartment and buying a house?",
    "Compare electric cars and gasoline cars in terms of everyday maintenance.",
    "What's the difference between a hurricane and a typhoon?",
    "Compare the benefits of running versus swimming for cardio exercise.",
    "What are the pros and cons of a four-day work week?",
]

EMAIL_TEMPLATES = [
    "Draft a short email asking a coworker to review a document by Friday.",
    "Write a brief message to a landlord asking about a leaking radiator.",
    "Draft an email to a client letting them know a shipment will be a day late.",
    "Write a short note to a teacher asking for an extension on an assignment.",
    "Draft a message inviting a few neighbors to {e}.",
]

TRANSLATE_TEMPLATES = [
    "Translate this sentence into {lang}: \"{phrase}\"",
    "How would you say this in {lang}? \"{phrase}\"",
]

CODE_TEMPLATES = [
    "Write a short Python function that {c}.",
    "Give me a one-line Python snippet that {c}.",
]

ADVICE_TEMPLATES = [
    "What's one piece of advice for {t}?",
    "Do you have any tips for {t}?",
    "What would you suggest to someone struggling with {t}?",
]

REVIEW_TEMPLATES = [
    "Write a short, positive review for {p}.",
    "Draft a two-sentence customer review for {p}.",
]

# ---------------------------------------------------------------------------
# hard near-miss controls: look like choose/classify/rank tasks but are not any of the five typed
# decisions. Grouped by the surface pattern they're meant to probe.
# ---------------------------------------------------------------------------

NEAR_MISS_TONE_TITLE_COLOUR = [
    "Pick the better tone for a birthday card: 'warm and funny' or 'formal and elegant'? Explain briefly.",
    "Which title sounds catchier for a cooking blog: 'Simmer & Spice' or 'The Weeknight Kitchen'?",
    "Choose a paint colour for a small reading nook and say why.",
    "Which name do you like better for a sailboat: 'Second Wind' or 'Tailwind'?",
    "Pick a font style -- playful or minimal -- for a kids' birthday invitation, and explain your pick.",
    "Which greeting feels warmer for a holiday card: 'Warmest wishes' or 'Season's greetings'?",
    "Choose between a bold red or a soft teal for a startup's logo, and justify it.",
    "Which is the better dessert name for a menu: 'Midnight Mocha Torte' or 'Dark Chocolate Dream'?",
    "Pick a mood -- cozy or energetic -- for a playlist meant for a Sunday morning, and explain.",
    "Which sounds better as a podcast name: 'Coffee and Context' or 'The Long Story Short'?",
]

NEAR_MISS_RANK = [
    "Rank these three vacation spots by how relaxing they sound: a mountain cabin, a beach resort, "
    "a city food tour.",
    "Put these three breakfast options in order from healthiest to least healthy: oatmeal, pancakes, "
    "a smoothie.",
    "Rank these hobbies by how much they cost to get started: painting, skiing, reading.",
    "Order these three chores from most to least annoying: doing dishes, folding laundry, vacuuming.",
    "Rank these three movie genres by how likely you are to fall asleep during them: documentary, "
    "action, romance.",
]

NEAR_MISS_SENTIMENT_LANG = [
    "Is the sentiment of this review positive or negative? \"The soup was lukewarm and the service "
    "was painfully slow.\"",
    "What language is this sentence written in? \"Il fait beau aujourd'hui.\"",
    "Does this message sound happy, sad, or angry? \"I can't believe they cancelled the trip again.\"",
    "Is this tweet sarcastic or sincere? \"Wow, another Monday, my favorite day of the week.\"",
    "What language is this phrase: \"Guten Morgen, wie geht es dir?\"",
    "Is the tone of this comment friendly or hostile? \"Could you maybe try reading the instructions "
    "next time?\"",
    "Classify this short review as positive or negative: \"Fast delivery, but the box arrived crushed.\"",
    "What language is this greeting written in: \"Konnichiwa, ogenki desu ka?\"",
]

NEAR_MISS_WHICH_BETTER = [
    "Which of these two phrases reads more naturally: 'at this point in time' or 'now'?",
    "Which sentence is clearer: 'The report was reviewed by the team' or 'The team reviewed the "
    "report'?",
    "Which opening line grabs attention more: 'It was a dark and stormy night' or 'She never saw it "
    "coming'?",
    "Which phrase sounds more polite: 'Send it over' or 'Could you send that over, please?'",
    "Which word fits better in a formal letter: 'regarding' or 'about'?",
]

NEAR_MISS_YESNO_OPINION = [
    "Do you think pineapple belongs on pizza?",
    "Is it better to wake up early or sleep in on weekends, in your opinion?",
    "Should people tip at coffee shops where you just order at the counter?",
    "Is it worth upgrading to a bigger phone screen, in your opinion?",
    "Do you think four-day work weeks would actually help productivity?",
    "Is it better to read the book or watch the movie first?",
    "Should restaurants automatically add a service charge for parties of two?",
    "Is it rude to answer a text message days later?",
]

NEAR_MISS_CATEGORY_OTHER = [
    "What category of recipe is this: a dish made with rice, shrimp, and saffron?",
    "What genre would you file a novel about a detective solving crimes in Venice under?",
    "What kind of exercise is rock climbing -- strength, cardio, or both?",
    "What category of expense would a gym membership normally fall under in a personal budget?",
    "What type of cloud is a tall, anvil-shaped one usually associated with thunderstorms?",
    "What kind of plant is a basil -- herb, shrub, or vine?",
    "What category of music would a slow piano instrumental with no lyrics usually be filed under?",
    "What type of knot is best for tying down a kayak on a roof rack?",
]

NEAR_MISS_CHOOSE_ACTION = [
    "You have 20 minutes free before a meeting -- should you grab coffee or return a few emails? "
    "What would you pick?",
    "For a rainy Saturday, would you rather bake something or watch a movie? Pick one and say why.",
    "Should a birthday gift be a handwritten card or a small gift, if you had to pick just one?",
    "Given a choice between a window seat or an aisle seat on a long flight, which would you choose?",
    "If you had to choose only one kitchen appliance to keep, would you keep the toaster or the "
    "blender?",
]

# parametrized near-miss templates: generic enough to combine with the shared topic/product/event
# pools above (TOPICS_SHORT/PRODUCTS/EVENTS/CONCEPTS) while staying in the "looks like choosing or
# classifying, isn't one of the five typed decisions" family.

NEAR_MISS_TONE_PARAM_TEMPLATES = [
    "Pick a tone -- playful or formal -- for a short write-up about {t}, and explain your choice.",
    "Choose a one-word mood to describe {t}: cozy or energetic?",
    "What title would you give a short blog post recapping {t}?",
    "Pick a colour that best represents the feeling of {t}, and say why.",
    "Would a headline about {t} work better as a question or a statement? Pick one.",
]

NEAR_MISS_RANK_PARAM_TEMPLATES = [
    "Rank these three ways of dealing with {t} by how easy they'd be: ignoring it, fixing it "
    "yourself, asking someone else for help.",
    "Order these three reactions to {t} from calmest to most frustrated: a shrug, a sigh, a groan.",
]

NEAR_MISS_CATEGORY_PARAM_TEMPLATES = [
    "What category would {p} normally be filed under in a store's product catalog?",
    "What section of a shop would you expect to find {p} in?",
]

NEAR_MISS_CHOOSE_ACTION_PARAM_TEMPLATES = [
    "If you had to choose between skipping {e} entirely or just dropping by for an hour, which would "
    "you pick?",
    "Would you rather host {e} at home or somewhere else? Pick one and say why.",
]

NEAR_MISS_YESNO_PARAM_TEMPLATES = [
    "Do you think {t} is more annoying than it sounds, or not really a big deal?",
    "Is dealing with {t} worth complaining about, in your opinion?",
]


def _param_near_miss(templates, pool, slot="t"):
    out = []
    for tpl in templates:
        for item in pool:
            out.append(tpl.format(**{slot: item}))
    return out


NEAR_MISS_GROUPS = [
    ("nearmiss_tone_title_colour",
     NEAR_MISS_TONE_TITLE_COLOUR + _param_near_miss(NEAR_MISS_TONE_PARAM_TEMPLATES, TOPICS_SHORT)),
    ("nearmiss_rank",
     NEAR_MISS_RANK + _param_near_miss(NEAR_MISS_RANK_PARAM_TEMPLATES, TOPICS_SHORT)),
    ("nearmiss_sentiment_lang", NEAR_MISS_SENTIMENT_LANG),
    ("nearmiss_which_better", NEAR_MISS_WHICH_BETTER),
    ("nearmiss_yesno_opinion",
     NEAR_MISS_YESNO_OPINION + _param_near_miss(NEAR_MISS_YESNO_PARAM_TEMPLATES, TOPICS_SHORT)),
    ("nearmiss_category_other",
     NEAR_MISS_CATEGORY_OTHER + _param_near_miss(NEAR_MISS_CATEGORY_PARAM_TEMPLATES, PRODUCTS, slot="p")),
    ("nearmiss_choose_action",
     NEAR_MISS_CHOOSE_ACTION + _param_near_miss(NEAR_MISS_CHOOSE_ACTION_PARAM_TEMPLATES, EVENTS, slot="e")),
]


def build_normal_rows(rng, used_text):
    """Returns list of (family, template_key, text). template_key used for the dedup/overlap key."""
    out = []

    def add_combo(family, templates, pool, slot="t"):
        for tpl_i, tpl in enumerate(templates):
            for topic in pool:
                text = tpl.format(**{slot: topic})
                key = f"{family}:{tpl_i}:{topic}"
                out.append((family, key, text))

    add_combo("joke", JOKE_TEMPLATES, TOPICS_SHORT)
    add_combo("poem", POEM_TEMPLATES, TOPICS_SHORT)
    add_combo("slogan", SLOGAN_TEMPLATES, PRODUCTS, slot="p")
    add_combo("story_opener", STORY_OPENER_TEMPLATES, TOPICS_SHORT)
    add_combo("brainstorm_ideas", BRAINSTORM_IDEAS_TEMPLATES, TOPICS_SHORT)
    add_combo("brainstorm_title", BRAINSTORM_TITLE_TEMPLATES, TOPICS_SHORT)
    add_combo("explain", EXPLAIN_TEMPLATES, CONCEPTS)
    add_combo("howto", HOWTO_TEMPLATES, SKILLS)
    add_combo("advice", ADVICE_TEMPLATES, ADVICE_TOPICS)
    add_combo("review", REVIEW_TEMPLATES, PRODUCTS, slot="p")

    for tpl_i, tpl in enumerate(REWRITE_TEMPLATES):
        out.append(("rewrite", f"rewrite:{tpl_i}", tpl))
    for tpl_i, tpl in enumerate(COMPARE_TEMPLATES):
        out.append(("compare", f"compare:{tpl_i}", tpl))

    for name in BRAINSTORM_NAME_TEMPLATES:
        out.append(("brainstorm_name", f"brainstorm_name:{name}", name))

    for tpl_i, tpl in enumerate(SUMMARY_TEMPLATES):
        for p_i, passage in enumerate(PASSAGES):
            text = tpl.format(t=passage)
            out.append(("summary", f"summary:{tpl_i}:{p_i}", text))

    for tpl_i, tpl in enumerate(EMAIL_TEMPLATES):
        if "{e}" in tpl:
            for ev in EVENTS:
                out.append(("email", f"email:{tpl_i}:{ev}", tpl.format(e=ev)))
        else:
            out.append(("email", f"email:{tpl_i}", tpl))

    for tpl_i, tpl in enumerate(TRANSLATE_TEMPLATES):
        for phrase in PHRASES_TO_TRANSLATE:
            for lang in LANGS:
                text = tpl.format(lang=lang, phrase=phrase)
                out.append(("translate", f"translate:{tpl_i}:{phrase}:{lang}", text))

    for tpl_i, tpl in enumerate(CODE_TEMPLATES):
        for task in CODE_TASKS:
            out.append(("code", f"code:{tpl_i}:{task}", tpl.format(c=task)))

    for text in mk_math_word_problems():
        out.append(("math", f"math:{text}", text))

    for text in mk_trivia():
        out.append(("trivia", f"trivia:{text}", text))

    rng.shuffle(out)

    rows, seen_keys = [], set()
    for family, key, text in out:
        if text in used_text or key in seen_keys:
            continue
        lowered = text.lower()
        if any(b in lowered for b in BANNED_SUBSTRINGS):
            continue
        seen_keys.add(key)
        used_text.add(text)
        rows.append((family, text))
    return rows


def build_near_miss_rows(rng, used_text):
    rows = []
    for family, pool in NEAR_MISS_GROUPS:
        for text in pool:
            if text in used_text:
                continue
            used_text.add(text)
            rows.append((family, text))
    rng.shuffle(rows)
    return rows


def build(seed=SEED):
    rng = random.Random(seed)
    used_text = set()

    near_miss = build_near_miss_rows(rng, used_text)
    normal = build_normal_rows(rng, used_text)

    n_near_miss = min(N_NEAR_MISS, len(near_miss))
    n_normal = N_TOTAL - n_near_miss

    selected = near_miss[:n_near_miss] + normal[:n_normal]
    rng.shuffle(selected)

    rows = []
    for i, (family, text) in enumerate(selected):
        rows.append(dict(
            id=f"p7rc2-{i:03d}", user=text, kind="control", type="control",
            gold_label=None, options_with_descriptions=None, item_ref=None,
            label_surfaces={}, family=family))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def check_disjoint(rows, eval_rows, hybrid_rows):
    fails = []
    v2_text = {r["user"] for r in rows}

    eval_ctrl_text = {r["user"] for r in eval_rows if r.get("kind") == "control"}
    overlap_eval = v2_text & eval_ctrl_text
    if overlap_eval:
        fails.append(f"{len(overlap_eval)} exact user-text overlap with eval controls: "
                      f"{list(overlap_eval)[:5]}")

    hybrid_text = {r["user"] for r in hybrid_rows}
    overlap_hybrid = v2_text & hybrid_text
    if overlap_hybrid:
        fails.append(f"{len(overlap_hybrid)} exact user-text overlap with hybrid prompts: "
                      f"{list(overlap_hybrid)[:5]}")

    lowered = [r["user"].lower() for r in rows]
    for banned in BANNED_SUBSTRINGS:
        hits = [t for t in lowered if banned in t]
        if hits:
            fails.append(f"{len(hits)} rows reuse banned eval-template substring {banned!r}")

    ids = [r["id"] for r in rows]
    dup_ids = [k for k, v in collections.Counter(ids).items() if v > 1]
    if dup_ids:
        fails.append(f"duplicate ids within v2 set: {dup_ids[:10]}")

    dup_text = [k for k, v in collections.Counter(r["user"] for r in rows).items() if v > 1]
    if dup_text:
        fails.append(f"{len(dup_text)} duplicate user texts within the v2 set")

    return fails


def report(rows, fails):
    families = collections.Counter(r["family"] for r in rows)
    n_templates_estimate = (
        len(JOKE_TEMPLATES) + len(POEM_TEMPLATES) + len(SLOGAN_TEMPLATES) + len(STORY_OPENER_TEMPLATES)
        + len(BRAINSTORM_IDEAS_TEMPLATES) + len(BRAINSTORM_TITLE_TEMPLATES) + len(BRAINSTORM_NAME_TEMPLATES)
        + len(REWRITE_TEMPLATES) + len(SUMMARY_TEMPLATES) + len(EXPLAIN_TEMPLATES) + len(HOWTO_TEMPLATES)
        + len(COMPARE_TEMPLATES) + len(EMAIL_TEMPLATES) + len(TRANSLATE_TEMPLATES) + len(CODE_TEMPLATES)
        + len(ADVICE_TEMPLATES) + len(REVIEW_TEMPLATES)
        + len(NEAR_MISS_TONE_TITLE_COLOUR) + len(NEAR_MISS_RANK) + len(NEAR_MISS_SENTIMENT_LANG)
        + len(NEAR_MISS_WHICH_BETTER) + len(NEAR_MISS_YESNO_OPINION) + len(NEAR_MISS_CATEGORY_OTHER)
        + len(NEAR_MISS_CHOOSE_ACTION)
        + len(NEAR_MISS_TONE_PARAM_TEMPLATES) + len(NEAR_MISS_RANK_PARAM_TEMPLATES)
        + len(NEAR_MISS_CATEGORY_PARAM_TEMPLATES) + len(NEAR_MISS_CHOOSE_ACTION_PARAM_TEMPLATES)
        + len(NEAR_MISS_YESNO_PARAM_TEMPLATES)
    )
    print(f"controls v2: {len(rows)} rows")
    print(f"  family counts: {dict(sorted(families.items()))}")
    print(f"  distinct templates (incl. near-miss standalone prompts counted as templates): "
          f"{n_templates_estimate}")
    print(f"  overlap check vs {EVAL_FILE} (controls) and {HYBRID_FILE}: "
          f"{'FAILED' if fails else 'passed, zero overlap'}")
    if fails:
        for f in fails:
            print(f"  - {f}")
    print("  8 sample prompts:")
    rng = random.Random(SEED + 99)
    sample = rng.sample(rows, min(8, len(rows)))
    for r in sample:
        print(f"    [{r['id']}] family={r['family']}")
        print(f"      user: {r['user'][:160]!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    rows = build(a.seed)
    eval_rows = load_jsonl(EVAL_FILE)
    hybrid_rows = load_jsonl(HYBRID_FILE)
    fails = check_disjoint(rows, eval_rows, hybrid_rows)
    report(rows, fails)

    if a.selftest:
        return 1 if fails else 0

    if fails:
        print("NOT writing output: disjointness check failed")
        return 1

    write_jsonl(OUT_PATH, rows)
    print(f"-> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
