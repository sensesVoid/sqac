"""Deterministic graded task generator for the thought-injection experiment.

Each task: a unit/temperature conversion word problem, its ground-truth
number, and the expected skill that should be routed to it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Task:
    question: str
    answer: float
    skill: str
    family: str


# Exact conversion factors shared with think_skills.yaml.
FACTORS = {
    ("mile", "km"): 1.609344,
    ("km", "mile"): 1 / 1.609344,
    ("foot", "m"): 0.3048,
    ("m", "foot"): 1 / 0.3048,
    ("inch", "cm"): 2.54,
    ("cm", "inch"): 1 / 2.54,
    ("lb", "kg"): 0.45359237,
    ("kg", "lb"): 1 / 0.45359237,
    ("oz", "g"): 28.349523125,
    ("g", "oz"): 1 / 28.349523125,
}

LEN_UNITS = {
    "mile": "miles",
    "km": "kilometers",
    "foot": "feet",
    "m": "meters",
    "inch": "inches",
    "cm": "centimeters",
}
MASS_UNITS = {"lb": "pounds", "kg": "kilograms", "oz": "ounces", "g": "grams"}
SMALLER = {("mile", "km"), ("foot", "m"), ("inch", "cm"), ("lb", "kg"), ("oz", "g")}


def _len_question(v: float, src: str, dst: str) -> str:
    a, b = LEN_UNITS[src], LEN_UNITS[dst]
    style = random.choice(
        [
            f"Convert {v} {a} to {b}.",
            f"How many {b} are in {v} {a}?",
            f"Change {v} {a} into {b}.",
            f"Express {v} {a} as {b}.",
        ]
    )
    return style


def _mass_question(v: float, src: str, dst: str) -> str:
    a, b = MASS_UNITS[src], MASS_UNITS[dst]
    style = random.choice(
        [
            f"Convert {v} {a} to {b}.",
            f"How many {b} is {v} {a}?",
            f"How many {b} are in {v} {a}?",
            f"Convert {v} {a} into {b}.",
        ]
    )
    return style


def _temp_question(v: float, c2f: bool) -> str:
    if c2f:
        return random.choice(
            [
                f"Convert {v:g} degrees Celsius to Fahrenheit.",
                f"What is {v:g} degrees C in Fahrenheit?",
                f"How hot is {v:g} celsius in fahrenheit?",
            ]
        )
    return random.choice(
        [
            f"Convert {v:g} degrees Fahrenheit to Celsius.",
            f"What is {v:g} degrees F in Celsius?",
            f"How many degrees C is {v:g} Fahrenheit?",
        ]
    )


def _rate_question(speed: float, hours: float) -> str:
    return random.choice(
        [
            f"How far do you travel at {speed:g} miles per hour for {hours:g} hours?",
            f"A car drives at {speed:g} miles per hour for {hours:g} hours. "
            "How many miles does it travel?",
            f"Traveling at {speed:g} mph for {hours:g} hours, how many miles do you cover?",
        ]
    )


def generate(seed: int = 7, per_family: int = 10) -> list[Task]:
    rng = random.Random(seed)
    random.seed(seed + 1)  # keep phrasing choices reproducible across runs
    tasks: list[Task] = []

    # length: 3 fixed pairs, random-ish values
    pairs = [
        ("mile", "km"), ("km", "mile"), ("foot", "m"),
        ("m", "foot"), ("inch", "cm"), ("cm", "inch"),
    ]
    for i in range(per_family):
        src, dst = rng.choice(pairs)
        v = round(rng.uniform(0.5, 500), 2)
        expect = v * FACTORS[(src, dst)]
        tasks.append(
            Task(
                question=_len_question(v, src, dst),
                answer=round(expect, 2),
                skill="length-conversion",
                family="length",
            )
        )

    for i in range(per_family):
        src, dst = rng.choice([("lb", "kg"), ("kg", "lb"), ("oz", "g")])
        v = round(rng.uniform(0.5, 400), 2)
        expect = v * FACTORS[(src, dst)]
        tasks.append(
            Task(
                question=_mass_question(v, src, dst),
                answer=round(expect, 2),
                skill="mass-conversion",
                family="mass",
            )
        )

    for i in range(per_family):
        c2f = rng.random() < 0.5
        v = round(rng.uniform(-40, 120), 1)
        expect = (v * 9 / 5 + 32) if c2f else ((v - 32) * 5 / 9)
        tasks.append(
            Task(
                question=_temp_question(v, c2f),
                answer=round(expect, 1),
                skill="temperature-conversion",
                family="temperature",
            )
        )

    for i in range(per_family):
        speed = round(rng.uniform(10, 90), 1)
        hours = round(rng.uniform(0.5, 12), 1)
        expect = speed * hours
        tasks.append(
            Task(
                question=_rate_question(speed, hours),
                answer=round(expect, 2),
                skill="rate-times-time",
                family="rate",
            )
        )

    return tasks


if __name__ == "__main__":
    ts = generate()
    print(f"{len(ts)} tasks")
    for t in ts[:6]:
        print(f"  [{t.skill}] {t.question}  ->  {t.answer}")