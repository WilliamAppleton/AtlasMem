"""A small ALFWorld-style household environment for exercising AtlasMem end to end.

The layout (which object lives in which receptacle) is fixed per ``layout_seed`` and is
shared across tasks, so experience from earlier episodes genuinely helps later ones:
an agent with memory can go straight to the right cabinet instead of searching.
"""

from __future__ import annotations

import random
import re

OPENABLE = ["fridge 1", "microwave 1", "cabinet 1", "cabinet 2", "cabinet 3", "cabinet 4",
            "drawer 1", "drawer 2", "drawer 3"]
OPEN_SURFACES = ["countertop 1", "countertop 2", "shelf 1", "diningtable 1", "stoveburner 1"]
APPLIANCES = {"sinkbasin 1": "clean", "microwave 1": "hot", "fridge 1": "cool"}
RECEPTACLES = OPEN_SURFACES + OPENABLE + ["sinkbasin 1"]
OBJECTS = ["mug", "apple", "tomato", "knife", "plate", "egg", "cup", "bowl", "spoon", "potato"]
# Where objects may be hidden (not appliances, so state changes stay deliberate).
HIDING = [r for r in RECEPTACLES if r not in ("sinkbasin 1", "microwave 1")]
TARGETS = ["countertop 1", "shelf 1", "diningtable 1", "cabinet 1", "drawer 1", "cabinet 4"]
STATE_WORD = {"clean": "clean", "hot": "hot", "cool": "cool"}

TASK_RE = re.compile(r"put an? (?:(clean|hot|cool) )?(\w+) (?:in|on) (?:the )?([a-z]+ \d)")

INSTRUCTIONS = """Available commands:
  look | inventory | go to <receptacle> | open <receptacle> | close <receptacle>
  take <object> from <receptacle> | put <object> in/on <receptacle>
  clean <object> with sinkbasin 1 | heat <object> with microwave 1 | cool <object> with fridge 1
Receptacles: """ + ", ".join(RECEPTACLES) + """
You can hold one object at a time. Closed receptacles must be opened before taking from or \
putting into them. You must be at a receptacle to interact with it."""


def _a(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def make_layout(layout_seed: int) -> dict[str, str]:
    rng = random.Random(layout_seed)
    return {obj: rng.choice(HIDING) for obj in OBJECTS}


def sample_tasks(n: int, seed: int = 0, layout_seed: int = 0) -> list[str]:
    """Tasks never target the receptacle the object already starts in."""
    rng = random.Random(seed)
    layout = make_layout(layout_seed)
    tasks = []
    while len(tasks) < n:
        obj = rng.choice(OBJECTS)
        target = rng.choice([t for t in TARGETS if t != layout[obj]])
        state = rng.choice([None, None, "clean", "hot", "cool"])
        art = "an" if (state or obj)[0] in "aeiou" else "a"
        desc = f"{state} {obj}" if state else obj
        tasks.append(f"put {art} {desc} in {target}")
    return tasks


class KitchenEnv:
    instructions = INSTRUCTIONS

    def __init__(self, layout_seed: int = 0, max_steps: int = 30):
        self.layout_seed = layout_seed
        self.max_steps = max_steps
        self.goal: tuple[str | None, str, str] | None = None

    def reset(self, task: str) -> str:
        m = TASK_RE.search(task.lower())
        if not m:
            raise ValueError(f"unrecognised kitchen task: {task!r}")
        self.goal = (m.group(1), m.group(2), m.group(3))
        layout = make_layout(self.layout_seed)
        self.contents: dict[str, list[str]] = {r: [] for r in RECEPTACLES}
        for obj, rec in layout.items():
            self.contents[rec].append(obj)
        self.states: dict[str, set[str]] = {o: set() for o in OBJECTS}
        self.opened: set[str] = set()
        self.location: str | None = None
        self.holding: str | None = None
        self.t = 0
        return ("You are in the middle of a kitchen. Around you are: "
                + ", ".join(RECEPTACLES) + ".")

    # -- helpers ---------------------------------------------------------------
    def _describe(self, rec: str) -> str:
        if rec in OPENABLE and rec not in self.opened:
            return f"The {rec} is closed."
        items = self.contents[rec]
        prep = "In" if rec in OPENABLE or rec == "sinkbasin 1" else "On"
        return f"{prep} the {rec}, you see " + (", ".join(_a(o) for o in items) or "nothing") + "."

    def _accessible(self, rec: str) -> bool:
        return rec not in OPENABLE or rec in self.opened

    def _success(self) -> bool:
        assert self.goal is not None
        state, obj, target = self.goal
        return obj in self.contents[target] and (state is None or state in self.states[obj])

    # -- step ------------------------------------------------------------------
    def step(self, action: str) -> tuple[str, bool, bool]:
        self.t += 1
        obs = self._apply(" ".join(action.lower().split()))
        success = self._success()
        done = success or self.t >= self.max_steps
        return obs, done, success

    def _apply(self, a: str) -> str:
        nothing = "Nothing happens."
        if a == "look":
            return f"You are at {self.location}. " + self._describe(self.location) if self.location \
                else "You are in the middle of the kitchen."
        if a == "inventory":
            return f"You are carrying: {_a(self.holding)}." if self.holding else "You are not carrying anything."
        if m := re.fullmatch(r"go to (?:the )?([a-z]+ \d)", a):
            rec = m.group(1)
            if rec not in self.contents:
                return nothing
            self.location = rec
            return f"You arrive at {rec}. " + self._describe(rec)
        if m := re.fullmatch(r"(open|close) (?:the )?([a-z]+ \d)", a):
            verb, rec = m.groups()
            if rec != self.location or rec not in OPENABLE:
                return nothing
            if verb == "open":
                self.opened.add(rec)
                return f"You open the {rec}. " + self._describe(rec)
            self.opened.discard(rec)
            return f"You close the {rec}."
        if m := re.fullmatch(r"take (?:the |a |an )?(\w+) from (?:the )?([a-z]+ \d)", a):
            obj, rec = m.groups()
            if (rec != self.location or self.holding or not self._accessible(rec)
                    or obj not in self.contents.get(rec, [])):
                return nothing
            self.contents[rec].remove(obj)
            self.holding = obj
            return f"You pick up the {obj} from the {rec}."
        if m := re.fullmatch(r"put (?:the |a |an )?(\w+) (?:in|on|in/on) (?:the )?([a-z]+ \d)", a):
            obj, rec = m.groups()
            if rec != self.location or self.holding != obj or not self._accessible(rec):
                return nothing
            self.contents[rec].append(obj)
            self.holding = None
            return f"You put the {obj} in/on the {rec}."
        if m := re.fullmatch(r"(clean|heat|cool) (?:the |a |an )?(\w+) with (?:the )?([a-z]+ \d)", a):
            verb, obj, rec = m.groups()
            want = {"clean": "sinkbasin 1", "heat": "microwave 1", "cool": "fridge 1"}[verb]
            if rec != want or self.location != rec or self.holding != obj:
                return nothing
            self.states[obj].add(APPLIANCES[rec])
            return f"You {verb} the {obj} using the {rec}."
        return nothing


def oracle_actions(task: str, layout_seed: int = 0) -> list[str]:
    """Shortest action script that solves a kitchen task (used by tests and judge evals)."""
    m = TASK_RE.search(task.lower())
    if not m:
        raise ValueError(f"unrecognised kitchen task: {task!r}")
    state, obj, target = m.groups()
    src = make_layout(layout_seed)[obj]
    acts = [f"go to {src}"]
    if src in OPENABLE:
        acts.append(f"open {src}")
    acts.append(f"take {obj} from {src}")
    if state:
        app = {"clean": "sinkbasin 1", "hot": "microwave 1", "cool": "fridge 1"}[state]
        verb = {"clean": "clean", "hot": "heat", "cool": "cool"}[state]
        acts += [f"go to {app}", f"{verb} {obj} with {app}"]
    acts.append(f"go to {target}")
    if target in OPENABLE:
        acts.append(f"open {target}")
    acts.append(f"put {obj} in {target}")
    return acts
