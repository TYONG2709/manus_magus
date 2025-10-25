from enum import Enum, auto


class Spell(Enum):
    THUMBSUP = auto()
    FIREBALL = auto() # Open palm fingers apart like a claw, casting a fireball
    SHIELD = auto() # Open palm like a stop sign
    BIND = auto()  # Closed fist
    INVALID = auto()