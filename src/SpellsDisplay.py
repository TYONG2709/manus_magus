from enum import Enum, auto, global_str
import cv2

spell_displayed = None
images = {
    "thumbsup": "../resources/images/thumbsup.jpg",
    "fireball": "../resources/images/fireball.jpg",
    "shield": "../resources/images/shield.jpeg", # JPEG
    "bind": "../resources/images/bind.webp", # WEBP
    "invalid": "../resources/images/invalid.png",
}
# global_spell_display_frame = cv2.imread(images["thumbsup"]) # Default frame is thumbs up image


class Spell(Enum):
    THUMBSUP = auto()
    FIREBALL = auto() # Open palm fingers apart like a claw, casting a fireball
    SHIELD = auto() # Open palm like a stop sign
    BIND = auto()  # Closed fist
    INVALID = auto()

def display_spell(spell: Spell):
    if spell in images:
        return cv2.imread(images[spell])
    else:
        return cv2.imread(images['invalid'])