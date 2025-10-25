from enum import Enum, auto, global_str
import cv2

spell_displayed = None
images ={
    "thumbsup": "../resources/images/thumbsup.jpg",
    "fireball": "../resources/images/fireball.jpg",
    "shield": "../resources/images/shield.jpg",
    "bind": "../resources/images/bind.jpg",
}
global_spell_display_frame = cv2.imread(images["thumbsup"]) # Default frame is thumbs up image


class Spell(Enum):
    THUMBSUP = auto()
    FIREBALL = auto() # Open palm fingers apart like a claw, casting a fireball
    SHIELD = auto() # Open palm like a stop sign
    BIND = auto()  # Closed fist
    INVALID = auto()

def display_spell(spell: Spell):
    global spell_displayed
    global global_spell_display_frame

    # If the current spell being displayed is the same as the next spell to be displayed, don't do anything
    if spell == spell_displayed:
        return
    # Assign the spell we want to display to spell_displayed
    spell_displayed = spell
    match spell:
        case Spell.THUMBSUP:
            print("Thumbsup")
            global_spell_display_frame = cv2.imread(images["thumbsup"])
        case Spell.FIREBALL:
            print("Fireball")
            global_spell_display_frame = cv2.imread(images["fireball"])
        case Spell.SHIELD:
            print("Shield")
            global_spell_display_frame = cv2.imread(images["shield"])
        case Spell.BIND:
            print("Bind")
            global_spell_display_frame = cv2.imread(images["bind"])
        case Spell.INVALID:
            print("Invalid hand gesture")
            return

    cv2.imshow("Spell", global_spell_display_frame)
    cv2.waitKey(1)