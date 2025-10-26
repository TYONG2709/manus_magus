import random
import time
from TkinterHelper import update_label

spells = [
    "thumbsup", "fireball", "shield", "bind"
]

casted = None

def startGame():
    global casted

    while True:
        intermission()
        game_round()
        casted = None


def intermission():
    countdown = 5
    cast_text = "Be prepared..."
    casted_spell_text = ""
    timer_text = str(countdown)

    while (countdown > 0):
        update_label(cast_text, casted_spell_text, timer_text)
        time.sleep(1)
        countdown -= 1

def game_round():
    global casted

    countdown = 10
    cast_target = spells[random.randrange(1, len(spells))]

    cast_text = "Cast the following spell: " + cast_target
    casted_spell_text = "I'm waiting for you to cast something... >_<"
    timer_text = str(countdown)
    update_label(cast_text, casted_spell_text, timer_text)

    while (casted != cast_target):
        time.sleep(0.05)
        casted = cast_target

        if (casted == None):
            casted_spell_text = "I'm waiting for you to cast something... >_<"
        elif (casted == "invalid"):
            casted_spell_text = "This doesn't seem to be a valid spell :("
        elif (casted != cast_target):
            casted_spell_text = "You casted " + casted_spell_text + ", that isn't what we wanted :/"
        else:
            casted_spell_text = "You casted " + casted_spell_text

        countdown -= -0.05

        timer_text = "You have " + str(countdown) + " more seconds!!!"
        update_label(cast_text, casted_spell_text, timer_text)

        if (countdown == 0):
            cast_text = "You ran out of time! Shame on you ):<"
            casted_spell_text = ""
            update_label(cast_text, casted_spell_text, timer_text)
            break



