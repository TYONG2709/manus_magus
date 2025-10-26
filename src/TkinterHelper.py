import tkinter as tk
from PIL import Image, ImageTk
import cv2

camera_frame = None
spell_image_frame = None

title_label = None
cast_target_label = None
casted_spell_label = None
timer_label = None


def cv2_to_tk(frame):
    """Convert OpenCV BGR image to Tkinter PhotoImage."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return ImageTk.PhotoImage(img)

root = None

# Create the tkinter window, must be in main thread
def create_tk_window():
    global root, camera_frame, spell_image_frame, title_label, cast_target_label, casted_spell_label, timer_label

    # Create tkinter window
    root = tk.Tk()
    root.title("Manus Magus")

    title_label = tk.Label(root, text="Manus Magus", font=("Arial", 20, "bold"), pady=15)
    title_label.pack(expand=True, fill='x')

    cast_target_label = tk.Label(root, text="Cast the following spell: ...", font=("Arial", 25), pady=15)
    cast_target_label.pack(expand=True, fill='x')

    cv2_frame = tk.Frame(root)
    cv2_frame.pack(expand=True, fill='x')

    camera_frame = tk.Label(cv2_frame)
    camera_frame.pack(side=tk.LEFT)

    spell_image_frame = tk.Label(cv2_frame)
    spell_image_frame.pack(side=tk.RIGHT)

    casted_spell_label = tk.Label(root, text="You casted: ....", font=("Arial", 15), pady=5)
    casted_spell_label.pack(side=tk.BOTTOM, expand=True, fill='x')

    timer_label = tk.Label(root, text="You have <<TIMER>> seconds!", font=("Arial", 20, "bold"), pady=15)
    timer_label.pack(side=tk.BOTTOM, expand=True, fill='x')

    root.mainloop()


# Update the content on function invocation
def update_window(camera_frame_np, spell_image_frame_np):
    global camera_frame, spell_image_frame

    if camera_frame is not None:
        camera_image = cv2_to_tk(camera_frame_np)
        camera_frame.imgtk = camera_image
        camera_frame.configure(image=camera_image)

    if spell_image_frame is not None and spell_image_frame_np is not None:
        spell_image = cv2_to_tk(spell_image_frame_np)
        spell_image_frame.imgtk = spell_image
        spell_image_frame.configure(image=spell_image)

def update_label(cast_target_text, casted_spell_text, timer_text):
    global cast_target_label, casted_spell_label, timer_label

    if cast_target_label is not None and cast_target_text is not None:
        cast_target_label.config(text=cast_target_text)

    if casted_spell_label is not None and casted_spell_text is not None:
        casted_spell_label.config(text=casted_spell_text)

    if timer_label is not None and timer_text is not None:
        timer_label.config(text=timer_text)