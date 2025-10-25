import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import time

camera_frame = None
spell_image_frame = None
spell_name = "None"
top_text = "Top Label"
bottom_text = "Bottom Label"

def cv2_to_tk(frame):
    """Convert OpenCV BGR image to Tkinter PhotoImage."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return ImageTk.PhotoImage(img)

root = None

def create_tk_window():
    global root, camera_frame, spell_image_frame

    # Create tkinter window
    root = tk.Tk()
    root.title("Manus Magus")

    test_level = tk.Label(root, text="Test Label")
    test_level.pack(expand=True, fill=tk.BOTH)

    camera_frame = tk.Label(root)
    camera_frame.pack(side=tk.LEFT)

    spell_image_frame = tk.Label(root)
    spell_image_frame.pack(side=tk.RIGHT)

    root.mainloop()


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
