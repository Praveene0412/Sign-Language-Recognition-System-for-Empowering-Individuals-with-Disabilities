import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import queue
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Text-to-Speech setup
engine = pyttsx3.init()
tts_queue = queue.Queue()

# TTS handler
def tts_handler():
    while True:
        text = tts_queue.get()
        if text is None:  # Stop condition
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

# Start TTS thread
tts_thread = threading.Thread(target=tts_handler, daemon=True)
tts_thread.start()

def speak_text(text):
    tts_queue.put(text)

# Label mapping
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '1', 27: '2',
    28: '3', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '10', 36: 'NewLabel'
}
expected_features = 42

# Buffers and history
stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""

# GUI Setup
root = tk.Tk()
root.title("Sign Language to Speech Conversion")
root.geometry("1300x650")
root.configure(bg="#2c2f33")
root.resizable(False, False)

# Variables for GUI
current_alphabet = StringVar(value="N/A")
current_word = StringVar(value="N/A")
current_sentence = StringVar(value="N/A")
is_paused = StringVar(value="False")

# Title
Label(root, text="Sign Language to Speech Conversion", font=("Arial", 28, "bold"), fg="#ffffff", bg="#2c2f33").grid(row=0, column=0, columnspan=2, pady=10)

# Layout Frames
video_frame = Frame(root, bg="#2c2f33", bd=5, relief="solid", width=500, height=400)
video_frame.grid(row=1, column=0, rowspan=3, padx=20, pady=20)
video_frame.grid_propagate(False)

content_frame = Frame(root, bg="#2c2f33")
content_frame.grid(row=1, column=1, sticky="n", padx=(20, 40), pady=(60, 20))

button_frame = Frame(root, bg="#2c2f33")
button_frame.grid(row=3, column=1, pady=(10, 20), padx=(10, 20), sticky="n")

# Video feed
video_label = tk.Label(video_frame)
video_label.pack(expand=True)

# Labels
Label(content_frame, text="Current Alphabet:", font=("Arial", 20), fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(0, 10))
Label(content_frame, textvariable=current_alphabet, font=("Arial", 24, "bold"), fg="#1abc9c", bg="#2c2f33").pack(anchor="center")

Label(content_frame, text="Current Word:", font=("Arial", 20), fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_word, font=("Arial", 20), fg="#f39c12", bg="#2c2f33", wraplength=500, justify="left").pack(anchor="center")

Label(content_frame, text="Current Sentence:", font=("Arial", 20), fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_sentence, font=("Arial", 20), fg="#9b59b6", bg="#2c2f33", wraplength=500, justify="left").pack(anchor="center")

# Reset sentence
def reset_sentence():
    global word_buffer, sentence
    word_buffer = ""
    sentence = ""
    current_word.set("N/A")
    current_sentence.set("N/A")
    current_alphabet.set("N/A")

def toggle_pause():
    if is_paused.get() == "False":
        is_paused.set("True")
        pause_button.config(text="Play")
    else:
        is_paused.set("False")
        pause_button.config(text="Pause")

# Add word to sentence function
def add_word_to_sentence():
    global word_buffer, sentence
    if word_buffer.strip():
        speak_text(word_buffer)
        sentence += word_buffer + " "
        current_sentence.set(sentence.strip())
        word_buffer = ""
        current_word.set("N/A")

# Buttons (modified for square layout: 2 per row)
Button(button_frame, text="Reset Sentence", font=("Arial", 16), command=reset_sentence, bg="#e74c3c", fg="#ffffff", relief="flat", height=2, width=14).grid(row=0, column=0, padx=10, pady=10)
pause_button = Button(button_frame, text="Pause", font=("Arial", 16), command=toggle_pause, bg="#3498db", fg="#ffffff", relief="flat", height=2, width=12)
pause_button.grid(row=0, column=1, padx=10, pady=10)

Button(button_frame, text="Add Word", font=("Arial", 16), command=add_word_to_sentence, bg="#2ecc71", fg="#ffffff", relief="flat", height=2, width=12).grid(row=1, column=0, padx=10, pady=10)
Button(button_frame, text="Speak Sentence", font=("Arial", 16), command=lambda: speak_text(current_sentence.get()), bg="#27ae60", fg="#ffffff", relief="flat", height=2, width=14).grid(row=1, column=1, padx=10, pady=10)

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

last_registered_time = time.time()
registration_delay = 1.5

def process_frame():
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time

    ret, frame = cap.read()
    if not ret:
        return

    if is_paused.get() == "True":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)
        root.after(10, process_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            if len(data_aux) < expected_features:
                data_aux.extend([0] * (expected_features - len(data_aux)))
            elif len(data_aux) > expected_features:
                data_aux = data_aux[:expected_features]

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            stabilization_buffer.append(predicted_character)
            if len(stabilization_buffer) > 30:
                stabilization_buffer.pop(0)

            if stabilization_buffer.count(predicted_character) > 25:  # Stabilization threshold
                current_time = time.time()
                if current_time - last_registered_time > registration_delay:
                    stable_char = predicted_character
                    last_registered_time = current_time
                    current_alphabet.set(stable_char)

                    if stable_char == ' ':
                        if word_buffer.strip():
                            speak_text(word_buffer)
                            sentence += word_buffer + " "
                            current_sentence.set(sentence.strip())
                        word_buffer = ""
                        current_word.set("N/A")
                    elif stable_char == '.':
                        if word_buffer.strip():
                            speak_text(word_buffer)
                            sentence += word_buffer + "."
                            current_sentence.set(sentence.strip())
                        word_buffer = ""
                        current_word.set("N/A")
                    else:
                        word_buffer += stable_char
                        current_word.set(word_buffer)

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    cv2.putText(frame, f"Alphabet: {current_alphabet.get()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, process_frame)

process_frame()
root.mainloop()

# Clean up resources
cap.release()
hands.close()
tts_queue.put(None)  # Stop TTS thread
tts_thread.join()
