from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tkinter as tk
from keras.models import model_from_json, Sequential
import os
import pygame
from pygame import mixer
import tensorflow as tf

# Register the Sequential class
tf.keras.utils.get_custom_objects().update({'Sequential': Sequential})

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create model
json_file = open('models/emotion_model_0.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into new model
emotion_model.load_weights("models/emotion_model_0.h5")
print("Loaded model from disk")

# MUSIC
Angry = os.listdir("Songs/Angry")
Disgusted = os.listdir("Songs/Disgusted")
Fearful = os.listdir("Songs/Fearful")
Happy = os.listdir("Songs/Happy")
Neutral = os.listdir("Songs/Neutral")
Sad = os.listdir("Songs/Sad")
Surprised = os.listdir("Songs/Surprised")

temp = [Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised]
temp1 = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

count = 0
for i in temp:
    count1 = 0
    for j in i:
        i[count1] = "Songs/" + temp1[count] + "/" + j
        count1 += 1
    count += 1

playing = False
pause = False
total_duration = 0

LIMIT = 8
next = []
prev = []
current = ''
song = ""
stable_emotion = "Neutral"

mixer.init()

def format_time(seconds):
    # Convert the time in seconds to a formatted string (mm:ss)
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def update_progress(app):
    global total_duration
    # Get the current position of the music playback
    current_pos = pygame.mixer.music.get_pos() // 1000  # Convert to seconds

    # Calculate the percentage of progress
    progress_percent = (current_pos / total_duration) * 100

    # Update the progress bar
    app.progress_bar["value"] = progress_percent

    # Update the music timer label
    app.timer_label["text"] = format_time(current_pos)

    # Schedule the next update after a delay (in milliseconds)
    app.window.after(100, update_progress, app)

def on_music_end():
    # Function to be called when the music playback stops
    print("Music playback has ended")
    Next()

# Function to play music
def play_music(app):
    global current, song, next, LIMIT, prev, pause, total_duration, stable_emotion
    print("Prev:", prev, "Current:", current, "Next:", next)
    pygame.init()
    if pause or pygame.mixer.music.get_busy():
        print("Unpaused")
        mixer.music.unpause()
    else:
        print("Loaded new song")
        mixer.music.load(current)
        
        Text = "Current Song\nbased on :\n" + stable_emotion
        app.Current_song_emotion_label.config(text=Text)
    
        mixer.music.play()
        SONG = current.split('/')[-1]
        print("Playing song:", SONG)
        app.music_label.config(text=SONG)
    music = pygame.mixer.Sound(current)
    total_duration = music.get_length()
    update_progress(app)

    # Set the custom event for music end
    music_end_event = pygame.USEREVENT + 1
    pygame.mixer.music.set_endevent(music_end_event)

    # Bind the custom event to the on_music_end function
    pygame.event.set_allowed(music_end_event)
    pygame.event.set_blocked(pygame.USEREVENT)

def Next():
    print("Next song")
    mixer.music.unload()
    global current, song, next, LIMIT, prev, app, pause, playing, stable_emotion
    app.prev_button.config(state="active")
    queue(stable_emotion)
    if len(prev) > LIMIT:
        prev.pop(0)
    prev.append(current)    
    current = next.pop(0)
    app.progress_bar['value'] = 0
    pause = False
    playing = True  
    play_music(app) 

def Prev():
    print("Previous song")
    mixer.music.unload()
    global current, song, next, LIMIT, prev, app, pause, playing
    if len(next) > LIMIT:
        next.pop()
    next.insert(0, current)
    current = prev.pop()
    app.progress_bar['value'] = 0
    pause = False
    playing = True
    if len(prev) == 0:
        app.prev_button.config(state="disabled")
    play_music(app)

def queue(emotion):
    global current, song, next, LIMIT, prev, Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
    print("Queueing songs for emotion:", emotion)
    if emotion == "":
        song = Neutral.pop(0)
    elif emotion == "Angry":
        song = Angry.pop(0)
    elif emotion == "Disgusted":
        song = Disgusted.pop(0)
    elif emotion == "Fearful":
        song = Fearful.pop(0)
    elif emotion == "Happy":
        song = Happy.pop(0)
    elif emotion == "Neutral": 
        song = Neutral.pop(0)
    elif emotion == "Sad":
        song = Sad.pop(0)
    elif emotion == "Surprised":
        song = Surprised.pop(0) 
    if current == "":
        current = song
    else:
        next.append(song)
    print("Current song queued:", current)

# Tkinter GUI code
class App:
    def __init__(self, window, video_source=0):

        self.window = window
        self.window.title("Face Melody")
        self.window.configure(bg="#F7F7F7")

        self.emotion = "Neutral"

        # Disable full screen mode
        window.resizable(False, False) 

        # Set window icon
        self.window.iconbitmap('Images/icon.ico')  

        # Open video source (webcam)
        self.vid = cv2.VideoCapture(video_source)

        # Create a canvas for displaying video frames
        self.canvas = tk.Canvas(window, width=800, height=500, bg="#333333")
        self.canvas.pack()

        # Create a label for displaying the music name
        global current
        self.music_label = tk.Label(window, text=current, font=("Arial", 12), bg="#F7F7F7")
        self.music_label.pack(pady=10)

        # Create buttons for control (Play, Pause, Next)
        button_frame = tk.Frame(window, bg="#F7F7F7")
        button_frame.pack()

        self.prev_button = tk.Button(button_frame, text="⏮️ Previous", font=("Arial", 12), command=Prev, bg="#2196F3", fg="white", state="disabled")
        self.prev_button.pack(side=tk.LEFT, padx=10)
        self.play_button = tk.Button(button_frame, text="▶ Play", font=("Arial", 12), command=self.play, bg="#4CAF50", fg="white")
        self.play_button.pack(side=tk.LEFT, padx=10)
        self.pause_button = tk.Button(button_frame, text="|| Pause", font=("Arial", 12), command=self.pause, bg="#FF9800", fg="white", state="disabled")
        self.pause_button.pack(side=tk.LEFT, padx=10)
        self.next_button = tk.Button(button_frame, text="⏭ Next", font=("Arial", 12), command=Next, bg="#2196F3", fg="white")
        self.next_button.pack(side=tk.LEFT, padx=10)
        
        # Create a progress bar for music playback
        self.progress_bar = ttk.Progressbar(window, orient="horizontal", length=600, mode="determinate")
        self.progress_bar.pack(pady=5)

        # Create a music timer label
        self.timer_label = tk.Label(window, text="00:00")
        self.timer_label.pack()

        # Create a label for displaying detected emotion
        self.emotion_label = tk.Label(window, text="", font=("Arial", 16), bg="#F7F7F7")
        self.emotion_label.pack(ipadx=20, ipady=20, fill=tk.BOTH, expand=True, side=tk.RIGHT)

        # Create a label for displaying detected emotion
        global stable_emotion
        self.Current_song_emotion_label = tk.Label(window, text="", font=("Arial", 16), bg="#F7F7F7")
        self.Current_song_emotion_label.pack(ipadx=20, ipady=20, fill=tk.BOTH, expand=True, side=tk.RIGHT)

        # Create a label for displaying emotion logo
        self.logo_label = tk.Label(window, bg="#F7F7F7")
        self.logo_label.pack(side=tk.RIGHT, padx=10)

        self.emotion_counts = {"Angry":1, "Disgusted":1, "Fearful":1, "Happy":1, "Neutral":1, "Sad":1, "Surprised":1}

        self.delay = 10
        self.update()

        self.window.mainloop()

    def play(self):
        global playing, pause
        if not playing:
            play_music(self)
            self.play_button.config(state="disabled")
            self.pause_button.config(state="active")
            playing = True
            pause = False

    def pause(self):
        global playing, pause
        if playing:
            mixer.music.pause()
            self.play_button.config(state="active")
            self.pause_button.config(state="disabled")
            playing = False
            pause = True

    def update(self):
        istrue, frame = self.vid.read()
        if istrue:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
            # Predict emotion
            self.emotion = self.detect_emotion(frame)
            self.emotion_label.config(text=self.emotion)
            print(f"Detected emotion: {self.emotion}")  # Debug print
    
            # Display emotion logo
            self.update_emotion_logo()

            global current, song, next, LIMIT, prev, stable_emotion
            # Queue songs based on emotion if there's no song currently playing
            if not mixer.music.get_busy() and not pause:
                stable_emotion = self.emotion
                queue(stable_emotion)
                play_music(self)

        self.window.after(self.delay, self.update)

    def detect_emotion(self, frame):
        roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(roi_gray, 1.3, 5)
        emotion = "Neutral"
        print(f"Faces detected: {len(faces)}")  # Debug print

        for (x, y, w, h) in faces:
            roi_gray = roi_gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            emotion = emotion_dict[maxindex]
            self.emotion_counts[emotion] += 1
            print(f"Predicted emotion: {emotion}")  # Debug print
        
        return emotion

    def update_emotion_logo(self):
        global stable_emotion
        logo_path = {
            "Angry": "Images/angry.png",
            "Disgusted": "Images/disgusted.png",
            "Fearful": "Images/fearful.png",
            "Happy": "Images/happy.png",
            "Neutral": "Images/neutral.png",
            "Sad": "Images/sad.png",
            "Surprised": "Images/surprised.png"
        }
        
        if self.emotion in logo_path:
            logo_image = Image.open(logo_path[self.emotion])
            logo_image = logo_image.resize((100, 100), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            self.logo_label.config(image=self.logo_photo)
        else:
            self.logo_label.config(image='')

window = tk.Tk()
app = App(window)
