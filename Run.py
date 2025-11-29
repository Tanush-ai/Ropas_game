import cv2
import os
import time
import pygame
import json
import random
import numpy as np
import sys
import threading

# Optional imports – if unavailable we fall back gracefully
try:
    import pyttsx3
    TTS_ENGINE = pyttsx3.init()
except Exception:
    TTS_ENGINE = None
    print("[RPS] Warning: pyttsx3 not available – voice cues disabled")

try:
    import Hand_Detector
    detector = Hand_Detector.handDetector(detectionCon=0.75)
except Exception:
    detector = None
    print("[RPS] Warning: MediaPipe not found – hand detection disabled")

import Hand_Classifier
import RPSGame

# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------
pygame.init()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def img_path(name):
    return os.path.join(BASE_DIR, "images", name)

def music_path(name):
    return os.path.join(BASE_DIR, "music", name)

if "--wait" in sys.argv:
    try:
        input("[RPS] --wait given: press Enter to start")
    except Exception:
        pass

# Load background music (if any)
try:
    mpath = music_path("foo.wav")
    if os.path.exists(mpath):
        pygame.mixer.music.load(mpath)
        pygame.mixer.music.play(-1)
    else:
        print("[RPS] Warning: music/foo.wav not found")
except Exception as e:
    print("[RPS] Warning: failed to load music:", e)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_sounds():
    """Load click / win / lose / draw sounds if they exist."""
    sounds = {}
    base = os.path.join(BASE_DIR, "sounds")
    for name in ["click", "win", "lose", "draw"]:
        path = os.path.join(base, f"{name}.wav")
        if os.path.exists(path):
            try:
                sounds[name] = pygame.mixer.Sound(path)
            except Exception:
                pass
    return sounds

def play_sound(sounds, key):
    if key in sounds:
        sounds[key].play()

def speak(text):
    if TTS_ENGINE:
        TTS_ENGINE.say(text)
        TTS_ENGINE.runAndWait()

def load_score_data():
    score_file = os.path.join(BASE_DIR, "scores.json")
    if os.path.exists(score_file):
        try:
            with open(score_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"high_score": 0, "total_wins": 0, "total_losses": 0, "leaderboard": []}

def save_score_data(data):
    score_file = os.path.join(BASE_DIR, "scores.json")
    try:
        with open(score_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("[RPS] Failed to save scores:", e)

def update_leaderboard(data, player_score):
    lb = data.get("leaderboard", [])
    lb.append({"score": player_score, "time": time.time()})
    lb = sorted(lb, key=lambda x: (-x["score"], -x["time"]))[:5]
    data["leaderboard"] = lb

SOUNDS = load_sounds()

# Load persistent scores
SCORE_DATA = load_score_data()
player_score = SCORE_DATA.get("player_score", 0)
computer_score = SCORE_DATA.get("computer_score", 0)

# Camera discovery (robust for Windows)
wCam, hCam = 640, 480
cap = None
for i in range(3):
    print(f"[RPS] Testing camera {i}...")
    temp = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp.isOpened():
        ret, frame = temp.read()
        if ret and frame is not None and frame.size > 0:
            print(f"[RPS] Found working camera at index {i}")
            cap = temp
            break
        else:
            temp.release()
if cap is None:
    print("[RPS] Fallback: trying index 0 without DSHOW")
    cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
print("[RPS] Camera status:", cap.isOpened())
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

# Load overlay images (with placeholders if missing)
def placeholder(txt):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, txt, (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
    return img

rock_img = cv2.imread(img_path("Rock.jpeg"))
rock = rock_img if rock_img is not None else placeholder("ROCK")
paper_img = cv2.imread(img_path("Paper.jpeg"))
paper = paper_img if paper_img is not None else placeholder("PAPER")
scissor_img = cv2.imread(img_path("Scissor.jpeg"))
scissor = scissor_img if scissor_img is not None else placeholder("SCISSOR")

def scale_overlay(img, max_w=200, max_h=150):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

rock_s = scale_overlay(rock)
paper_s = scale_overlay(paper)
scissor_s = scale_overlay(scissor)
overlaylist = [scissor_s, paper_s, rock_s]  # 0:Scissor, 1:Paper, 2:Rock

# ROI settings
roi_size = 250
roi_x = 50
roi_y = 100

# Game configuration
DIFFICULTIES = ["easy", "medium", "hard"]
current_difficulty = "easy"
THEMES = {
    "dark": {
        "bg": (30, 30, 30),
        "roi": (0, 255, 0),
        "text": (255, 255, 255),
        "win": (0, 255, 0),
        "lose": (0, 0, 255),
        "draw": (255, 255, 0),
    },
    "light": {
        "bg": (220, 220, 220),
        "roi": (0, 150, 0),
        "text": (0, 0, 0),
        "win": (0, 150, 0),
        "lose": (150, 0, 0),
        "draw": (150, 150, 0),
    },
}
current_theme = "dark"

# State machine constants
STATE_WAITING = 0
STATE_COUNTDOWN = 1
STATE_RESULT = 2
STATE_TRAINING = 3
STATE_PAUSED = 4

current_state = STATE_WAITING
state_start_time = 0
countdown_duration = 2.0
result_duration = 1.5
status_text = "Press 'T' to Train or R/P/S to Play"

# Initialize classifier (will load saved model if present)
classifier = Hand_Classifier.HandClassifier()

pTime = 0

# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
while True:
    success, img = cap.read()
    if not success or img is None:
        print("[RPS] Warning: failed to read frame")
        time.sleep(0.1)
        continue
    img = cv2.flip(img, 1)

    # Apply theme background colour
    theme = THEMES[current_theme]
    # img[:] = theme["bg"]  # Commented out to keep camera feed visible

    # Draw ROI
    cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), theme["roi"], 2)
    roi_img = img[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]

    # Keyboard handling
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        play_sound(SOUNDS, "click")

    # Global shortcuts
    if key == ord('p'):
        if current_state != STATE_PAUSED:
            current_state = STATE_PAUSED
            status_text = "PAUSED – press 'p' to resume"
        else:
            current_state = STATE_WAITING
            status_text = "Press 'T' to Train or R/P/S to Play"
        continue
    if key == ord('c'):
        current_theme = "light" if current_theme == "dark" else "dark"
    if key == ord('d'):
        idx = DIFFICULTIES.index(current_difficulty)
        current_difficulty = DIFFICULTIES[(idx + 1) % len(DIFFICULTIES)]
        status_text = f"Difficulty: {current_difficulty.title()}"
    if key == ord('h'):
        help_start = time.time()
        while time.time() - help_start < 5:
            help_img = img.copy()
            lines = [
                "Controls:",
                "  T – toggle Training Mode",
                "  1/2/3 – add samples (Training)",
                "  SPACE – train model / lock move",
                "  R/P/S – manual play",
                "  D – change difficulty",
                "  C – toggle theme",
                "  H – show this help",
                "  P – pause/resume",
                "  Q – quit",
            ]
            for i, line in enumerate(lines):
                cv2.putText(help_img, line, (20, 30 + i * 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.imshow("Image", help_img)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        continue
    if key == ord('q'):
        break

    current_time = time.time()

    if current_state == STATE_TRAINING:
        cv2.putText(img, "TRAINING MODE", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "1:Rock 2:Paper 3:Scissor", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        cv2.putText(img, "SPACE: Train Model", (20, 110), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        sample_counts = f"Samples - R:{classifier.labels.count(1)} P:{classifier.labels.count(2)} S:{classifier.labels.count(3)}"
        cv2.putText(img, sample_counts, (20, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        if key == ord('1'):
            classifier.add_sample(roi_img, 1)
        elif key == ord('2'):
            classifier.add_sample(roi_img, 2)
        elif key == ord('3'):
            classifier.add_sample(roi_img, 3)
        elif key == ord(' '):
            if classifier.train():
                status_text = "Model Trained! Press T to Play"
                play_sound(SOUNDS, "win")
        continue

    if current_state == STATE_WAITING:
        cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, theme["text"], 2)
        cv2.putText(img, "Put hand in box", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        if classifier.is_trained:
            pred = classifier.predict(roi_img)
            pred_text = "?" if pred == 0 else ["", "Rock", "Paper", "Scissor"][pred]
            cv2.putText(img, f"Detected: {pred_text}", (roi_x, roi_y + roi_size + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
            if key == ord(' ') and pred != 0:
                player_choice = pred
                current_state = STATE_COUNTDOWN
                state_start_time = current_time
                speak("Go!")
        if key == ord('r'):
            player_choice = 1
            current_state = STATE_COUNTDOWN
            state_start_time = current_time
            speak("Rock")
        elif key == ord('p'):
            player_choice = 2
            current_state = STATE_COUNTDOWN
            state_start_time = current_time
            speak("Paper")
        elif key == ord('s'):
            player_choice = 3
            current_state = STATE_COUNTDOWN
            state_start_time = current_time
            speak("Scissor")
        continue

    if current_state == STATE_COUNTDOWN:
        elapsed = current_time - state_start_time
        remaining = countdown_duration - elapsed
        bar_w, bar_h = 400, 20
        bar_x = (wCam - bar_w) // 2
        bar_y = 400
        prog = min(1.0, elapsed / countdown_duration)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * prog), bar_y + bar_h), (0, 255, 255), -1)
        if remaining > 1:
            cv2.putText(img, "2", (280, 280), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 255), 10)
            if TTS_ENGINE:
                speak("Two")
        elif remaining > 0:
            cv2.putText(img, "1", (280, 280), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 255), 10)
            if TTS_ENGINE:
                speak("One")
        else:
            if classifier.is_trained:
                player_choice = classifier.predict(roi_img)
                if player_choice == 0:
                    player_choice = 1
            if current_difficulty == "easy":
                computer_choice = random.randint(1, 3)
            elif current_difficulty == "medium":
                beats = {1: 2, 2: 3, 3: 1}
                if random.random() < 0.6:
                    computer_choice = beats.get(player_choice, random.randint(1, 3))
                else:
                    computer_choice = random.randint(1, 3)
            else:  # hard
                if 'last_player' in globals() and last_player == player_choice:
                    counter = {1: 2, 2: 3, 3: 1}
                    computer_choice = counter[player_choice]
                else:
                    computer_choice = random.randint(1, 3)
            last_player = player_choice
            status, player_score, computer_score, computer_choice = RPSGame.Game(player_choice, player_score, computer_score)
            SCORE_DATA["player_score"] = player_score
            SCORE_DATA["computer_score"] = computer_score
            if "Player Wins" in status:
                SCORE_DATA["total_wins"] = SCORE_DATA.get("total_wins", 0) + 1
                play_sound(SOUNDS, "win")
            elif "Computer" in status:
                SCORE_DATA["total_losses"] = SCORE_DATA.get("total_losses", 0) + 1
                play_sound(SOUNDS, "lose")
            else:
                play_sound(SOUNDS, "draw")
            if player_score > SCORE_DATA.get("high_score", 0):
                SCORE_DATA["high_score"] = player_score
            update_leaderboard(SCORE_DATA, player_score)
            save_score_data(SCORE_DATA)
            status_text = status
            current_state = STATE_RESULT
            state_start_time = current_time
        continue

    if current_state == STATE_RESULT:
        elapsed = current_time - state_start_time
        idx = [0, 2, 1, 0][computer_choice]
        if overlaylist[idx] is not None:
            hO, wO, _ = overlaylist[idx].shape
            img[0:hO, wCam - wO:wCam] = overlaylist[idx]
        if "Player Wins" in status_text:
            col = theme["win"]
        elif "Computer" in status_text:
            col = theme["lose"]
        else:
            col = theme["draw"]
        cv2.putText(img, status_text, (100, 250), cv2.FONT_HERSHEY_COMPLEX, 2, col, 5)
        if elapsed > result_duration:
            current_state = STATE_WAITING
            player_choice = 0
            status_text = "Press 'T' to Train or R/P/S to Play"
        continue

    if current_state == STATE_PAUSED:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (wCam, hCam), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        lines = [
            "PAUSED",
            f"Difficulty: {current_difficulty.title()}",
            f"Theme: {current_theme.title()}",
            f"High Score: {SCORE_DATA.get('high_score',0)}",
            f"Wins: {SCORE_DATA.get('total_wins',0)}  Losses: {SCORE_DATA.get('total_losses',0)}",
            "Press 'p' to resume",
        ]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (20, 30 + i * 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        lb = SCORE_DATA.get("leaderboard", [])
        cv2.putText(img, "Leaderboard:", (wCam - 250, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        for i, entry in enumerate(lb):
            cv2.putText(img, f"{i+1}. {entry['score']}", (wCam - 250, 60 + i * 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.imshow("Image", img)
        continue

    # Draw scores and FPS (always visible)
    cv2.putText(img, f"Player: {player_score}", (430, 440), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f"Comp: {computer_score}", (30, 440), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (500, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image", img)

cap.release()
cv2.destroyAllWindows()