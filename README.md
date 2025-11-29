# 🎮 ROPAS: AI-Powered Rock Paper Scissors

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Pygame](https://img.shields.io/badge/Pygame-Game%20Engine-red?style=for-the-badge&logo=pygame)

> A futuristic, interactive Rock-Paper-Scissors game that uses **Computer Vision** to detect your hand gestures in real-time. Play against an adaptive AI, train your own models, or enjoy a quick web-based battle!

---

## 📑 Table of Contents
- [✨ Features](#-features)
- [⚙️ Installation](#-installation)
- [🕹️ How to Play](#-how-to-play)
- [🧠 AI Training](#-ai-training)
- [📂 Project Structure](#-project-structure)
- [🔧 Troubleshooting](#-troubleshooting)

---

## ✨ Features

### 🤖 Advanced AI & Computer Vision
-   **Real-time Gesture Recognition:** Uses OpenCV and K-Nearest Neighbors (KNN) to instantly identify Rock, Paper, or Scissors gestures from your webcam.
-   **Adaptive Difficulty:**
    -   🟢 **Easy:** Random moves (classic RNG).
    -   🟡 **Medium:** Smart counter-moves.
    -   🔴 **Hard:** Adaptive AI that learns your playing patterns and predicts your next move.

### 🎮 Immersive Gameplay
-   **Voice Cues:** Integrated Text-to-Speech (TTS) for countdowns ("One, Two, Go!") and results.
-   **Visual Themes:** Switch between **Dark Mode** (Neon/Cyberpunk) and **Light Mode** (Clean/Minimalist).
-   **Leaderboard:** Tracks your high scores and win streaks locally.

### 🌐 Dual Modes
1.  **Python App:** The full experience with camera control and voice.
2.  **Web Game:** A polished, single-file HTML5 version (`game.html`) for quick click-based play.

---

## ⚙️ Installation

### Prerequisites
-   Python 3.8 or higher
-   A working webcam

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Tanush-ai/Ropas_game.git
    cd Ropas_game
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries: `opencv-python`, `numpy`, `pygame`, `pyttsx3`, `kagglehub`.*

---

## 🕹️ How to Play

### Option A: Python Camera Game
Run the main script to start the computer vision game:
```bash
python Run.py
```

**🎮 Controls:**
| Key | Action |
| :--- | :--- |
| **Space** | **Lock In Move** (during countdown) |
| **T** | Toggle **Training Mode** |
| **D** | Change **Difficulty** (Easy/Med/Hard) |
| **C** | Toggle **Theme** (Dark/Light) |
| **P** | **Pause** / Resume Game |
| **H** | Show **Help** / Controls |
| **Q** | **Quit** Game |
| **R / P / S** | Manual Play (Rock/Paper/Scissors) |

### Option B: Web Game
Simply double-click **`game.html`** to open it in your web browser. No installation required!

---

## 🧠 AI Training

The game comes with a pre-trained model (`model.xml`). You can retrain it to improve accuracy or adapt it to your specific lighting conditions.

### 1. Auto-Train (Recommended)
Download a massive dataset (2000+ images) and train automatically.
**Dataset:** [Rock Paper Scissors Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
```bash
# Step 1: Download dataset from Kaggle
python download_data.py

# Step 2: Train the model
python train_model.py
```

### 2. Manual In-Game Training
Teach the AI your specific hand gestures:
1.  Press **`T`** in-game to enter Training Mode.
2.  Position your hand for **Rock** and press **`1`** repeatedly to add samples.
3.  Repeat for **Paper (`2`)** and **Scissors (`3`)**.
4.  Press **`Space`** to train and save the new model.

---

## 📂 Project Structure

```text
Ropas/
├── Run.py              # 🚀 Main game entry point
├── Hand_Classifier.py  # 🧠 AI Model logic (KNN)
├── RPSGame.py          # ⚖️ Game logic (Win/Loss rules)
├── game.html           # 🌐 Standalone Web Version
├── train_model.py      # 🏋️ Script to batch train model
├── download_data.py    # 📥 Script to fetch Kaggle dataset
├── requirements.txt    # 📦 Python dependencies
├── model.xml           # 💾 Saved AI Model
└── images/             # 🖼️ UI Assets (Rock.jpeg, etc.)
```

---

## 🔧 Troubleshooting

-   **Black Screen?** Ensure your webcam is not being used by another app (Zoom, Teams).
-   **Laggy?** Try switching to a well-lit room. The CV model works best with good lighting.
-   **No Sound?** Ensure your system volume is up. If `pyttsx3` fails, the game will run silently.

---

*Created with ❤️ by V.Tanush*
