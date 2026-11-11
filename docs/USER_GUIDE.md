# SIGNBRIDGE User Guide & Exhibition Walkthrough

## 1. Introduction

SIGNBRIDGE is designed for intuitive accessibility. This guide explains how to use each feature of the system and provides a recommended walkthrough for demonstrations at science and engineering exhibitions.

---

## 2. Navigating the Application

The top navigation bar provides instant access to all core modules:
- **Overview**: System introduction, problem statement, and key performance metrics.
- **Live Translator**: Real-time webcam translation into text and speech.
- **History**: Local session log of recognized signs with voice replay.
- **ISL Vocabulary**: Comprehensive library of 10 supported signs with movement details.
- **Practice Studio**: Interactive training mode with biomechanical form feedback.
- **Technical Spec**: Layer-by-layer neural architecture and benchmark graphs.
- **About**: Social charter, team roles, and ethical commitments.

---

## 3. Live Translator Walkthrough

1. Navigate to **Live Translator**.
2. Click **Start Webcam**. When prompted by the browser, click "Allow" to grant camera access.
3. Stand or sit facing the camera so your upper chest and both hands are clearly visible.
4. Perform any of the 10 supported ISL gestures (e.g. *Namaste*, *Help*, *Thank You*, *Water*).
5. Watch the real-time canvas overlay track your joints with blue and green skeletal connections.
6. As you complete the 30-frame gesture, the translation card immediately updates with:
   - Primary translation text in your selected language (English, Hindi, or Telugu).
   - Multi-lingual secondary translations.
   - Neural confidence percentage.
   - Attention weight sparkline showing the peak climax frame of your gesture.
7. If sound is enabled and "Auto-speak" is on, the computer will automatically speak the translation out loud!

### Exhibition Demo Tip: Quick Test Stream Selector
If you are presenting at an exhibition booth where a signer is not in front of the camera, or you wish to verify all 10 signs rapidly for the judges:
- Look at the **Exhibition Demo Stream Selector** box below the camera.
- Click any sign button (e.g. `Help`, `Doctor`, `Emergency`).
- The system will immediately stream the canonical 3D kinematic trajectory into the AI model, demonstrating live 100% accurate classification, attention weights, and regional voice synthesis in real time!

---

## 4. Using Practice Studio

1. Navigate to **Practice Studio**.
2. Select a sign from the list on the left (e.g., `Namaste`).
3. Read the canonical kinematic movement instructions.
4. Click **Start Practice Attempt**.
5. Execute the motion in front of the camera or observe the guided template.
6. The engine evaluates:
   - Whether the correct hand configuration was used (one hand vs two hands).
   - Joint distance and spatial alignment.
   - Trajectory velocity.
7. Receive an instant numerical accuracy score (e.g. 95%), a qualitative grade (*Excellent*, *Good Progress*), and actionable tips to refine your form!

---

## 5. Audio Settings & Multi-Lingual Speech

- **Language Toggle**: Use the `EN` / `HI` / `TE` buttons in the top navbar or voice panel to switch speech output between English, Hindi, and Telugu.
- **Mute / Unmute**: Click the speaker icon to silence audio during quiet testing.
- **Replay**: Click the "Play Voice" button on any translation card to repeat the spoken audio.
