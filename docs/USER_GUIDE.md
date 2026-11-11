# SIGNBRIDGE 2.0 User Guide & Exhibition Manual

Welcome to SIGNBRIDGE 2.0. This guide walks you through using each component of the platform for demonstrations, classroom practice, and assistive communication.

---

## 1. Operating Modes in Translator Workspace

Open **Live Translator** (`/translator`) from the top navigation bar.

### Mode A: Video Upload (No Webcam Required)
1. Select the **Upload Video** tab.
2. Drag and drop any recorded video clip of an Indian Sign Language gesture (MP4, WebM, MOV, or AVI up to 50MB), or click **Select File**.
3. Click **Run AI Recognition**.
4. The system decodes the video, extracts landmarks, runs the BiLSTM classifier, displays the recognized sign, and speaks the translation in your chosen language.
5. The temporary video file is deleted immediately after processing.

### Mode B: Live Camera (For Devices with Webcams)
1. Select the **Live Camera** tab.
2. Click **Start Webcam** and grant camera permissions when prompted.
3. Stand 1–2 meters from the camera ensuring your chest, shoulders, and both hands are clearly visible in the video frame.
4. Execute an authentic ISL gesture steadily over ~1 second.
5. The real-time skeletal overlay tracks your motion, and confident predictions trigger localized voice output.
6. If no camera is detected, a notice will appear with 1-click shortcuts to Video Upload or Demonstration mode.

### Mode C: Demonstration Mode (For Booths & Exhibitions)
1. Select the **Demo Mode** tab.
2. Choose any sign from the vocabulary roster (e.g., `Namaste`, `Help`, `Doctor`, `Water`).
3. Click **Start Simulation**.
4. The system renders canonical biomechanical motion arcs on the canvas with an explicit `DEMONSTRATION MODE` watermark, illustrating how the AI translates the gesture.

---

## 2. Multi-Lingual Speech Output

Use the top toolbar to switch between speech output languages:
- **EN**: English (`en-IN`)
- **HI**: Hindi (`hi-IN`)
- **TE**: Telugu (`te-IN`)
- **Mute / Unmute**: Toggle audio playback at any time using the speaker icon.

---

## 3. Interactive Practice Studio

Open **Practice Studio** (`/learning`) to practice signing form:
1. Select a target sign from the left sidebar.
2. Review the canonical instructions, movement path, and single/two-handed requirements.
3. Choose your practice method:
   - **Interactive Trajectory**: Click to view real-time feedback against synthetic target form.
   - **Upload Attempt Video**: Click **Upload Attempt Video** to evaluate a recorded video of your signing.
4. Receive a similarity score (0–100%), qualitative grade (`Excellent`, `Good`, `Needs Improvement`), and kinematic feedback.

---

## 4. Research & Model Performance Dashboard

Open **Research & Metrics** (`/research`) to explore the scientific foundations:
- **Live Model KPIs**: Real-time parameter count (796,426), CPU inference latency (2.07 ms), and threshold floor ($\tau = 0.75$).
- **Synthetic vs. Real-World Targets**: Transparent comparison of in-silico benchmarks vs. unconstrained clinical expectations.
- **Class Performance Table**: Per-sign precision, recall, and F1 scores.
- **Confusion Matrix**: Quantitative visualization across all 10 vocabulary classes.

---

## 5. Translation History & Privacy Controls

Open **History** (`/history`) to review past translations:
- Each item displays the translated text, regional language, confidence percentage, timestamp, and input mode badge (`Upload`, `Camera`, `Demo`).
- **Hear Voice**: Click the speaker icon to replay the translation audio.
- **Copy Text**: Click the copy icon to copy the text to your clipboard.
- **Delete Single Item**: Click the trash icon next to any individual record.
- **Export JSON**: Download your full session log for offline documentation.
- **Clear All**: Purge all local history with one click.
