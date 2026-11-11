# SIGNBRIDGE Dataset Documentation

## 1. Overview & Linguistic Grounding

The SIGNBRIDGE dataset captures 10 high-priority communicative and emergency gestures from **Indian Sign Language (ISL)**. The morphological definition of each gesture follows the national standards codified by the Indian Sign Language Research and Training Centre (ISLRTC).

### Supported Vocabulary Roster

| Class ID | Canonical Label | Category | Hand Configuration | Kinematic Stroke Description |
|---|---|---|---|---|
| `namaste` | Namaste / Welcome | Greetings | Two-handed | Both palms joined at center chest; slight forward tilt. |
| `thank_you` | Thank You | Social | Single-handed | Flat hand touches chin, sweeps forward-downward toward viewer. |
| `help` | Help / Need Assistance | Emergency | Two-handed | Dominant fist on open non-dominant palm, lifting together. |
| `yes` | Yes / Affirmation | Social | Single-handed | Closed fist nodding vertically up and down at wrist. |
| `no` | No / Negation | Social | Single-handed | Open flat hand or index finger waving horizontally side to side. |
| `water` | Water / Drink | Daily Needs | Single-handed | Curved hand elevates toward mouth and tilts back mimicking sipping. |
| `food` | Food / Eat | Daily Needs | Single-handed | Bunched fingertips repeatedly tapping lips. |
| `emergency` | Emergency / Urgent | Emergency | Two-handed | Both hands raised, crossing frantically and waving at chest height. |
| `doctor` | Doctor / Hospital | Medical | Two-handed | Dominant two fingers tap twice on non-dominant radial wrist pulse. |
| `please` | Please / Request | Social | Single-handed | Dominant open hand making clockwise circular smoothing strokes on chest. |

---

## 2. Dataset Synthesis & Biomechanical Augmentation

To establish a reproducible benchmark without compromising privacy, the dataset was synthesized through kinematic trajectory generators modeling canonical ISLRTC joint angles and velocities, followed by extensive physical stochastic augmentations:

1. **Spatial Translation**: Gaussian shifts $(\Delta x, \Delta y, \Delta z) \sim \mathcal{U}(-0.05, 0.05)$ simulating standing slightly off-center.
2. **Torso Scale Variance**: Signer torso and arm length scaling $S \sim \mathcal{U}(0.88, 1.12)$.
3. **Perspective Yaw Rotation**: 3D camera angle yaw rotations $\pm 10^\circ$.
4. **Sensor Noise**: Keypoint micro-jitter $\mathcal{N}(0, 0.005)$ modeling real-world webcam landmark detection fluctuations.
5. **Temporal Warping**: Gesture speed variations (20 to 40 frames resampled to 30 frames) modeling rapid vs deliberate signing.

---

## 3. Dataset Splits & Structure

- **Total Sequences**: 1,200 (120 sequences per class)
- **Sequence Length**: 30 frames (1.0 second at 30 FPS)
- **Features Per Frame**: 153 normalized floating-point coordinates
- **Total Frames**: 36,000 frames
- **Storage Location**: `backend/data/isl_sequences.npz`

| Partition | Percentage | Sequence Count | Samples Per Class |
|---|---|---|---|
| Training Set | 70% | 840 | 84 |
| Validation Set | 15% | 180 | 18 |
| Test Set (Held-Out) | 15% | 180 | 18 |

---

## 4. Expansion Pipeline for Real Signer Video

To contribute real-world video recordings:
1. Capture MP4 video clips (640x480 at 30 FPS) under informed consent.
2. Run `python backend/app/services/landmark_extractor.py` to extract and normalize landmarks into `.npz` records.
3. Verify annotations against the ISLRTC dictionary.
4. Append to training partition and re-evaluate on test split.
