# SIGNBRIDGE TITAN — ZERO-LEAK DATA COLLECTION PROTOCOL
## Multi-Signer Human Subject Acquisition, Ethical Governance & Generalization Benchmark Standard

**Protocol Version:** 1.0.0-PROD  
**Research Mission:** OPERATION ZERO-LEAK  
**Target Repository Branch:** `experiment/titan-zero-leak`  
**Reference Commit:** `9489e8c`  
**Status:** ACTIVE SPECIFICATION & OPERATIONAL INGESTION WORKFLOW  

---

## 1. Executive Summary & Scientific Purpose

The core limitation of contemporary sign language gesture recognition systems is their tendency to overfit to homogeneous signer cohorts or synthetic augmentations. When models are evaluated on the same signers seen during training, empirical accuracy frequently exceeds 98%, creating an illusion of real-world readiness. However, when deployed before novel human users with differing hand morphologies, finger aspect ratios, flexibility constraints, and natural resting angles, accuracy precipitously drops.

The **SIGNBRIDGE TITAN ZERO-LEAK Data Collection Protocol** defines an immutable, scientifically defensible, and ethically governed methodology for capturing real human signers for:
1. Multi-signer domain adaptation training.
2. Unseen signer validation.
3. Completely untainted, frozen-model held-out testing.

```
       ========================================================================
                          ZERO-LEAK COHORT PARTITION DESIGN
       ========================================================================

        [ Signer_P01 ... Signer_P15 ]   -->  TRAINING COHORT (15 Signers)
             |                                - Multi-signer morphological diversity
             v                                - Parameter optimization
        
        [ Signer_P16 ... Signer_P20 ]   -->  VALIDATION COHORT (5 Signers)
             |                                - Hyperparameter validation & early stopping
             v                                - NOT used in final test metrics
        
        [ Signer_P21 ... Signer_P25 ]   -->  PERMANENT HELD-OUT TEST COHORT (5 Signers)
                                              - FROZEN: ZERO training exposure
                                              - ZERO threshold tuning exposure
                                              - ZERO feature-selection influence
       ========================================================================
```

---

## 2. Participant Recruitment & Stratification Standard

The target protocol specifies **25 consenting human participants**, selected across diverse physiological categories:

### 2.1 Participant Stratification Matrix
| Participant Category | Target Signer IDs | Cohort Allocation | Primary Function |
| :--- | :--- | :--- | :--- |
| **Training Pool** | `Signer_P01` – `Signer_P15` | Train (60%) | Base parameter optimization |
| **Validation Pool** | `Signer_P16` – `Signer_P20` | Validation (20%) | Cross-validation & early stopping |
| **Held-Out Test Pool** | `Signer_P21` – `Signer_P25` | **Held-Out Test (20%)** | **Unseen Zero-Leak Generalization** |

### 2.2 Invariant Firewalls for Held-Out Test Participants (`Signer_P21`–`Signer_P25`)
1. **Zero Weight Adaptation:** No gradient update or fine-tuning of any neural layer may ingest samples from `Signer_P21` through `Signer_P25`.
2. **Zero Post-Hoc Threshold Tuning:** Temperature scaling parameter ($T = 1.08$), confidence gate threshold ($\tau_{\text{conf}} = 0.55$), and entropy rejection threshold ($\tau_{\text{entropy}} = 1.35$) must remain strictly fixed.
3. **Zero Preprocessing Leaks:** Normalization statistics (such as mean joint lengths or feature variance) must never be re-centered using held-out test data.

---

## 3. Ethical Standards & Non-PII Safeguards

To ensure full compliance with international data privacy standards (including GDPR Article 17 "Right to Erasure" and institutional review guidelines for educational human trials):

### 3.1 Non-PII Kinematic Ingestion
- **No RGB Video Storage:** The video stream captured by the webcam is processed strictly in volatile GPU/CPU RAM via client-side MediaPipe Web Assembly. Video frames are discarded immediately upon landmark extraction.
- **No Facial Biometrics:** Face detection is deactivated. Only hand landmarks ($21$ joints: $x, y, z$) are digitized.
- **Pseudonymous Identification:** Every participant is assigned a randomized, non-reconstructable identifier (`Signer_P01` through `Signer_P25`). No names, emails, IP addresses, or device MAC addresses are saved in dataset manifests.

### 3.2 Informed Consent & Right to Erasure Workflow
1. **Active Pre-Capture Consent:** The browser application (`ZeroLeakDataCollector.tsx`) mandates an explicit opt-in checkbox before the capture shutter or trajectory recorder is activated.
2. **Real-Time Deletion:** The user interface provides individual sample deletion (`handleDeleteSample`) and total session purge (`handleClearSession`) buttons, allowing participants to delete any capture prior to export.

```
       +---------------------------------------------------------------+
       |             ETHICAL DATA ACQUISITION WORKFLOW                 |
       +---------------------------------------------------------------+
       | 1. Participant briefed on non-PII kinematic coordinate study   |
       | 2. Participant checks "Consent Verified" in Zero-Leak Studio  |
       | 3. Webcam initialized; landmarks processed in local browser   |
       | 4. Static / Dynamic signs captured to local session buffer    |
       | 5. Participant reviews samples; deletes any unsatisfactory    |
       | 6. Anonymized JSON manifest exported                          |
       +---------------------------------------------------------------+
```

---

## 4. Multi-Condition Capture Matrix per Participant

Each participant is requested to complete recordings encompassing systematic environmental and physical variations.

### 4.1 Target Sign Vocabulary (26 Classes)
- **Static Manual Alphabet (24 Signs):** `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`.
- **Dynamic Manual Alphabet (2 Signs):**
  - `J`: Extended pinky finger drawing a 3D downward curve terminating in an upward hook.
  - `Z`: Extended index finger tracing a 3-stroke horizontal zigzag in 3D space.

### 4.2 Environmental Variations Matrix
| Dimension | Condition Levels | Operational Definition |
| :--- | :--- | :--- |
| **Camera Distance** | Near (<0.5m) | Hand occupies >45% of camera vertical frame height. |
| | Medium (0.5m–1.0m) | Standard desk operating distance; hand occupies 20–35% of frame. |
| | Far (>1.0m) | Standing distance; hand occupies <15% of frame. |
| **Hand Orientation** | Frontal (0°) | Palm normal directly facing camera lens. |
| | Lateral 30° / 45° | Wrist yaw rotated 30° to 45° toward sagittal plane. |
| | Lateral 60° | Extreme yaw rotation displaying significant self-occlusion. |
| | Angled / Tilted | Pitch / roll variation (hand tilted ±25° upward/downward). |
| **Ambient Lighting** | Natural Daylight | Indirect window sunlight (300–600 lux). |
| | Fluorescent Office | Overhead artificial lighting (400–500 lux). |
| | Dim Ambient | Low ambient lighting (<100 lux). |
| | Strong Backlit | Bright light source directly behind the signer. |
| **Handedness** | Right Hand / Left Hand | Natural dominant and non-dominant demonstrations. |

---

## 5. Technical Data Specification & Schema

Captured datasets exported by the Zero-Leak studio adhere strictly to the schema below:

```json
{
  "exportVersion": "1.0.0-zero-leak",
  "sessionMetadata": {
    "sessionId": "SES_XXXXXXXX",
    "exportTimestamp": "2026-11-14T11:40:00Z",
    "totalSamples": 26,
    "signerCohortDistribution": {
      "train": 0,
      "val": 0,
      "test": 26
    },
    "ethicalCompliance": {
      "piiCollected": false,
      "consentObtained": true,
      "rightToEraseAvailable": true
    }
  },
  "records": [
    {
      "id": "SMP_A_1731580800000",
      "timestamp": "2026-11-14T11:40:05Z",
      "signerId": "Signer_P21",
      "cohort": "test",
      "letter": "A",
      "isDynamic": false,
      "metadata": {
        "distance": "Medium (0.5m-1.0m)",
        "orientation": "Frontal (0°)",
        "lighting": "Natural Daylight",
        "handedness": "Right",
        "resolution": "640x480",
        "fps": 30,
        "trackingQualityScore": 96.5
      },
      "landmarks": [
        { "x": 0.512, "y": 0.741, "z": 0.002 }
      ],
      "features42": [
        0.412, 0.523, 0.618
      ]
    },
    {
      "id": "DYN_J_1731580815000",
      "timestamp": "2026-11-14T11:40:20Z",
      "signerId": "Signer_P21",
      "cohort": "test",
      "letter": "J",
      "isDynamic": true,
      "metadata": {
        "distance": "Medium (0.5m-1.0m)",
        "orientation": "Frontal (0°)",
        "lighting": "Natural Daylight",
        "handedness": "Right",
        "resolution": "640x480",
        "fps": 30,
        "trackingQualityScore": 94.0
      },
      "trajectory": [
        { "x": 0.501, "y": 0.380, "t": 1731580815100 }
      ],
      "resampledTrajectory32": [
        0.0, 0.0, -0.012, 0.184
      ]
    }
  ]
}
```

---

## 6. Execution Guidelines: How to Operate the Zero-Leak Studio

1. **Launch SIGNBRIDGE:** Open the web application and navigate to the alphabet recognition view (`LiveISLAlphabetRecognizer.tsx`).
2. **Open Studio:** Click the **"Zero-Leak Studio"** button in the header (gold database icon).
3. **Verify Consent:** Review the non-PII disclosure with the participant and check the consent box.
4. **Configure Signer & Session:**
   - Select the participant's pseudonym ID (e.g., `Signer_P21`). The cohort will automatically lock to `test`.
   - Select the current environmental metadata (lighting, distance, orientation).
5. **Capture Static Signs:**
   - Select letter (`A` through `Y`, excluding `J`).
   - Ask signer to form the handshape and stabilize.
   - Click **"Capture Static '[Letter]'"**.
6. **Capture Dynamic Signs (`J` and `Z`):**
   - Select letter `J` or `Z`.
   - Click **"Record [Letter] (1.2s)"**.
   - Signer executes stroke in air while progress bar fills.
7. **Audit & Export:**
   - Review the captured table.
   - Click **"Export Zero-Leak Dataset (JSON)"** to download the signed session file.

---

## 7. Handling Incomplete Captures & Missing Data Policy

- **No Synthetic Imputation:** Missing letters must be documented as `MISSING_CAPTURE` in the manifest rather than replaced with interpolated or synthetic samples.
- **Occlusion Tracking Failures:** When MediaPipe loses tracking during extreme lateral angles (>60°), the sample is flagged with `trackingQualityScore < 50` and recorded with `isTrackingLost: true`.
