# Privacy & Data Sovereignty Architecture
## SIGNBRIDGE 2.0 Assistive Platform

Assistive technology bridges vulnerable communication gaps. SIGNBRIDGE is architected with strict mathematical and procedural privacy safeguards.

---

## 1. Core Principles

1. **Local Inference by Default**: Machine learning inference executes on local compute (in-browser or local container). No audio or video streams are forwarded to external multi-tenant cloud APIs.
2. **Ephemerality of Raw Sensor Data**: Raw video frames from webcams exist solely in volatile RAM for the duration of landmark calculation and are overwritten immediately.
3. **Decoupled Landmark Representation**: The deep learning model receives only 153 normalized geometric coordinates $(x, y, z)$. It possesses no capacity to reconstruct user facial features, biometric identities, or room backgrounds.
4. **Instant Upload Purging**: When processing uploaded video files, the server writes the file to a restricted temporary directory, performs frame extraction, runs BiLSTM inference, and deletes the temporary file in a mandatory `finally` block within sub-500ms.
5. **Client-Owned History**: Translation history is stored exclusively in the user's browser `localStorage`. No centralized server database persists user messages, timestamps, or locations.

---

## 2. Data Flow Lifecycle

```
[User Camera / Video File]
          │
          ▼
   MediaPipe Vision (RAM Only)
          │  Extracts 153 Coordinates
          ▼
   Geometric Normalization
   (Translates to sternum origin; strips scale/background)
          │
          ▼
   PyTorch BiLSTM (Sub-3ms Inference)
          │
          ▼
   Classification Label & Translation String
          │
          ▼
   Browser SpeechSynthesis (Local audio playback)
          │
          ▼
   Local Browser Storage (Client-only logs, exportable/clearable)
```

---

## 3. Storage & Retention Matrix

| Data Class | Collected? | Stored on Server? | Persisted in Cloud? | Client Control |
|---|---|---|---|---|
| **Webcam Video Stream** | In-Memory Only | No | No | Camera toggle / Stop button |
| **Uploaded Video File** | Transiently | Deleted instantly | No | Local disk only |
| **Biometric Face Embeddings** | No | No | No | N/A |
| **Translation Logs** | Client-side only | No | No | 1-Click Clear / JSON Export |
| **Speech Audio Recordings** | Generated in browser | No | No | Mute toggle / Local device |

---

## 4. Compliance & Ethics

SIGNBRIDGE aligns with:
- **Article 9 of the UN Convention on the Rights of Persons with Disabilities (CRPD)**: Ensuring accessible technology respects personal privacy and individual autonomy.
- **Data Protection Principles**: Strict data minimization, purpose limitation, and storage limitation.
