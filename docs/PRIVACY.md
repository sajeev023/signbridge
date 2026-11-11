# Privacy & Data Sovereignty Architecture
## SIGNBRIDGE 2.1 Assistive Platform

Assistive technology bridges vulnerable communication gaps. SIGNBRIDGE is architected with strict mathematical and procedural privacy safeguards.

---

## 1. Core Principles & Operational Disclosure

1. **Dual-Tier Processing Architecture**:
   - **Mode B (Live Camera)**: Operates within browser RAM. Video frames are analyzed in-memory and discarded without cloud transmission.
   - **Mode A (Video Upload)**: Ingested transiently by the application server into an isolated temporary folder for uniform temporal sampling and BiLSTM inference. The file is deleted immediately in a mandatory server-side `finally` block within sub-500ms.
   - **Mode C (Demonstration)**: Generates synthetic trajectories purely in client JavaScript/HTML5 canvas.
2. **Ephemerality of Raw Sensor Data**: Raw video frames from webcams are never recorded, saved, or persisted on server disks.
3. **Decoupled Landmark Representation**: The deep learning model receives only 153 normalized geometric coordinates $(x, y, z)$. It cannot reconstruct user facial features, biometric identities, or room backgrounds.
4. **Instant Upload Purging**: Server-side video files are deleted immediately after keyframe extraction. No archive, logs, or secondary copies are created.
5. **Client-Owned History**: Translation history is stored exclusively in the user's browser `localStorage`. No centralized database persists user messages, timestamps, or locations.

---

## 2. Data Flow Lifecycle

```
[Live Webcam (Client RAM)]              [Video Upload (MP4/WebM)]
          │                                         │
          │ (Client-side tracking)                  ▼
          │                             Transient Server Ingestion
          │                             (/tmp/signbridge_uploads)
          │                                         │
          ▼                                         ▼
   MediaPipe Vision ─────────────────► Keyframe Sampling (30 Frames)
   (RAM Only)                                       │
          │                                         ▼
          │                             Instant Temp File Deletion (sub-500ms)
          │                                         │
          ▼                                         ▼
   Geometric Normalization ──────────► 153-D Normalized Coordinates
   (Translates to sternum origin; strips scale/background)
          │
          ▼
   PyTorch BiLSTM (2.07ms CPU Inference)
          │
          ▼
   Classification Label & Translation String
          │
          ▼
   Browser SpeechSynthesis (Local audio playback in EN/HI/TE)
          │
          ▼
   Local Browser Storage (Client-only logs, exportable/clearable)
```

---

## 3. Storage & Retention Matrix

| Data Class | Collected? | Stored on Server? | Persisted in Cloud? | Client Control |
|---|---|---|---|---|
| **Webcam Video Stream** | In-Memory Only | No | No | Camera toggle / Stop button |
| **Uploaded Video File** | Transiently (Mode A) | Deleted in < 500ms | No | User-initiated file upload |
| **Biometric Face Embeddings** | No | No | No | N/A |
| **Translation Logs** | Client-side only | No | No | 1-Click Clear / JSON Export |
| **Speech Audio Recordings** | Generated in browser | No | No | Mute toggle / Local device |

---

## 4. Compliance & Ethics

SIGNBRIDGE aligns with:
- **Article 9 of the UN Convention on the Rights of Persons with Disabilities (CRPD)**: Ensuring accessible technology respects personal privacy and individual autonomy.
- **Data Protection Principles**: Strict data minimization, purpose limitation, storage limitation, and operational transparency.
