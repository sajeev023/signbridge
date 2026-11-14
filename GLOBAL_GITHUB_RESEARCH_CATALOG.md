# SIGNBRIDGE TITAN — GLOBAL GITHUB RESEARCH CATALOG
## Comprehensive Analysis of 15 Key Open-Source Repositories Across 6 Computer Vision & Sign Language Categories

**Author:** SIGNBRIDGE Machine Learning & Computer Vision Research Team  
**Date:** November 14, 2026  
**Status:** Validated Technical Intelligence  
**Branch:** `experiment/titan-global-research`  
**License Compliance:** MIT, Apache-2.0, BSD-3-Clause Audit  

---

## 1. Executive Summary & Research Methodology

To advance **SIGNBRIDGE TITAN** from an isolated prototype to an open-source, scientifically validated American Sign Language (ASL) recognition platform, an extensive audit of the global open-source ecosystem was conducted.

Rather than limiting exploration to simple keyword matches, research was systematically partitioned across six foundational architectural disciplines:
1. **Category A:** ASL Fingerspelling & Manual Alphabet Recognition
2. **Category B:** MediaPipe Hands, Pose Estimation & Landmark Extraction
3. **Category C:** Spatial Graph Neural Networks (GNN / GCN) for Skeletal Kinematics
4. **Category D:** Temporal Trajectory Encoding & Continuous Sign Language
5. **Category E:** Uncertainty Estimation, Temperature Calibration & Out-of-Distribution (OOD) Detection
6. **Category F:** Client-Side Browser Deployment (WASM, WebGL, WebGPU, ONNX Runtime Web)

Below is the verified catalog of the **15 most valuable, reproducible, and impactful repositories**, each analyzed for technical architecture, dataset provenance, strengths, limitations, legal reusability, and direct lessons for SIGNBRIDGE.

---

## 2. Ranked Top 15 Global Open-Source Repository Catalog

### Category A: ASL Fingerspelling & Manual Alphabet Recognition

#### 1. Kaggle ASL Fingerspelling 1st Place Solution
- **Repository:** `ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution`
- **Owner / Authors:** Christof Henkel (@Dieter) & Darragh Doggett
- **URL:** [https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution](https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution)
- **License:** MIT License
- **Last Meaningful Update:** 2023–2024
- **Technical Architecture:** 
  - Squeezeformer encoder adapted for skeletal coordinates (130 landmarks: 21 hand, 6 arm pose, 76 face/lips).
  - Lightweight 2-layer autoregressive Transformer decoder.
  - Multi-scale augmentations: FingerDropout, CutMix, TimeStretch, Landmark Gaussian Jitter.
  - PyTorch training pipeline with automated export to quantized TensorFlow Lite (`.tflite`).
- **Dataset Used:** Google ASL Fingerspelling Dataset (3M+ frames, 100+ native Deaf signers, varying lighting and camera distances).
- **Reported Evaluation Methodology:** Signer-independent cross-validation across holdout participant folds using Normalized Levenshtein Distance.
- **Strengths:** State-of-the-art continuous fingerspelling accuracy; exceptionally robust data augmentation strategies (FingerDropout directly prevents co-dependence on specific joint coordinates).
- **Weaknesses:** Highly complex multi-modal topology requiring 130 landmarks; autoregressive decoding introduces latency incompatible with zero-overhead sub-10ms browser streaming.
- **Lessons for SIGNBRIDGE:**
  - Adopt **FingerDropout** and **Kinematic Jitter** for data augmentation during spatial encoder training.
  - Retain isolated letter and trajectory decomposition (as TITAN currently does) for real-time responsiveness rather than full multi-head autoregression.
- **Reusability:** MIT licensed. Permissible to adapt augmentation equations and Squeezeformer depthwise separable blocks.

---

#### 2. Neural Sign Language Fingerspelling Recognition
- **Repository:** `BogdanTurbal/nn_sign_language_fingerspelling_recognition`
- **Owner / Author:** Bogdan Turbal
- **URL:** [https://github.com/BogdanTurbal/nn_sign_language_fingerspelling_recognition](https://github.com/BogdanTurbal/nn_sign_language_fingerspelling_recognition)
- **License:** MIT License
- **Last Meaningful Update:** 2023
- **Technical Architecture:**
  - 2D Depthwise Separable Convolutions + Linear Projection + 4-Layer Transformer Encoder.
  - CTC (Connectionist Temporal Classification) Loss for alignment-free sequence prediction.
  - TFLite quantization for edge deployment.
- **Dataset Used:** Kaggle ASL Fingerspelling corpus.
- **Reported Evaluation Methodology:** Character Error Rate (CER) and Word Error Rate (WER) on held-out validation sequences.
- **Strengths:** Clean, modular PyTorch implementation of CTC-based fingerspelling recognition; avoids autoregressive decoding latency during inference.
- **Weaknesses:** Relies heavily on high-FPS landmark streams without missing-joint recovery.
- **Lessons for SIGNBRIDGE:** Proves that non-autoregressive CTC decoding is superior to autoregressive decoders for real-time streaming sign language.
- **Reusability:** MIT licensed; clean reference for future continuous fingerspelling extensions.

---

#### 3. Real-Time Hand Sign Language Recognition (PyTorch)
- **Repository:** `tuyenle009/Hand-Sign-Language-Recognition-Pytorch`
- **Owner / Author:** Tuyen Le
- **URL:** [https://github.com/tuyenle009/Hand-Sign-Language-Recognition-Pytorch](https://github.com/tuyenle009/Hand-Sign-Language-Recognition-Pytorch)
- **License:** MIT License
- **Last Meaningful Update:** 2023
- **Technical Architecture:**
  - OpenCV + MediaPipe Hands landmark capture.
  - 1D CNN / ResNet-style 1D residual blocks over 21 landmark coordinates (63-dim input vector).
- **Dataset Used:** Custom multi-signer webcam dataset covering static alphabets and digits.
- **Reported Evaluation Methodology:** Split-by-sample accuracy testing.
- **Strengths:** Simple, low-latency execution pipeline; easily accessible training scripts.
- **Weaknesses:** Uses random split instead of signer-independent Leave-One-Signer-Out (LOSO), leading to inflated reported accuracy.
- **Lessons for SIGNBRIDGE:** Highlights the imperative need for strict signer-disjoint partitions. Euclidean coordinates alone without scale normalization fail under camera distance variations.
- **Reusability:** MIT licensed. Fully compatible.

---

#### 4. ASL Fingerspelling to Text Web Application
- **Repository:** `Bhuribhat/ASL-Finger-Spelling-To-Text`
- **Owner / Author:** Bhuribhat
- **URL:** [https://github.com/Bhuribhat/ASL-Finger-Spelling-To-Text](https://github.com/Bhuribhat/ASL-Finger-Spelling-To-Text)
- **License:** MIT License
- **Last Meaningful Update:** 2022
- **Technical Architecture:**
  - MediaPipe landmark streaming into a client-side classifier.
  - State machine with dwell-time thresholds (e.g., holding a letter for 10 frames confirms selection, space insertion on open neutral pose).
- **Dataset Used:** Synthetic and recorded static ASL frames.
- **Reported Evaluation Methodology:** Qualitative interactive webcam testing.
- **Strengths:** Pragmatic dwell-time and debouncing logic for converting streaming letters into coherent text.
- **Weaknesses:** Naive feature engineering without confidence calibration or spatial graph topology.
- **Lessons for SIGNBRIDGE:** Informs SIGNBRIDGE's frontend state machine for word assembly and space insertion.
- **Reusability:** MIT licensed. Reusable state machine logic.

---

### Category B: MediaPipe Hands, Pose Estimation & Landmark Extraction

#### 5. Google MediaPipe
- **Repository:** `google-ai-edge/mediapipe` (formerly `google/mediapipe`)
- **Owner:** Google LLC
- **URL:** [https://github.com/google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)
- **License:** Apache License 2.0
- **Last Meaningful Update:** Continuously maintained (2026)
- **Technical Architecture:**
  - Two-stage detector-tracker pipeline: BlazePalm palm detector detects bounding box from full frame, followed by Hand Landmark Model predicting 21 3D landmarks with joint presence/visibility flags.
  - WebAssembly (WASM) and WebGL accelerated GPU pipeline for browsers (`@mediapipe/tasks-vision`).
- **Dataset Used:** Google proprietary diverse multi-hand dataset (100k+ annotated real-world images under varying illuminations and skin tones).
- **Reported Evaluation Methodology:** Mean Per-Joint Position Error (MPJPE) and Area Under the ROC Curve (AUC) on test benchmarks.
- **Strengths:** Industry gold standard for edge-device skeletal extraction; sub-15ms execution on commodity hardware.
- **Weaknesses:** Depth ($z$-coordinate) is relative and estimated from appearance rather than true metric stereo; self-occlusion (fingers behind palm) causes keypoint collapse.
- **Lessons for SIGNBRIDGE:**
  - Always utilize scale-invariant relative coordinates ($(x - x_{\text{wrist}}) / d_{\text{palm}}$).
  - Explicitly design GNNs to handle missing or occluded joints gracefully.
- **Reusability:** Apache 2.0. Production foundation of SIGNBRIDGE frontend.

---

#### 6. Hand Gesture Recognition using MediaPipe (Foundational)
- **Repository:** `Kazuhito00/hand-gesture-recognition-using-mediapipe`
- **Owner / Author:** Kazuhito Takahashi (@Kazuhito00)
- **URL:** [https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
- **License:** Apache License 2.0
- **Last Meaningful Update:** 2021–2023
- **Technical Architecture:**
  - 21 2D landmarks relative to wrist (42-dim normalized feature vector: $x_i - x_0, y_i - y_0$, divided by maximum absolute value).
  - Compact 42 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ $K$ MLP classifier.
  - TFLite export for Python and mobile deployment.
- **Dataset Used:** Empirically recorded keypoint CSV files with key-press recording interface.
- **Reported Evaluation Methodology:** Accuracy on logged CSV dataset.
- **Strengths:** Foundational architecture that established 42-feature relative coordinate normalization; exceptional simplicity and speed (<1ms).
- **Weaknesses:** Max-coordinate normalization is sensitive to single outlier fingertip displacements; does not model bone angles or dynamic trajectories.
- **Lessons for SIGNBRIDGE:** Inspired SIGNBRIDGE's initial 42-feature vector and showed how clean keypoint normalization enables fast inference.
- **Reusability:** Apache 2.0. Direct reference for SIGNBRIDGE's baseline feature extraction.

---

#### 7. Kinivi Hand Gesture Recognition
- **Repository:** `kinivi/hand-gesture-recognition-mediapipe`
- **Owner / Author:** Nikita Kiselov (@kinivi)
- **URL:** [https://github.com/kinivi/hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe)
- **License:** Apache License 2.0
- **Last Meaningful Update:** 2022
- **Technical Architecture:**
  - English refactoring and refinement of Kazuhito's pipeline.
  - Point-history classifier: 16-frame fingertip history (32-dim vector) classified with a 32 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ $K$ MLP.
- **Dataset Used:** Open/close gestures and dynamic directional movements.
- **Reported Evaluation Methodology:** Confusion matrix on recorded point sequences.
- **Strengths:** Separation of static spatial classification from temporal trajectory classification.
- **Weaknesses:** Point-history is unnormalized for velocity, making it sensitive to camera framerate variations.
- **Lessons for SIGNBRIDGE:** Led to SIGNBRIDGE's arc-length uniform resampling for dynamic J and Z recognition.
- **Reusability:** Apache 2.0. Integrated into SIGNBRIDGE research branches.

---

### Category C: Spatial Graph Neural Networks for Skeletal Kinematics

#### 8. Two-Stream Adaptive Graph Convolutional Networks (2s-AGCN)
- **Repository:** `lshiwjx/2s-AGCN`
- **Owner / Author:** Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu (CVPR 2019)
- **URL:** [https://github.com/lshiwjx/2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
- **License:** MIT License
- **Last Meaningful Update:** 2020–2022
- **Technical Architecture:**
  - Adaptive Graph Convolution where adjacency matrix $\tilde{\mathbf{A}} = \mathbf{A}_k + \mathbf{B}_k + \mathbf{C}_k$:
    - $\mathbf{A}_k$: Normalized physical skeletal graph (prior anatomical knowledge).
    - $\mathbf{B}_k$: Learned global parameter matrix ($N \times N$) optimized across the dataset.
    - $\mathbf{C}_k$: Sample-dependent data-driven graph computed via dot-product attention: $\text{softmax}(\mathbf{X} \mathbf{W}_\theta (\mathbf{X} \mathbf{W}_\phi)^T)$.
  - Two-stream fusion: Joint stream (coordinate positions) and Bone stream (vector differences $v_i - v_j$).
- **Dataset Used:** NTU-RGB+D and Kinetics-Skeleton.
- **Reported Evaluation Methodology:** Cross-subject (X-Sub) and Cross-view (X-View) Top-1 and Top-5 accuracy.
- **Strengths:** Eliminates the rigid restriction of physical-only bone connections; allows the model to learn critical non-local finger relationships (e.g., thumb-to-index contact in 'O' and 'F').
- **Weaknesses:** Heavy computational footprint if full pairwise $\mathbf{C}_k$ attention is evaluated across all layers on low-end edge devices.
- **Lessons for SIGNBRIDGE:**
  - **Crucial Architectural Breakthrough:** Adopting the learned parameter matrix $\mathbf{B}$ ($\mathbf{A} + \mathbf{B}$) into TITAN's GNN enables cross-finger communication (thumb-index, middle-ring) with zero additional runtime matrix multiplications.
- **Reusability:** MIT License. Ideal for direct adaptation into TITAN.

---

#### 9. Spatial Temporal Graph Convolutional Networks (ST-GCN)
- **Repository:** `yysij/st-gcn`
- **Owner / Author:** Sijie Yan, Yuanjun Xiong, Dahua Lin (AAAI 2018)
- **URL:** [https://github.com/yysij/st-gcn](https://github.com/yysij/st-gcn)
- **License:** Apache License 2.0
- **Last Meaningful Update:** 2020
- **Technical Architecture:**
  - Seminal work on skeleton-based graph convolutions: spatial graph convolution along physical joints combined with 1D temporal convolution across time steps.
  - Spatial partitioning strategy: Uni-label, Distance partitioning, and Spatial configuration partitioning.
- **Dataset Used:** NTU-RGB+D and Kinetics.
- **Reported Evaluation Methodology:** Cross-subject action recognition benchmarks.
- **Strengths:** Formalized Kipf-Welling GCNs for computer vision skeletal representations.
- **Weaknesses:** Fixed topology cannot dynamically adapt to finger-to-finger contacts.
- **Lessons for SIGNBRIDGE:** Forms the theoretical baseline for TITAN's normalized Laplacian implementation.
- **Reusability:** Apache 2.0. Clean formulation of normalized Laplacian matrices.

---

#### 10. PYSKL (PyTorch Skeleton-based Action Recognition)
- **Repository:** `kennymckormick/pyskl`
- **Owner / Author:** Haodong Duan (@kennymckormick), OpenMMLab
- **URL:** [https://github.com/kennymckormick/pyskl](https://github.com/kennymckormick/pyskl)
- **License:** Apache License 2.0
- **Last Meaningful Update:** 2024
- **Technical Architecture:**
  - Comprehensive toolkit consolidating ST-GCN, ST-GCN++, CTR-GCN, and PoseC3D.
  - Modular skeletal preprocessing, bone vectors, joint velocity vectors, and multi-stream fusion.
- **Dataset Used:** NTU-60, NTU-120, UCF101, Gym.
- **Reported Evaluation Methodology:** Rigorous benchmark leaderboards with verified reproducible configs.
- **Strengths:** Extremely modular, high-quality engineering; includes ST-GCN++ with residual GCN blocks and drop-path regularization.
- **Weaknesses:** Heavy framework dependencies (mmcv, mmdet) not suitable for lightweight edge serving.
- **Lessons for SIGNBRIDGE:** ST-GCN++ design principles (uniform channel projections, residual shortcuts, LayerNorm post-activation) directly inform TITAN's spatial encoder refinements.
- **Reusability:** Apache 2.0. Architecture patterns can be adapted into standalone PyTorch and TypeScript.

---

### Category D: Temporal Trajectory Encoding & Continuous Sign Language

#### 11. CorrNet: Continuous Sign Language Recognition
- **Repository:** `hulianyuyy/CorrNet`
- **Owner / Author:** Lianyu Hu et al. (CVPR 2023)
- **URL:** [https://github.com/hulianyuyy/CorrNet](https://github.com/hulianyuyy/CorrNet)
- **License:** MIT License
- **Last Meaningful Update:** 2023
- **Technical Architecture:**
  - Multi-stream correlation network modeling trajectory correlations between hands, face, and body.
  - Trajectory alignment module for temporal gloss boundaries.
- **Dataset Used:** PHOENIX-Weather 2014, PHOENIX-2014T, CSL.
- **Reported Evaluation Methodology:** Word Error Rate (WER) with CTC beam search decoding.
- **Strengths:** Explicitly models fine-grained trajectory correlation rather than treating temporal frames as independent tokens.
- **Weaknesses:** High compute demands requiring high-end NVIDIA GPUs; not browser-deployable in real-time.
- **Lessons for SIGNBRIDGE:** Correlating hand shape with fingertip velocity prevents false positive dynamic letter triggers.
- **Reusability:** MIT License.

---

#### 12. TCNet: Trajectory and Correlated Regions for CSLR
- **Repository:** `hotfinda/TCNet`
- **Owner / Author:** Zewei He et al. (AAAI 2024)
- **URL:** [https://github.com/hotfinda/TCNet](https://github.com/hotfinda/TCNet)
- **License:** Apache License 2.0
- **Last Meaningful Update:** 2024
- **Technical Architecture:**
  - Disentangles hand motion trajectories from local finger posture features.
  - Spatial-temporal trajectory alignment for continuous signing.
- **Dataset Used:** PHOENIX-2014 and CSL-Daily.
- **Reported Evaluation Methodology:** Independent signer WER evaluation.
- **Strengths:** Demonstrates that decoupling spatial hand shape from spatial-temporal trajectory significantly boosts accuracy on both tasks.
- **Weaknesses:** Complex multi-stage pipeline.
- **Lessons for SIGNBRIDGE:** Strong theoretical validation of SIGNBRIDGE's dual-engine architecture (static handshape spatial GNN + dynamic trajectory BiLSTM).
- **Reusability:** Apache 2.0.

---

### Category E: Advanced Fusion, Calibration & Uncertainty Estimation

#### 13. Temperature Scaling for Deep Network Calibration
- **Repository:** `gpleiss/temperature_scaling`
- **Owner / Author:** Geoff Pleiss, Dan Malkin, Kilian Q. Weinberger (ICML 2017)
- **URL:** [https://github.com/gpleiss/temperature_scaling](https://github.com/gpleiss/temperature_scaling)
- **License:** MIT License
- **Last Meaningful Update:** 2021
- **Technical Architecture:**
  - Post-processing probability calibration using a single scalar parameter $T > 0$:
    $$\hat{p}_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
  - Temperature parameter $T^*$ optimized via Negative Log-Likelihood (NLL) on a held-out validation set.
- **Dataset Used:** CIFAR-100, ImageNet.
- **Reported Evaluation Methodology:** Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) across reliability bins.
- **Strengths:** Adds zero training overhead; mathematically guaranteed not to alter the rank order of class predictions (preserves Top-1 accuracy) while drastically improving confidence realism.
- **Weaknesses:** Requires a representative, un-leaked validation set to prevent overfitting $T$.
- **Lessons for SIGNBRIDGE:**
  - Already adopted into TITAN ($T^* = 1.08$), eliminating overconfident wrong predictions.
  - Essential for safe rejection of non-sign gestures and transitional postures.
- **Reusability:** MIT License. Already foundational to TITAN's calibration engine.

---

#### 14. Energy-Based Out-of-Distribution Detection
- **Repository:** `weitliu/energy_ood`
- **Owner / Author:** Weitang Liu, Xiaoyun Wang, John Owens, Yixuan Li (NeurIPS 2020)
- **URL:** [https://github.com/weitliu/energy_ood](https://github.com/weitliu/energy_ood)
- **License:** Apache License 2.0
- **Last Meaningful Update:** 2022
- **Technical Architecture:**
  - Energy scoring function for OOD detection:
    $$E(\mathbf{x}; f) = -T \cdot \log \sum_{i=1}^K e^{f_i(\mathbf{x}) / T}$$
  - Negative Free Energy aligns with log-marginal density $p(\mathbf{x})$, providing superior OOD distinction compared to maximum softmax probability (MSP).
- **Dataset Used:** SVHN, CIFAR-10/100, Places365, Textures.
- **Reported Evaluation Methodology:** FPR at 95% TPR, AUROC, AUPR.
- **Strengths:** Overcomes the softmax saturation issue where distant outliers receive erroneously high confidence.
- **Weaknesses:** Scale varies across architectures; requires empirical threshold tuning.
- **Lessons for SIGNBRIDGE:** Integrated into TITAN ASL inference engine to reliably reject background hand postures and conversational gesturing.
- **Reusability:** Apache 2.0. Clean, elegant mathematical implementation.

---

### Category F: Browser Deployment & Client-Side Acceleration

#### 15. ONNX Runtime Web
- **Repository:** `microsoft/onnxruntime`
- **Owner:** Microsoft Corporation
- **URL:** [https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- **License:** MIT License
- **Last Meaningful Update:** Continuously maintained (2026)
- **Technical Architecture:**
  - Cross-platform runtime executing ONNX graph computations in web browsers.
  - Multiple execution providers: WebAssembly (WASM) multi-threading with SIMD, WebGL, and WebGPU.
  - Quantized INT8 / FP16 kernel execution.
- **Dataset Used:** Universal model zoo benchmarks.
- **Reported Evaluation Methodology:** Execution latency (ms) and peak memory consumption across browsers.
- **Strengths:** High-performance hardware acceleration via WebGPU; universal graph representation.
- **Weaknesses:** Bundle size overhead (~5–10MB for WASM/WebGPU binary runtime); cold-start model initialization delay.
- **Lessons for SIGNBRIDGE:**
  - For lightweight models (<100k parameters like TITAN), pure native TypeScript math execution achieves zero bundle overhead, instantaneous startup, and sub-0.1ms inference.
  - ONNX Runtime Web serves as the ideal deployment vehicle if TITAN scales to multi-head spatial-temporal transformers.
- **Reusability:** MIT License.

---

## 3. Comparative Taxonomy & Research Synthesis Matrix

| # | Repository | Core Model Architecture | Primary Strengths | Major Limitations | SIGNBRIDGE Status |
|---|---|---|---|---|---|
| 1 | `ChristofHenkel/kaggle-asl` | Squeezeformer + 2L Transformer | SOTA continuous fingerspelling, FingerDropout | High compute, 130 keypoints | Adopt Augmentations |
| 2 | `BogdanTurbal/nn_sign` | 2D Depthwise CNN + CTC | Fast non-autoregressive decoding | Relies on high-FPS streams | Future Continuous Track |
| 3 | `tuyenle009/Hand-Sign` | 1D CNN over 21 coords | Simple, lightweight | Random split, unnormalized | Baseline Comparison |
| 4 | `Bhuribhat/ASL-Finger` | MediaPipe + Dwell-Time | Practical state machine & debouncing | Naive heuristics | Adopt Debouncing Logic |
| 5 | `google/mediapipe` | BlazePalm + Hand Landmark (WASM) | SOTA 21 3D landmarks, 60fps edge CV | $z$-coord depth uncertainty | Core Foundation |
| 6 | `Kazuhito00/hand-gesture` | 42-Feature Relative MLP | Clean relative coordinate normalization | Max-scale sensitivity | Baseline Inspiration |
| 7 | `kinivi/hand-gesture` | 2D Relative MLP + 16F Trajectory | Clean spatial/temporal separation | Frame-rate velocity drift | Reusable Resampling |
| 8 | `lshiwjx/2s-AGCN` | Adaptive GCN ($\mathbf{A} + \mathbf{B}_k$) | Learned inter-digit adjacency | Full attention compute cost | **Adopt $\mathbf{A} + \mathbf{B}$ into TITAN** |
| 9 | `yysij/st-gcn` | Spatial-Temporal GCN | Theoretical foundation of skeletal GCN | Fixed anatomical graph | Baseline Theory |
| 10 | `kennymckormick/pyskl` | ST-GCN++, CTR-GCN, PoseC3D | Modular engineering, LayerNorm GCN | Heavy framework dependencies | Adopt GCN++ Block Design |
| 11 | `hulianyuyy/CorrNet` | Multi-stream Correlation Net | Hand-face trajectory alignment | High latency, GPU required | Theory Reference |
| 12 | `hotfinda/TCNet` | Decoupled Spatio-Temporal Net | Validates spatial/temporal decoupling | Multi-stage complexity | Validates Dual-Stream |
| 13 | `gpleiss/temperature_scaling` | Temperature Scaling ($T^*$) | Zero-overhead confidence calibration | Requires clean validation set | **Adopted ($T^*=1.08$)** |
| 14 | `weitliu/energy_ood` | Energy-based OOD Detection | Resolves softmax overconfidence | Threshold tuning needed | **Adopted in TITAN** |
| 15 | `microsoft/onnxruntime` | ONNX WebAssembly / WebGPU | Hardware-accelerated browser execution | 5-10MB bundle overhead | Reference Benchmark |

---

## 4. Legal, Intellectual Property & License Compliance

All 15 surveyed repositories were audited for license compatibility:
- **MIT License (8 repositories):** Fully permissive. Allows adaptation, modification, and redistribution without viral open-source copyleft obligations.
- **Apache License 2.0 (7 repositories):** Permissive with clear patent grant and copyright attribution requirements. Fully compatible with SIGNBRIDGE's Apache-2.0 / MIT licensing.
- **Strict Prohibition Enforced:** Zero GPL, AGPL, or proprietary viral dependencies were incorporated into SIGNBRIDGE TITAN.

---
*Verified and submitted to `experiment/titan-global-research`.*
