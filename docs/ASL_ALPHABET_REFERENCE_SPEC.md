# SIGNBRIDGE — ASL Alphabet Reference Specification & Biomechanical Analysis
**Document ID:** `SPEC-ASL-ALPHABET-GENESIS-01`  
**Reference Visual:** American Sign Language Alphabet Chart (Gérard Aflague Collection)  
**Convention:** Single-Hand American Sign Language (ASL) Manual Alphabet  
**Target Engine:** SIGNBRIDGE Dual/Single Pipeline (Isolated from ISLRTC Bimanual)

---

## 1. Executive Summary & Domain Separation

This document establishes the canonical biomechanical and morphological reference specification for the 26 letters of the American Sign Language (ASL) manual alphabet, as depicted in the reference chart.

### Critical Pipeline Isolation Notice
- **ASL Single-Hand Alphabet**: Signs are formed using a **single dominant hand** in 3D camera space. The vocabulary comprises 24 static postures ($A\text{--}I, K\text{--}Y$) and 2 dynamic trajectory signs ($J, Z$), plus a designated resting/transition class `NEUTRAL`.
- **ISLRTC Bimanual Pipeline**: Indian Sign Language (ISLRTC) uses two-handed contacts and vowel-to-fingertip topology ($A, E, I, O, U$ mapped onto left-hand fingertips).
- **Separation Mandate**: The ASL single-hand model is strictly evaluated on 42-feature single-hand biomechanics and must **never** be co-mingled with or overwritten by ISLRTC two-hand weight matrices.

---

## 2. Anatomical Hand Landmark Convention (MediaPipe 21 Keypoints)

The SIGNBRIDGE recognition pipeline represents hand geometry using MediaPipe's canonical 21 3D landmarks:

```
                  8 (Index Tip)    12 (Middle Tip)   16 (Ring Tip)   20 (Pinky Tip)
                     |                |                 |               |
                  7 (Index DIP)    11 (Middle DIP)   15 (Ring DIP)   19 (Pinky DIP)
                     |                |                 |               |
                  6 (Index PIP)    10 (Middle PIP)   14 (Ring PIP)   18 (Pinky PIP)
  4 (Thumb Tip)      |                |                 |               |
     |            5 (Index MCP)    9 (Middle MCP)    13 (Ring MCP)   17 (Pinky MCP)
  3 (Thumb IP)       \                |                 /               /
     |                \               |                /               /
  2 (Thumb MCP)        \              |               /               /
     \                  ------------------------------               /
      1 (Thumb CMC)                   |                             /
           \                          |                            /
            ----------------------- 0 (Wrist) ---------------------
```

---

## 3. Comprehensive Morphological Specification (Letters A–Z)

| Letter | Sign Type | Primary Kinematic Configuration | Thumb Placement & Relationship | Finger Extension State | Palm Orientation | Key Biomechanical Invariants |
| :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| **A** | Static | Fist with vertical thumb | Thumb extended along radial margin of index MCP, resting upright | Index, middle, ring, pinky fully flexed into palm | Facing forward / observer | High fist compactness ($>0.85$); thumb $y$-coordinate elevated; thumb-to-index MCP distance low. |
| **B** | Static | 4 extended vertical fingers | Thumb fully adducted & flexed across palm at base of fingers | Index, middle, ring, pinky fully extended upright, zero lateral spread | Facing forward | Zero finger spread ($\cos \approx 1.0$); 4 high extension ratios ($>1.0$); thumb tucked. |
| **C** | Static | Curved C-arc | Thumb abducted and curved forward opposing fingertips | All fingers flexed at MCP and PIP into continuous arc | Facing medial / lateral | Semi-curled fingers; thumb tip separated from index tip by open gap ($\approx 0.4\text{--}0.6$ palm scale). |
| **D** | Static | Upright index with circular loop | Thumb tip pressed against middle fingertip (and ring/pinky tips) | Index fully extended upright; middle, ring, pinky curled into loop | Facing forward | Index extension high ($>1.1$); middle/ring/pinky extension low; thumb-to-middle tip distance $<0.15$. |
| **E** | Static | Tight fist with horizontal thumb tuck | Thumb flexed tightly horizontally across palm underneath all 4 bent fingertips | All 4 fingers curled down tightly, tips pressing near palm base | Facing forward | Maximum fist compactness; low tip-to-wrist distances; thumb tucked below fingertips ($y < y_{\text{tips}}$). |
| **F** | Static | 3 fingers upright, index-thumb loop | Thumb tip touching index fingertip forming an "O" loop aperture | Middle, ring, pinky extended upright and abducted; index curled into loop | Facing forward | Loop closure metric (thumb-to-index tip $<0.15$); middle, ring, pinky extensions $>1.0$; wide inter-finger spread. |
| **G** | Static | Horizontal pointing pinch | Thumb extended horizontally parallel to index, small gap | Index extended horizontally across body; middle, ring, pinky curled into fist | Facing signer / medial | Horizontal canonical index direction ($|\text{idx\_dir\_x}| \gg |\text{idx\_dir\_y}|$, $y \approx 0$); thumb-index parallel. |
| **H** | Static | 2 horizontal extended fingers | Thumb tucked alongside middle finger | Index and middle extended together horizontally; ring and pinky curled | Facing signer / medial | Horizontal direction ($|\text{idx\_dir\_x}| \gg |\text{idx\_dir\_y}|$); index-middle parallel ($\cos > 0.95$). |
| **I** | Static | Upright little finger | Thumb folded across curled index, middle, ring fingers | Pinky fully extended vertically upright; index, middle, ring curled tightly | Facing forward | Pinky extension $>1.1$; index/middle/ring curled ($<0.0$); thumb folded over fingers. |
| **J** | **Dynamic** | **Motion-dependent J-hook** | Same handshape as 'I' (pinky upright, others curled) | Pinky extended; traces downward vertical path then curves into upward hook | Forward $\to$ Medial rotation | **Temporal trajectory**: $\Delta y_{\text{down}} > 0.08$, $\Delta x_{\text{hook}} < -0.04$, $\Delta y_{\text{up}} > 0.02$ within 1200ms window. |
| **K** | Static | V-shape with thumb in cleft | Thumb tip placed in cleft between index and middle MCPs/PIPs | Index extended upright; middle extended forward/upward at $\sim 45^\circ$; ring/pinky curled | Facing forward | High index extension; middle tilted forward ($z$-offset); thumb positioned between index and middle joints. |
| **L** | Static | Right-angle "L" shape | Thumb extended horizontally at $90^\circ$ to index finger | Index extended vertically upright; middle, ring, pinky curled into palm | Facing forward | Perpendicular angle between thumb and index ($\cos \approx 0.0\text{--}0.2$, angle $\approx 90^\circ$); index upright. |
| **M** | Static | 3-finger drape over thumb | Thumb folded under index, middle, ring fingers, tip peeking between ring and pinky | Index, middle, ring bent down over thumb; pinky curled into palm | Facing forward | Thumb tip located between ring and pinky MCPs; fingers draped forward over thumb. |
| **N** | Static | 2-finger drape over thumb | Thumb folded under index and middle fingers, tip peeking between middle and ring | Index and middle bent down over thumb; ring and pinky curled into palm | Facing forward | Thumb tip located between middle and ring MCPs; 2 fingers draped over thumb. |
| **O** | Static | Full circle O-aperture | Thumb tip touching all fingertips or index/middle tips in closed circle | All fingers curved forward to meet thumb tip in circular loop | Facing forward / medial | Low loop closure distance across all fingertips; circular volume; no extended fingers. |
| **P** | Static | Inverted K pointing down | Thumb resting on middle finger joint | Index extended downward/forward; middle extended downward; ring/pinky curled | Facing downward / forward | Canonical index direction pointing downward ($y > 0$ in inverted camera frame); K-geometry inverted. |
| **Q** | Static | Inverted G pointing down | Thumb and index pointing downward in pinch | Index extended downward; thumb pointing down; middle, ring, pinky curled | Facing downward / signer | Canonical index direction downward ($y > 0$); thumb-to-index pinch aperture. |
| **R** | Static | Crossed index and middle fingers | Thumb folded across ring and pinky | Index and middle extended upright and crossed over each other; ring/pinky curled | Facing forward | Distinct crossing metric: lateral crossing $\Delta x_{\text{tips}} / \Delta x_{\text{MCPs}} < 0$; 3D depth crossing $\Delta z \neq 0$. |
| **S** | Static | Tight fist with thumb across front | Thumb flexed tightly over the anterior surface of curled index/middle/ring | All 4 fingers curled into tight fist with thumb crossing in front of knuckles | Facing forward | Fist compactness $>0.90$; thumb rests in front of index/middle MCPs (distinct from 'A' where thumb is lateral). |
| **T** | Static | Thumb between index and middle | Thumb inserted between index and middle fingers under curled index | Index curled over thumb; middle, ring, pinky curled into palm | Facing forward | Thumb tip emerges between index and middle MCPs; index curled around thumb. |
| **U** | Static | 2 fingers upright together | Thumb folded over ring and pinky fingers | Index and middle extended upright and held tightly together (adducted); ring/pinky curled | Facing forward | Index and middle extension $>1.1$; inter-finger distance $<0.15$; parallel alignment $\cos > 0.98$. |
| **V** | Static | 2 fingers upright spread (V) | Thumb folded over ring and pinky fingers | Index and middle extended upright and spread apart (abducted); ring/pinky curled | Facing forward | Index and middle extension $>1.1$; inter-finger spread angle $>20^\circ$ ($\cos < 0.94$); distance $>0.35$. |
| **W** | Static | 3 fingers upright spread (W) | Thumb holds pinky finger tip down against palm | Index, middle, ring extended upright and spread apart; pinky curled | Facing forward | Index, middle, ring extension $>1.1$; pinky curled ($<0.0$); thumb holding pinky. |
| **X** | Static | Crooked index finger hook | Thumb tucked alongside or across curled fingers | Index finger flexed into hook shape (PIP flexed $\approx 90^\circ$); middle/ring/pinky curled | Facing forward / lateral | Index PIP angle flexed ($\approx 90^\circ$); tip-to-MCP distance lower than straight extension; others curled. |
| **Y** | Static | Extended thumb and pinky | Thumb fully extended laterally; pinky fully extended laterally | Index, middle, ring curled tightly into palm | Facing forward | Wide $Y$-span (thumb-to-pinky tip distance $>1.4$ palm scale); middle three fingers fully curled. |
| **Z** | **Dynamic** | **Index finger 3-stroke zigzag** | Thumb tucked over curled middle, ring, pinky fingers | Index finger extended straight forward pointing like stylus; others curled | Facing forward / target | **Temporal trajectory**: 3-stroke path (horizontal right $\to$ diagonal down-left $\to$ horizontal right). |

---

## 4. Distinction Between Similar and Easily Confused Signs

1. **A vs S vs E**:
   - `A`: Thumb is **alongside** the index MCP/PIP pointing upright.
   - `S`: Thumb is **crossed over the front** of the curled index/middle fingers.
   - `E`: All fingertips rest on knuckles; thumb is **tucked underneath** all four fingers across the lower palm.
2. **U vs V vs R vs D**:
   - `D`: ONLY index finger is upright; middle, ring, pinky curled touching thumb.
   - `U`: Index and middle are both upright and **touching** (parallel, no spread).
   - `V`: Index and middle are both upright and **spread apart** in a V.
   - `R`: Index and middle are both upright and **crossed over each other**.
3. **M vs N vs T vs A**:
   - `M`: Thumb tucks under **three** fingers (emerges between ring and pinky).
   - `N`: Thumb tucks under **two** fingers (emerges between middle and ring).
   - `T`: Thumb tucks under **one** finger (emerges between index and middle).
   - `A`: Thumb is outside the fist, alongside the index finger.
4. **G vs H vs L**:
   - `L`: Upright index, horizontal thumb ($90^\circ$ angle, vertical orientation).
   - `G`: Horizontal index, horizontal thumb pointing forward/side.
   - `H`: Horizontal index AND middle fingers together.

---

## 5. Architectural Implementation Decision: Static vs Dynamic

- **Static Signs (24 classes + NEUTRAL = 25 total classes)**:
  - Formulated as an instantaneous 42-feature spatial vector evaluated by the continuous biomechanical MLP.
  - Latency: $< 0.1$ ms per frame.
- **Dynamic Signs (J and Z)**:
  - Formulated as spatio-temporal trajectories of the dominant fingertip (pinky for J, index for Z) tracked across an adaptive 1200ms rolling window.
  - Dynamically triggered and prioritized when tracking confidence is verified.

