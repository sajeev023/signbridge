# SIGNBRIDGE TITAN: ASL ALPHABET VISUAL REFERENCE ANALYSIS
## Forensic Deconstruction of the Gérard Aflague American Sign Language Reference Chart

**Document ID:** `SB-TITAN-ASL-REF-01`  
**System Class:** Visual Reference Analysis & Biomechanical Specification  
**Active Research Branch:** `experiment/signbridge-titan`  
**Source Image:** `backend/titan/references/asl_alphabet_chart.png`  
**Date:** November 13, 2026  
**Status:** VALIDATED & CROPPED TO 26 CANONICAL REFERENCE PANELS  

---

## 1. VISUAL REFERENCE PROVENANCE & GLOBAL METRICS

The primary visual reference supplied for the ASL expansion is an authoritative educational poster from the **Gérard Aflague Collection** depicting the American Sign Language manual alphabet ($A$ through $Z$):

- **Image Dimensions:** $772 \times 1024\ \text{pixels}$ (RGB, 24-bit depth).
- **Physical Structure:** Outer black containment border spanning $x \in [5, 767]$, $y \in [7, 1017]$.
- **Header Section:** $y \in [7, 147]$ displaying title `"AMERICAN SIGN LANGUAGE ALPHABET"`.
- **Footer Section:** $y \in [948, 1017]$ displaying `"GÉRARD AFLAGUE COLLECTION"`.
- **Panel Grid Structure:** 5 rows containing exactly 26 letter panels:
  - **Rows 1 to 4:** 5 columns each ($5 \times 4 = 20$ panels), cell width $\approx 152.4\ \text{px}$.
  - **Row 5:** 6 columns ($1 \times 6 = 6$ panels), cell width $\approx 127.0\ \text{px}$.
- **Extracted Reference Crops:** Staged to [backend/titan/datasets/asl_alphabet/reference/crops/](file:///C:/Users/ADMIN/.gemini/antigravity/scratch/signbridge/backend/titan/datasets/asl_alphabet/reference/crops/) and mirrored to client public assets.

```
+-------------------------------------------------------------------------+
|                  AMERICAN SIGN LANGUAGE ALPHABET                        |
+-------------+-------------+-------------+-------------+-----------------+
|   Aa (0,0)  |   Bb (0,1)  |   Cc (0,2)  |   Dd (0,3)  |     Ee (0,4)    |
+-------------+-------------+-------------+-------------+-----------------+
|   Ff (1,0)  |   Gg (1,1)  |   Hh (1,2)  |   Ii (1,3)  |   Jj (1,4) [->] |
+-------------+-------------+-------------+-------------+-----------------+
|   Kk (2,0)  |   Ll (2,1)  |   Mm (2,2)  |   Nn (2,3)  |     Oo (2,4)    |
+-------------+-------------+-------------+-------------+-----------------+
|   Pp (3,0)  |   Qq (3,1)  |   Rr (3,2)  |   Ss (3,3)  |     Tt (3,4)    |
+---------+---+-----+-------+-----+-------+-----+-------+-----+---+-------+
| Uu (4,0)| Vv (4,1)|   Ww (4,2)  |   Xx (4,3)  |   Yy (4,4)  |Zz (4,5)[~]|
+---------+---------+-------------+-------------+-------------+-----------+
|                        GÉRARD AFLAGUE COLLECTION                        |
+-------------------------------------------------------------------------+
```

---

## 2. 26-LETTER REFERENCE FORENSIC SPECIFICATION

Each panel was audited across 8 anatomical axes: label, bounding box, wrist orientation, finger flexion, thumb placement, contacts, visible dynamics, and visual ambiguity.

### Row 1: Panels A through E

1. **Panel A (`Aa`)**
   - **Bounding Box:** $x \in [5, 157]$, $y \in [147, 310]$ ($152 \times 163\ \text{px}$).
   - **Hand Orientation:** Upright, anterior palm facing camera.
   - **Finger Configuration:** Four fingers curled into a clenched fist; PIP and DIP joints tightly flexed against the palm.
   - **Thumb Placement:** Extended upright along the radial lateral margin of the index finger. Thumb tip aligns with index PIP knuckle. Thumb is strictly not folded over the anterior face of fingers.
   - **Dynamic Indicators:** None (static).
   - **Ambiguities:** Readily confused with **S** (thumb crossed over anterior fist) and **E** (fingertips resting on thumb).

2. **Panel B (`Bb`)**
   - **Bounding Box:** $x \in [157, 310]$, $y \in [147, 310]$ ($153 \times 163\ \text{px}$).
   - **Hand Orientation:** Upright, anterior palm facing camera.
   - **Finger Configuration:** Four fingers fully extended vertically, held adducted (touching side-by-side without inter-digital gaps).
   - **Thumb Placement:** Flexed across the palm surface, thumb pad resting horizontally over the MCP flexion crease.
   - **Dynamic Indicators:** None (static).
   - **Ambiguities:** Confusable with numeric '4' (thumb abducted) or open palm if fingers splay.

3. **Panel C (`Cc`)**
   - **Bounding Box:** $x \in [310, 462]$, $y \in [147, 310]$ ($152 \times 163\ \text{px}$).
   - **Hand Orientation:** Lateral profile facing left, wrist upright.
   - **Finger Configuration:** All 4 fingers extended together in an arched curve (partial flexion at MCP, PIP, DIP).
   - **Thumb Placement:** Fully abducted and curved upward opposing the fingers, forming an open circular aperture.
   - **Dynamic Indicators:** None (static).
   - **Ambiguities:** Confusable with **O** (closed aperture where tips contact thumb) and **E** if viewed from anterior angle.

4. **Panel D (`Dd`)**
   - **Bounding Box:** $x \in [462, 614]$, $y \in [147, 310]$ ($152 \times 163\ \text{px}$).
   - **Hand Orientation:** Upright, quarter-turned anterior-medial.
   - **Finger Configuration:** Index finger extended straight vertically. Middle, ring, and pinky curled into a closed circle.
   - **Thumb Placement:** Arched up to contact the pad of the curled middle finger (and index base), enclosing a circular loop.
   - **Dynamic Indicators:** None (static).
   - **Ambiguities:** Differs from **F** (index touches thumb, other 3 extended) and **Z** (static start posture).

5. **Panel E (`Ee`)**
   - **Bounding Box:** $x \in [614, 767]$, $y \in [147, 310]$ ($153 \times 163\ \text{px}$).
   - **Hand Orientation:** Upright, palm anterior.
   - **Finger Configuration:** All 4 fingers curled down tightly at PIP and DIP joints into a flexed claw.
   - **Thumb Placement:** Tucked horizontally flat underneath all 4 fingertips; thumb tip rests near pinky/ring base. Fingertips press onto dorsal thumb surface.
   - **Dynamic Indicators:** None (static).
   - **Ambiguities:** High risk of confusion with **S** and **A** if thumb tucking is obscured by camera angle.

---

### Row 2: Panels F through J

6. **Panel F (`Ff`)**
   - **Bounding Box:** $x \in [5, 157]$, $y \in [310, 465]$ ($152 \times 155\ \text{px}$).
   - **Hand Orientation:** Upright, palm anterior.
   - **Finger Configuration:** Middle, ring, and pinky fingers extended straight up and slightly spread (abducted). Index finger curved downward.
   - **Thumb Placement:** Arched up to touch index fingertip pad, forming an enclosed circular aperture ("OK" sign posture).
   - **Dynamic Indicators:** None (static).
   - **Ambiguities:** Inversion of **D**.

7. **Panel G (`Gg`)**
   - **Bounding Box:** $x \in [157, 310]$, $y \in [310, 465]$ ($153 \times 155\ \text{px}$).
   - **Hand Orientation:** Horizontal, hand turned sideways facing left, wrist horizontally extended.
   - **Finger Configuration:** Index finger extended horizontally forward pointing left. Middle, ring, pinky curled into fist.
   - **Thumb Placement:** Extended horizontally parallel to index finger with a small gap ($\approx 1.5\ \text{cm}$) between them.
   - **Dynamic Indicators:** None (static).
   - **Ambiguities:** Confusable with **H** (two fingers extended) and **Q** (pointing downward).

8. **Panel H (`Hh`)**
   - **Bounding Box:** $x \in [310, 462]$, $y \in [310, 465]$ ($152 \times 155\ \text{px}$).
   - **Hand Orientation:** Horizontal, hand turned sideways facing left.
   - **Finger Configuration:** Index and middle fingers extended straight horizontally forward together, touching side-by-side (adducted). Ring and pinky curled into fist.
   - **Thumb Placement:** Tucked flat along lateral palm beneath middle finger base.
   - **Dynamic Indicators:** None (static).
   - **Ambiguities:** Confusable with **G** (1 finger) and **U** (rotated upright).

9. **Panel I (`Ii`)**
   - **Bounding Box:** $x \in [462, 614]$, $y \in [310, 465]$ ($152 \times 155\ \text{px}$).
   - **Hand Orientation:** Upright, palm anterior.
   - **Finger Configuration:** Pinky finger extended straight vertically. Index, middle, and ring tightly curled into fist.
   - **Thumb Placement:** Folded across the anterior surface of the curled index and middle fingers.
   - **Dynamic Indicators:** None (static).
   - **Ambiguities:** Identical to the starting handshape of **J**; static frames of **J** must not trigger **J** without trajectory evidence.

10. **Panel J (`Jj`) — DYNAMIC CLASS**
    - **Bounding Box:** $x \in [614, 767]$, $y \in [310, 465]$ ($153 \times 155\ \text{px}$).
    - **Hand Orientation:** Dynamic trajectory. The illustration captures the terminal phase of the movement with wrist ulnar deviation and supination.
    - **Finger Configuration:** Pinky extended, other fingers clenched into fist with thumb folded.
    - **Visible Movement Indicator:** A curved dashed arrow is clearly rendered to the left of the hand tracing a downward concave trajectory with an arrowhead pointing down and inward into a hook.
    - **Biomechanical Requirement:** Trajectory recognition must verify downward displacement of pinky tip followed by inward hook over $\ge 250\ \text{ms}$.

---

### Row 3: Panels K through O

11. **Panel K (`Kk`)**
    - **Bounding Box:** $x \in [5, 157]$, $y \in [465, 630]$ ($152 \times 165\ \text{px}$).
    - **Hand Orientation:** Upright, palm anterior.
    - **Finger Configuration:** Index extended straight up; middle finger extended and angled forward at $\approx 45^\circ$ pitch. Ring and pinky curled.
    - **Thumb Placement:** Extended upward between index and middle fingers, thumb pad contacting the inner joint of the middle finger.
    - **Ambiguities:** Confusable with **V** (both fingers coplanar) and **P** (inverted downward).

12. **Panel L (`Ll`)**
    - **Bounding Box:** $x \in [157, 310]$, $y \in [465, 630]$ ($153 \times 165\ \text{px}$).
    - **Hand Orientation:** Upright, palm anterior.
    - **Finger Configuration:** Index finger extended vertically; middle, ring, pinky curled into fist.
    - **Thumb Placement:** Extended horizontally outward at an exact $90^\circ$ right angle, forming a prominent `"L"`.
    - **Ambiguities:** Low confusion rate; highly stable geometric signature.

13. **Panel M (`Mm`)**
    - **Bounding Box:** $x \in [310, 462]$, $y \in [465, 630]$ ($152 \times 165\ \text{px}$).
    - **Hand Orientation:** Upright, palm anterior.
    - **Finger Configuration:** All 4 fingers curled down over the thumb.
    - **Thumb Placement:** Tucked under 3 fingers (index, middle, ring), thumb tip protruding slightly between ring and pinky.
    - **Ambiguities:** Extreme confusion target with **N** (2 fingers over thumb), **T** (1 finger over thumb), and **S** (0 fingers over thumb).

14. **Panel N (`Nn`)**
    - **Bounding Box:** $x \in [462, 614]$, $y \in [465, 630]$ ($152 \times 165\ \text{px}$).
    - **Hand Orientation:** Upright, palm anterior.
    - **Finger Configuration:** All 4 fingers curled down.
    - **Thumb Placement:** Tucked under 2 fingers (index, middle), thumb tip protruding between middle and ring fingers.
    - **Ambiguities:** Extreme confusion target with **M** and **T**.

15. **Panel O (`Oo`)**
    - **Bounding Box:** $x \in [614, 767]$, $y \in [465, 630]$ ($153 \times 165\ \text{px}$).
    - **Hand Orientation:** Upright, semi-lateral.
    - **Finger Configuration:** All 4 fingers curved smoothly downward into an arch.
    - **Thumb Placement:** Abducted and curved upward to contact all 4 fingertips, enclosing an oval `"O"`.
    - **Ambiguities:** Confusable with **C** (open gap) and **E** (clawed flat).

---

### Row 4: Panels P through T

16. **Panel P (`Pp`)**
    - **Bounding Box:** $x \in [5, 157]$, $y \in [630, 795]$ ($152 \times 165\ \text{px}$).
    - **Hand Orientation:** Downward inverted handshape, wrist flexed.
    - **Finger Configuration:** Same handshape as **K** (index forward, middle downward $90^\circ$, thumb on middle knuckle), but pointing downwards.
    - **Ambiguities:** Downward pitch separates **P** from **K**.

17. **Panel Q (`Qq`)**
    - **Bounding Box:** $x \in [157, 310]$, $y \in [630, 795]$ ($153 \times 165\ \text{px}$).
    - **Hand Orientation:** Downward inverted, pointing towards ground.
    - **Finger Configuration:** Index finger pointing down; thumb extended downward parallel with small gap.
    - **Ambiguities:** Inverted version of **G**.

18. **Panel R (`Rr`)**
    - **Bounding Box:** $x \in [310, 462]$, $y \in [630, 795]$ ($152 \times 165\ \text{px}$).
    - **Hand Orientation:** Upright, palm anterior.
    - **Finger Configuration:** Middle finger crossed over the dorsal (back) surface of the extended index finger. Ring and pinky curled into fist.
    - **Thumb Placement:** Folded across curled ring finger.
    - **Ambiguities:** Confusable with **U** if finger crossing is missed by optical landmark trackers.

19. **Panel S (`Ss`)**
    - **Bounding Box:** $x \in [462, 614]$, $y \in [630, 795]$ ($152 \times 165\ \text{px}$).
    - **Hand Orientation:** Upright, palm anterior.
    - **Finger Configuration:** All 4 fingers clenched into a fist.
    - **Thumb Placement:** Folded firmly across the anterior surface of the clenched index and middle fingers.
    - **Ambiguities:** Prime confusion target with **A** (thumb alongside) and **E** (fingertips resting on thumb).

20. **Panel T (`Tt`)**
    - **Bounding Box:** $x \in [614, 767]$, $y \in [630, 795]$ ($153 \times 165\ \text{px}$).
    - **Hand Orientation:** Upright, palm anterior.
    - **Finger Configuration:** All 4 fingers curled into fist.
    - **Thumb Placement:** Tucked under index finger only, protruding between index and middle fingers.
    - **Ambiguities:** Completes the **M / N / T** classification challenge.

---

### Row 5: Panels U through Z

21. **Panel U (`Uu`)**
    - **Bounding Box:** $x \in [5, 132]$, $y \in [795, 948]$ ($127 \times 153\ \text{px}$).
    - **Finger Configuration:** Index and middle fingers extended vertically straight up, touching side-by-side (adducted, zero inter-digit gap). Ring and pinky curled.
    - **Thumb Placement:** Folded across ring finger.
    - **Ambiguities:** Confusable with **V** (fingers spread) and **R** (crossed).

22. **Panel V (`Vv`)**
    - **Bounding Box:** $x \in [132, 259]$, $y \in [795, 948]$ ($127 \times 153\ \text{px}$).
    - **Finger Configuration:** Index and middle fingers extended vertically, spread apart in a distinct `"V"` peace shape (abducted, $\approx 30^\circ$ angle). Ring and pinky curled.
    - **Ambiguities:** Confusable with **U** (adducted) and **K** (thumb between).

23. **Panel W (`Ww`)**
    - **Bounding Box:** $x \in [259, 386]$, $y \in [795, 948]$ ($127 \times 153\ \text{px}$).
    - **Finger Configuration:** Index, middle, and ring fingers extended vertically and spread apart in a `"W"` shape. Pinky curled down.
    - **Thumb Placement:** Folded across pinky fingertip holding it in place.
    - **Ambiguities:** Confusable with **B** (4 fingers) and **V** (2 fingers).

24. **Panel X (`Xx`)**
    - **Bounding Box:** $x \in [386, 513]$, $y \in [795, 948]$ ($127 \times 153\ \text{px}$).
    - **Finger Configuration:** Index finger hooked at PIP and DIP joints into a curved pirate hook. Middle, ring, pinky curled into fist.
    - **Thumb Placement:** Folded across curled middle finger.
    - **Ambiguities:** Confusable with **D** (extended index) and **S** (fist).

25. **Panel Y (`Yy`)**
    - **Bounding Box:** $x \in [513, 640]$, $y \in [795, 948]$ ($127 \times 153\ \text{px}$).
    - **Finger Configuration:** Index, middle, and ring fingers clenched into fist. Thumb and pinky extended fully outward in opposite directions (shaka / phone sign).
    - **Ambiguities:** Highly distinct silhouette; low confusion.

26. **Panel Z (`Zz`) — DYNAMIC CLASS**
    - **Bounding Box:** $x \in [640, 767]$, $y \in [795, 948]$ ($127 \times 153\ \text{px}$).
    - **Hand Orientation:** Index finger extended pointing forward/upward as a stylus; other fingers clenched into fist.
    - **Visible Movement Indicator:** A prominent white dashed zigzag arrow with three sequential strokes:
      1. Stroke 1: Horizontal stroke from left to right.
      2. Stroke 2: Diagonal stroke down-and-left.
      3. Stroke 3: Horizontal stroke from left to right.
    - **Biomechanical Requirement:** Trajectory recognition must verify 3-stroke zigzag path of index tip over $\ge 300\ \text{ms}$. Static 'D' or '1' must not trigger 'Z'.

---

## 3. DECONSTRUCTION OF GRAPHIC SKELETON OVERLAYS

The poster features light blue structural overlays with circular joint markers. A rigorous forensic audit reveals that **these overlays are illustrative graphic annotations, NOT verified MediaPipe 21-point topology**:

1. **Non-Compliant Forearm Bones:**
   - Panels `G`, `H`, `J`, `P`, `Q`, `Xx`, `Zz` clearly show bone vectors extending down the forearm to a distal ulna/radius joint dot. MediaPipe HandLandmarker terminates at wrist Landmark 0; it does not provide forearm or elbow coordinates.
2. **Artificial Carpal Star Hub:**
   - The graphic overlay connects all 5 MCP joints directly to a single central carpal hub point in the palm center. This violates MediaPipe's anatomical graph topology, which routes through $[0,1]$, $[0,5]$, $[0,17]$ and transverse palm arch $[5,9]$, $[9,13]$, $[13,17]$.
3. **Planar Projection vs Optical Occlusion:**
   - On clenched fist poses (`A`, `E`, `M`, `N`, `S`, `T`), the graphic displays flat, evenly spaced grids that ignore 3D depth and optical joint occlusion experienced by monocular webcams.
4. **Mandatory Scientific Rule:**
   - The reference chart establishes the ground truth **class semantics and visual morphology**, but its graphic overlays must **never** be used as synthetic numerical training coordinates. Real training landmarks must originate from MediaPipe inference on multi-signer captures.
