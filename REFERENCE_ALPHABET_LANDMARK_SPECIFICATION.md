# SIGNBRIDGE — Reference Alphabet Landmark Specification
**Grounded In:** Canonical ASL Reference Posters (`media_1794665487769_3958bd7e.jpg` & `media_1794665668049_010708ce.jpg`)  
**Standard:** 21-Landmark MediaPipe Hand Topology  
**Version:** 2.0.0-coordinate-ai  

---

## 1. Landmark Topology & Canonical Hand Reference

The ASL Alphabet is defined by exact anatomical handshapes categorized by palm orientation, finger extension status, thumb opposition, and flexion angles.

```
       [8] TIP          [12] TIP         [16] TIP         [20] TIP
        |                 |                |                |
       [7] DIP          [11] DIP         [15] DIP         [19] DIP
        |                 |                |                |
       [6] PIP          [10] PIP         [14] PIP         [18] PIP
        \                 |                |                /
   [4]   [5] MCP ------ [9] MCP -------- [13] MCP ------ [17] MCP
   TIP    \               |                |             /
    |      \              |                |            /
   [3] IP   \             |                |           /
    |        \            |                |          /
   [2] MCP    \-----------+----------------+---------/
    |                    /
   [1] CMC              /
    \                  /
     \                /
      \-- [0] WRIST -/
```

---

## 2. Complete 26-Letter Landmark Morphological Matrix

| Letter | Handshape Type | Extended Fingers | Thumb Configuration | Distinguishing Angle & Curl Metrics | Ambiguity Class | Disambiguation Rule |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Compact Fist | None (all curled) | Extended upright beside index PIP | Thumb tip $y < P_5.y$; Thumb curl ratio $> 1.15$; Finger curls $< 0.85$ | S, T, M, N | Thumb alongside index side, not crossing palm or tucked |
| **B** | Flat Palm | Index, Mid, Ring, Pinky | Tucked across palm over MCP joints | 4 fingers curl $> 1.85$; Thumb curl $< 0.75$; Inter-finger angles $< 0.12$ rad | 4, H | Fingers tightly joined together with thumb folded flat across palm |
| **C** | Curved Arc | All fingers semi-curled | Curved opposite fingers forming 'C' | Inter-fingertip distance (Thumb-Index) $\in [0.8, 1.4]$; Finger angles $\in [1.1, 1.6]$ | O, E | Palm concave with open gap between thumb and fingertips |
| **D** | Single Index | Index | Tip touches thumb tip forming loop | Index curl $> 1.90$; Mid/Ring/Pinky curls $< 0.80$; Thumb-Mid dist $< 0.35$ | 1, Z, G | Thumb touches middle/ring fingertips, index points straight up |
| **E** | Clenched Claw | None | Tucked tightly against curled tips | All 4 finger DIP angles $< 1.2$ rad; Fingertips contact palm; Thumb under tips | S, O, C | Tips curled tightly against base of palm with thumb tucked below |
| **F** | Circle Pinch | Mid, Ring, Pinky | Tip touches index tip forming circle | Index-Thumb tip dist $< 0.22$; Mid/Ring/Pinky curls $> 1.80$; Spread $> 0.30$ | 9, B, D | Index and thumb touch forming circular aperture; other 3 fingers fan out |
| **G** | Horizontal Pointer | Index | Extended parallel to index | Index extended horizontally; Thumb parallel; Gap $\in [0.25, 0.55]$ | H, P, Q | Single index pointing sideways with thumb parallel |
| **H** | Dual Pointer | Index, Middle | Tucked flat beneath middle finger | Index & Mid extended horizontally side-by-side; Ring/Pinky curled $< 0.80$ | G, U, V | Two fingers extended together horizontally pointing sideways |
| **I** | Single Pinky | Pinky | Crossed over folded fingers | Pinky curl $> 1.85$; Index/Mid/Ring curls $< 0.75$; Thumb folded over fingers | J, Y | Only pinky extended upright; static posture |
| **J** | Dynamic Hook | Pinky | Folded over fingers | Dynamic: Pinky traces curved downward hook trajectory ($\Delta y > 0.8 S$) | I, Z | Dynamic stroke displacement with pinky tip path length $> 0.70 S$ |
| **K** | V-Thumb Wedge | Index, Middle | Upright between Index & Mid MCP | Index vertical; Mid slightly tilted forward; Thumb tip touches middle PIP | V, P, 2 | Thumb tip wedged upright between index and middle fingers |
| **L** | Right Angle | Thumb, Index | Extended perpendicular to index | Angle(Thumb, Index) $\in [1.35, 1.75]$ rad; Mid/Ring/Pinky curled $< 0.75$ | G, D | Strict 90-degree angle between upright index and horizontal thumb |
| **M** | 3-Finger Tuck | None (all curled) | Tucked under Index, Mid, Ring tips | Thumb tip extends past Ring MCP; 3 finger knuckles raised | N, T, S, A | Thumb emerges between ring and pinky fingers |
| **N** | 2-Finger Tuck | None (all curled) | Tucked under Index, Mid tips | Thumb tip extends past Mid MCP; 2 finger knuckles raised | M, T, S, A | Thumb emerges between middle and ring fingers |
| **O** | Closed Circle | None | Tips of all 4 fingers touch thumb | Distance(all fingertips to Thumb tip) $< 0.30$; Ring cavity open | C, E | All fingertips meet thumb tip creating closed oval aperture |
| **P** | Downward K | Index, Middle | Wedged between Index and Mid | Same as K but palm tilted downward ($\mathbf{v}_{	ext{long}} \cdot \mathbf{y} > 0.70$) | K, Q | Downward-pointing orientation of the K handshape |
| **Q** | Downward G | Index, Thumb | Extended parallel to index | Same as G but pointing downward ($\mathbf{v}_{	ext{long}} \cdot \mathbf{y} > 0.70$) | G, P | Downward-pointing orientation of the G handshape |
| **R** | Crossed Fingers | Index, Middle | Tucked over ring finger | Index and Mid crossed: $P_8.x > P_{12}.x$ (right hand); Ring/Pinky curled | U, V | Index and middle fingers entwined/crossed over each other |
| **S** | Wrapped Fist | None (all curled) | Wrapped across front of all 4 fingers | Thumb crosses fingers at PIP joints: $P_4.x \in [P_6.x, P_{14}.x]$ | A, E, T, M | Thumb rests over the middle of curled fingers (not to side) |
| **T** | Single Tuck | None (all curled) | Tucked between Index and Middle | Thumb tip emerges between index and middle finger knuckles | A, S, M, N | Only one finger knuckle rests over the thumb |
| **U** | Paired Upright | Index, Middle | Folded over ring finger | Index and Mid extended upright touching: Dist($P_8, P_{12}$) $< 0.25$ | V, H, R | Index and middle held vertically and tightly pressed together |
| **V** | Separated V | Index, Middle | Folded over ring finger | Index and Mid extended upright separated: Dist($P_8, P_{12}$) $> 0.50$ | U, K, 2 | Index and middle held vertically in distinct 'V' spread |
| **W** | Triple Upright | Index, Mid, Ring | Tucked over pinky tip | 3 fingers extended upright splayed; Pinky curled contacting thumb | 3, V, B | Index, middle, and ring fingers upright and separated |
| **X** | Hooked Index | Index (Hooked) | Tucked beside middle finger | Index PIP angle $< 1.4$ rad, DIP angle $< 1.3$ rad; Mid/Ring/Pinky curled | D, G | Index finger bent into a distinctive hooked shape |
| **Y** | Shaka / Horns | Thumb, Pinky | Extended outward | Thumb curl $> 1.4$, Pinky curl $> 1.5$; Index/Mid/Ring curled $< 0.75$ | I, J | Thumb and pinky extended wide; middle 3 fingers curled |
| **Z** | Dynamic Zig-Zag | Index | Tucked over middle finger | Dynamic: Index traces horizontal zig-zag stroke ($\Delta x, \Delta y, \Delta x$) | D, 1, J | Dynamic stroke displacement with index tip path length $> 0.70 S$ |

---

## 3. High-Confusion Disambiguation Clusters

### 3.1 The Fist Cluster: {A, S, T, M, N}
- **A:** Thumb extended vertically along side of index finger.
- **S:** Thumb wrapped across front of all four fingers.
- **T:** Thumb tucked under index finger only (emerging between index and middle).
- **M:** Thumb tucked under three fingers (emerging between ring and pinky).
- **N:** Thumb tucked under two fingers (emerging between middle and ring).

### 3.2 The Upright Pair Cluster: {U, V, R, K}
- **U:** Index and Middle extended upright touching ($D(8, 12) < 0.25 S$).
- **V:** Index and Middle extended upright separated ($D(8, 12) > 0.50 S$).
- **R:** Index and Middle crossed over each other ($P_8.x > P_{12}.x$).
- **K:** Middle tilted forward, thumb upright between index and middle.
