# PER-CLASS VALIDATION CHECKLIST (26 ASL LETTERS)
**System:** SIGNBRIDGE ASL Recognition Pipeline  
**Document Code:** `CHK-ALPHABET-26-001`  
**Engineer:** Senior QA & Computer Vision Validation Engineer  
**Date:** 2026-11-15 04:05:42 UTC  
**Scope:** Classes A–Z with Priority Focus on the 14 Zero-Human Classes  

---

## 1. High-Priority Classes (14 Missing Genuine-Human Capture Classes)

These 14 classes currently possess zero genuine physical human validation samples. Under zero-tolerance acceptance policy, **production sign-off is blocked until these classes are validated with physical human subjects**.

| Letter | Modality | Key Anatomical Criteria | Target Active Model Engine | Primary Hard-Negative Confusion Risks | Physical Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **G** | Static | Index extended horizontally forward; thumb parallel; middle/ring/pinky fully curled. | Multi-Signer Static MLP | `H` (2 fingers), `Q` (pointing down), `P` (downward angle) | **PENDING HUMAN COHORT** |
| **H** | Static | Index and middle fingers extended together horizontally; thumb folded across ring finger. | Multi-Signer Static MLP | `G` (1 finger), `U` (vertical orientation), `N` (fingers over thumb) | **PENDING HUMAN COHORT** |
| **I** | Static | Pinky finger extended vertically; index/middle/ring curled into palm; thumb over fingers. | Multi-Signer Static MLP | `J` (static vs dynamic hook), `Y` (thumb extended), `D` (index vs pinky) | **PENDING HUMAN COHORT** |
| **J** | **Dynamic** | Pinky extended; downward stroke carving a curved upward hook; speed $\ge 0.15	ext{ span/s}$. | Kinivi-Adapted Point History Engine | `I` (stationary pinky), `Z` (index stroke), accidental tremor | **PENDING HUMAN COHORT** |
| **K** | Static | Index extended vertically; middle angled forward $pprox 45^\circ$; thumb tip resting on middle PIP. | Multi-Signer Static MLP | `V` (middle vertical), `P` (angled down), `D` (middle curled) | **PENDING HUMAN COHORT** |
| **M** | Static | Thumb tucked under index, middle, and ring fingers; three knuckle humps visible. | Multi-Signer Static MLP | `N` (thumb under 2 fingers), `T` (thumb under 1), `S` (thumb in front), `A` (thumb beside) | **PENDING HUMAN COHORT** |
| **N** | Static | Thumb tucked under index and middle fingers only; two knuckle humps visible. | Multi-Signer Static MLP | `M` (thumb under 3 fingers), `T` (thumb under 1), `S` (thumb across), `A` | **PENDING HUMAN COHORT** |
| **P** | Static | Hand oriented downward $pprox 90^\circ$; index extended horizontally; middle angled down; thumb on PIP. | Multi-Signer Static MLP | `K` (vertical orientation), `Q` (downward G), `D` | **PENDING HUMAN COHORT** |
| **Q** | Static | Hand oriented downward; index and thumb extended in pinch downward; other fingers curled. | Multi-Signer Static MLP | `G` (horizontal orientation), `P` (middle extended), `O` | **PENDING HUMAN COHORT** |
| **S** | Static | All fingers curled tightly into a fist; thumb wrapped firmly across the front of all fingers. | Multi-Signer Static MLP | `A` (thumb alongside index), `E` (fingers resting on thumb), `T`, `M`, `N` | **PENDING HUMAN COHORT** |
| **T** | Static | Thumb tucked between index and middle fingers; index curled over thumb; other fingers curled. | Multi-Signer Static MLP | `A` (thumb beside), `S` (thumb in front), `N` (under 2 fingers), `M` (under 3) | **PENDING HUMAN COHORT** |
| **W** | Static | Index, middle, and ring fingers extended and spread vertically; thumb holds down pinky. | Multi-Signer Static MLP | `B` (4 fingers closed), `V` (2 fingers), `F` (thumb-index circle) | **PENDING HUMAN COHORT** |
| **X** | Static | Index finger hooked at DIP/PIP joint; thumb resting across middle; other fingers curled. | Multi-Signer Static MLP | `D` (straight index), `S` (closed fist), `E` (all curled) | **PENDING HUMAN COHORT** |
| **Z** | **Dynamic** | Index extended; traces 3-stroke zigzag in air: top horizontal, diagonal down-left, bottom horizontal. | Kinivi-Adapted Point History Engine | `D` (stationary index), `J` (pinky hook), random pointing | **PENDING HUMAN COHORT** |

---

## 2. Baseline Classes (12 Classes with Existing Benchmark Samples)

| Letter | Modality | Key Anatomical Criteria | Target Active Model Engine | Primary Hard-Negative Confusion Risks | Physical Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Static | Compact fist; all 4 fingers curled; thumb resting upright against lateral edge of index MCP. | Letter A Frozen Specialist + Gate | `S`, `E`, `T`, `M`, `N` | **FROZEN CANDIDATE RE-VERIFICATION** |
| **B** | Static | Four fingers extended vertically and pressed flat; thumb folded across palm. | Multi-Signer Static MLP | `W`, `4`, `D` | **PENDING HUMAN COHORT RE-CHECK** |
| **C** | Static | All fingers and thumb curved into an open 'C' shape matching palm profile. | Multi-Signer Static MLP | `O`, `E`, `S` | **PENDING HUMAN COHORT RE-CHECK** |
| **D** | Static | Index extended vertically; middle, ring, pinky curled touching thumb tip forming circle. | Multi-Signer Static MLP | `1`, `X`, `B`, `K` | **PENDING HUMAN COHORT RE-CHECK** |
| **E** | Static | All fingertips curled down tightly resting against top edge of folded thumb. | Multi-Signer Static MLP | `S`, `A`, `C`, `O` | **PENDING HUMAN COHORT RE-CHECK** |
| **F** | Static | Index and thumb touch forming small circle; middle, ring, pinky extended and spread. | Multi-Signer Static MLP | `B`, `D`, `9` | **PENDING HUMAN COHORT RE-CHECK** |
| **L** | Static | Index extended vertically, thumb extended horizontally at $90^\circ$ forming 'L'. | Multi-Signer Static MLP | `D`, `G`, `1` | **PENDING HUMAN COHORT RE-CHECK** |
| **O** | Static | All fingertips curved down touching thumb tip forming a closed circle 'O'. | Multi-Signer Static MLP | `C`, `E`, `S` | **PENDING HUMAN COHORT RE-CHECK** |
| **R** | Static | Index and middle fingers extended vertically and crossed (middle behind index). | Multi-Signer Static MLP | `U`, `V`, `K` | **PENDING HUMAN COHORT RE-CHECK** |
| **U** | Static | Index and middle fingers extended vertically together; thumb folded over ring finger. | Multi-Signer Static MLP | `V` (fingers spread), `H` (horizontal), `R` (crossed) | **PENDING HUMAN COHORT RE-CHECK** |
| **V** | Static | Index and middle fingers extended vertically and spread in 'V'; thumb folded. | Multi-Signer Static MLP | `U` (fingers together), `K` (thumb on middle), `W` (3 fingers) | **PENDING HUMAN COHORT RE-CHECK** |
| **Y** | Static | Thumb and pinky extended out in 'horns' shape; index, middle, ring curled tightly. | Multi-Signer Static MLP | `I`, `J`, `Shaka` | **PENDING HUMAN COHORT RE-CHECK** |
