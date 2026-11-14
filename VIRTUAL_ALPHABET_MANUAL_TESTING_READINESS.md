# SIGNBRIDGE — MANUAL TESTING READINESS PROTOCOL

**Target Workflow:** Live camera manual testing and physical cohort data collection.  
**System State:** Tested, hardened, verified, and ready for human operator interaction.  

---

## 1. Operator Environment Pre-Flight Checklist

1. **Lighting & Contrast:**
   - Ensure uniform, indirect frontal lighting on hands.
   - Avoid strong backlighting or direct sunlight causing camera sensor blooming.
2. **Camera Placement:**
   - Position webcam at eye level, roughly 0.6m to 1.0m from signer.
   - Hand must occupy approximately 15% to 25% of the camera frame height.
   - Maintain hand within the central 70% zone of the sensor view to avoid lens edge distortion.
3. **ZeroLeak Data Collector Operation:**
   - Navigate to `/` in the frontend application.
   - Select target alphabet class using the letter selector pills (A through Z).
   - Review the displayed anatomical prompt and reference crop (`/references/crops/<Letter>.png`).
   - For static signs (A-I, K-Y): Hold pose steady until HUD confirms detection.
   - For dynamic signs (J, Z): Position starting handshape ('I' for J; 'D' for Z), click 'Record', execute the stroke smoothly, and verify the progress bar completes.

---

## 2. Critical Operator Warnings for Confusable Letter Groups

- **Group C / O / Y:** Keep palm squared toward the camera (within 15° of frontal normal). Do not execute C or O at severe side angles, as lateral perspective causes optical foreshortening resembling Y.
- **Group M / N / T:** Elevate knuckles slightly toward the camera (10°–20° forward tilt) so the thumb protrusion remains visible beneath the index and middle fingers.
- **Group A / S / E:** Thumb must be held upright along the index finger knuckle for A. Folding thumb across the palm triggers 'E' or 'S'.