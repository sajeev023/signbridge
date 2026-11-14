# SIGNBRIDGE — ASL LETTER A SPECIALIST
## Participant Allocation & Data Integrity Governance Report

**Branch**: `experiment/letter-a-mastery`  
**Evaluation Scope**: Physical Human Validation Cohort (`Signer_P26` through `Signer_P35`)  
**Status**: AUDITED & INTEGRITY PROTOCOL ACTIVE  

---

### 1. Participant Cohort Allocation & Partition Table

To guarantee strict scientific independence and zero data leakage:
- Historical training signers: `Signer_1`, `Signer_2`
- Historical validation signer: `Signer_3`
- Historical held-out test signer: `Signer_4`
- Preliminary benchmark signers: `Signer_5`, `Signer_6`
- **Field Validation Cohort**: Exclusively allocated to `Signer_P26` through `Signer_P35`.

| Participant ID | Cohort Category | Mandatory Dataset Split Tag | Quota Target (A) | Quota Target (Hard Neg) | Session ID Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Signer_P26` | Final Evaluation | `physical_human_validation` | $\ge 30$ | $\ge 40$ (E,S,M,N,T) | `SES_P26_<timestamp>` |
| `Signer_P27` | Final Evaluation | `physical_human_validation` | $\ge 30$ | $\ge 40$ (E,S,M,N,T) | `SES_P27_<timestamp>` |
| `Signer_P28` | Final Evaluation | `physical_human_validation` | $\ge 30$ | $\ge 40$ (E,S,M,N,T) | `SES_P28_<timestamp>` |
| `Signer_P29` | Final Evaluation | `physical_human_validation` | $\ge 30$ | $\ge 40$ (E,S,M,N,T) | `SES_P29_<timestamp>` |
| `Signer_P30` | Final Evaluation | `physical_human_validation` | $\ge 30$ | $\ge 40$ (E,S,M,N,T) | `SES_P30_<timestamp>` |
| `Signer_P31` | Final Evaluation | `physical_human_validation` | $\ge 30$ | $\ge 40$ (E,S,M,N,T) | `SES_P31_<timestamp>` |
| `Signer_P32` | Final Evaluation | `physical_human_validation` | $\ge 30$ | $\ge 40$ (E,S,M,N,T) | `SES_P32_<timestamp>` |
| `Signer_P33` | Final Evaluation | `physical_human_validation` | $\ge 30$ | $\ge 40$ (E,S,M,N,T) | `SES_P33_<timestamp>` |
| `Signer_P34` | Final Evaluation | `physical_human_validation` | $\ge 30$ | $\ge 40$ (E,S,M,N,T) | `SES_P34_<timestamp>` |
| `Signer_P35` | Final Evaluation | `physical_human_validation` | $\ge 30$ | $\ge 40$ (E,S,M,N,T) | `SES_P35_<timestamp>` |

---

### 2. Data Integrity Safeguards & Leakage Defense

#### A. Overlap Audit Verification
Direct execution of dataset overlap audit confirmed:
- Zero signer overlap between training/validation corpora and `Signer_P26`–`P35`.
- Absolute disjoint partition between model development and field evaluation.

#### B. Safeguards Against Cross-Participant Contamination
1. **Dynamic Session Isolation**: Changing `signerId` immediately re-keys `sessionId`.
2. **Export Partitioning**: Each export file is strictly named `ZERO_LEAK_<signerId>_<sessionId>.json`, preventing accidental amalgamation.
3. **Partition Immutability**: All records for `Signer_P26` through `Signer_P35` automatically receive the immutable tag `datasetSplit: "physical_human_validation"`.

#### C. Safeguards Against Development / Synthetic Data Contamination
1. **Synthetic Sample Tagging**: Any sample recorded under the Simulated Test Harness is explicitly prefixed with `SIM_`, assigned `isSynthetic: true`, and assigned `capture_mode: "SIMULATED_TEST_HARNESS"`.
2. **Export Manifest Quarantine**: The export manifest embeds `isSyntheticHarnessActive: boolean` and filenames carry `_SYNTHETIC` if any simulated sample exists in the session.
3. **Downstream Acceptance Gate**: The physical human evaluation benchmark script strictly rejects any sample where `isSynthetic == true` or `capture_mode != "PHYSICAL_CAMERA"`.

#### D. Non-PII Safeguards & Privacy Compliance
1. **Zero Raw Video or Image Retention**: Frames are analyzed in transient canvas memory; no pixel buffers or base64 images are written to storage.
2. **Pseudonymous Identifiers**: No real participant names, email addresses, phone numbers, or IP addresses are collected.
3. **Mandatory Informed Consent**: The capture interface locks out the capture triggers until the participant/operator checks the explicit ethical consent box.
4. **Right to Erase**: Individual sample deletion and immediate full session buffer clearance are provided directly in the UI.

---

### 3. Conclusion

The participant allocation and data integrity governance protocol is complete, verified, and sealed against cross-contamination or PII leakage.
