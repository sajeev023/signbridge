/**
 * SIGNBRIDGE Unified Data Contracts — TypeScript Interface Definitions
 * Matches backend app.models.data_contracts.py bit-for-bit.
 */

export const FEATURE_SCHEMA_VERSION = "2.1.0-coordinate-94d";
export const MODEL_VERSION = "2.1.0";

export interface CandidatePrediction {
  letter: string;
  confidence: number;
}

export interface LandmarkPoint3D {
  x: number;
  y: number;
  z?: number;
}

export interface FrameInferencePayload {
  frame_id: number;
  timestamp_ms: number;
  has_hands: boolean;
  hand_count: number;
  handedness: 'Right' | 'Left' | 'Both' | 'None';
  landmarks_21?: LandmarkPoint3D[] | null;
  raw_coordinates_63?: number[] | null;
  enriched_features_94?: number[] | null;
  feature_schema_version?: string;
  model_version?: string;
  confidence_threshold?: number;
  session_id?: string;
  mode?: 'alphabet' | 'words';
  left_hand?: LandmarkPoint3D[] | null;
  right_hand?: LandmarkPoint3D[] | null;
  pose?: LandmarkPoint3D[] | null;
}

export interface InferenceResultEnvelope {
  frame_id: number;
  timestamp_ms: number;
  status: 'recognized' | 'uncertain' | 'rejected' | 'buffering' | 'no_hand' | 'idle';
  predicted_sign: string;
  confidence: number;
  is_accepted: boolean;
  is_dynamic: boolean;
  rejection_reason?: string | null;
  top_candidates: CandidatePrediction[];
  probabilities: Record<string, number>;
  feature_schema_version: string;
  model_version: string;
  latency_ms: number;
  translations?: Record<string, string> | null;
  label?: string | null;
  category?: string | null;
  fallback_used?: boolean;
}
