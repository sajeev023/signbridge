export type SupportedLanguage = 'en' | 'hi' | 'te';
export type RecognitionMode = 'video_upload' | 'live_camera' | 'demonstration';

export interface Translations {
  en: string;
  hi: string;
  te: string;
}

export interface ISLSign {
  id: string;
  name: string;
  label: string;
  category: 'greetings' | 'social' | 'emergency' | 'daily_needs' | 'affirmation' | 'negation' | 'medical';
  translations: Translations;
  description: string;
  hand_config: 'single_handed' | 'two_handed';
  movement: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
}

export interface Landmark3D {
  x: number;
  y: number;
  z: number;
}

export type PredictionStatus = 'idle' | 'buffering' | 'evaluating' | 'recognized' | 'uncertain' | 'error';

export interface VideoMetadata {
  total_frames: number;
  fps: number;
  duration_seconds: number;
  width: number;
  height: number;
  sampled_frames: number;
}

export interface PredictionResult {
  status: PredictionStatus;
  is_stable?: boolean;
  is_confident?: boolean;
  sign_id?: string;
  label?: string;
  category?: string;
  confidence?: number;
  translations?: Translations;
  description?: string;
  message?: string;
  threshold?: number;
  latency_ms?: number;
  processing_duration_ms?: number;
  timestamp?: number;
  attention_weights?: number[];
  all_probabilities?: Record<string, number>;
  buffer_fill?: number;
  buffer_count?: number;
  video_metadata?: VideoMetadata;
  model_version?: string;
  input_mode?: RecognitionMode;
}

export interface TranslationRecord {
  id: string;
  sign_id: string;
  label: string;
  category: string;
  translation_text: string;
  language: SupportedLanguage;
  confidence: number;
  timestamp: number;
  spoken: boolean;
  recognition_mode: RecognitionMode;
}

export interface PracticeEvaluation {
  target_sign: string;
  target_label: string;
  category: string;
  accuracy_score: number;
  grade: 'Excellent' | 'Good Progress' | 'Needs Practice' | 'Beginning';
  status: 'mastered' | 'proficient' | 'learning' | 'starting';
  feedback: string[];
  canonical_movement: string;
  difficulty: string;
  live_prediction?: PredictionResult;
  video_metadata?: VideoMetadata;
}

export interface ResearchMetrics {
  timestamp: string;
  hardware: string;
  test_samples: number;
  overall_accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  latency_ms: {
    mean: number;
    p95: number;
    min: number;
    max: number;
  };
  per_class: Array<{
    class_id: string;
    precision: number;
    recall: number;
    f1: number;
    support: number;
  }>;
  confusion_matrix: number[][];
  validation_status: {
    synthetic_benchmark: string;
    real_world_validation: string;
    confidence_threshold: number;
    model_version: string;
  };
}

export interface ModelMetadata {
  model_name: string;
  version: string;
  trained_at: string;
  architecture: {
    type: string;
    input_dim: number;
    hidden_dim: number;
    num_layers: number;
    num_classes: number;
    total_parameters: number;
    sequence_length: number;
  };
  dataset_split: {
    train_samples: number;
    val_samples: number;
    test_samples: number;
    total_samples: number;
  };
  performance: {
    best_val_accuracy: number;
    test_accuracy: number;
    epochs_trained: number;
    training_time_seconds: number;
  };
  vocabulary: string[];
}
