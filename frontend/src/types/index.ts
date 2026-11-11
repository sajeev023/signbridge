export type SupportedLanguage = 'en' | 'hi' | 'te';

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

export interface PredictionResult {
  status: PredictionStatus;
  is_stable?: boolean;
  sign_id?: string;
  label?: string;
  category?: string;
  confidence?: number;
  translations?: Translations;
  description?: string;
  message?: string;
  threshold?: number;
  latency_ms?: number;
  timestamp?: number;
  attention_weights?: number[];
  all_probabilities?: Record<string, number>;
  buffer_fill?: number;
  buffer_count?: number;
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
