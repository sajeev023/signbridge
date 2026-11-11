import { ISLSign, PredictionResult, PracticeEvaluation, ModelMetadata, TranslationRecord } from '../types';

export const FALLBACK_VOCABULARY: ISLSign[] = [
  {
    id: "namaste",
    name: "Namaste",
    label: "Namaste / Welcome",
    category: "greetings",
    translations: {
      en: "Namaste / Welcome",
      hi: "नमस्ते / स्वागत है",
      te: "నమస్తే / స్వాగతం"
    },
    description: "Both hands joined palms together at chest level in traditional Indian reverential greeting with slight forward head tilt.",
    hand_config: "two_handed",
    movement: "Upward trajectory to center chest, palms pressed firmly, brief hold.",
    difficulty: "Beginner"
  },
  {
    id: "thank_you",
    name: "Thank You",
    label: "Thank You",
    category: "social",
    translations: {
      en: "Thank You",
      hi: "धन्यवाद / शुक्रिया",
      te: "ధన్యవాదాలు"
    },
    description: "Dominant flat open hand touching chin/lips, then sweeping smoothly outward and downward towards the conversation partner.",
    hand_config: "single_handed",
    movement: "Chin touch to forward downward arc.",
    difficulty: "Beginner"
  },
  {
    id: "help",
    name: "Help",
    label: "Help / Need Assistance",
    category: "emergency",
    translations: {
      en: "Help! I need assistance",
      hi: "मदद! मुझे सहायता चाहिए",
      te: "సహాయం! నాకు సహాయం కావాలి"
    },
    description: "Dominant hand closed in a fist with thumb up, placed upon open palm of non-dominant hand, both hands lifting upward together.",
    hand_config: "two_handed",
    movement: "Base hand supports lifting fist vertically.",
    difficulty: "Intermediate"
  },
  {
    id: "yes",
    name: "Yes",
    label: "Yes / Agreed",
    category: "affirmation",
    translations: {
      en: "Yes, I agree",
      hi: "हाँ, मैं सहमत हूँ",
      te: "అవును, నేను అంగీకరిస్తున్నాను"
    },
    description: "Dominant hand in a soft fist nodding at wrist up and down repeatedly, mimicking a nodding head.",
    hand_config: "single_handed",
    movement: "Vertical wrist flexion nod 2-3 times.",
    difficulty: "Beginner"
  },
  {
    id: "no",
    name: "No",
    label: "No / Disagree",
    category: "negation",
    translations: {
      en: "No, I disagree",
      hi: "नहीं, मैं असहमत हूँ",
      te: "కాదు / వద్దు"
    },
    description: "Dominant hand held upright with index finger or open palm waving horizontally side to side.",
    hand_config: "single_handed",
    movement: "Horizontal lateral sweep side-to-side.",
    difficulty: "Beginner"
  },
  {
    id: "water",
    name: "Water",
    label: "Water / Thirsty",
    category: "daily_needs",
    translations: {
      en: "I need water",
      hi: "मुझे पानी चाहिए",
      te: "నాకు నీళ్లు కావాలి"
    },
    description: "Curved dominant hand tilted towards mouth mimicking sipping water from a traditional glass or vessel.",
    hand_config: "single_handed",
    movement: "Tilting motion toward lips.",
    difficulty: "Beginner"
  },
  {
    id: "food",
    name: "Food / Hungry",
    label: "Food / Eat",
    category: "daily_needs",
    translations: {
      en: "I want food / I am hungry",
      hi: "मुझे खाना चाहिए / मुझे भूख लगी है",
      te: "నాకు ఆహారం కావాలి / ఆకలిగా ఉంది"
    },
    description: "All fingertips of dominant hand bunched together, moving towards and lightly touching the lips repeatedly.",
    hand_config: "single_handed",
    movement: "Repeated touch to lips.",
    difficulty: "Beginner"
  },
  {
    id: "emergency",
    name: "Emergency",
    label: "Emergency / Urgent",
    category: "emergency",
    translations: {
      en: "Emergency! Immediate attention needed",
      hi: "आपातकाल! तत्काल सहायता चाहिए",
      te: "అత్యవసరం! వెంటనే సహాయం కావాలి"
    },
    description: "Both hands raised to chest/head height with open palms frantically crossing and waving outward in rapid alarm.",
    hand_config: "two_handed",
    movement: "Rapid alternating crossed sweeps at upper torso.",
    difficulty: "Advanced"
  },
  {
    id: "doctor",
    name: "Doctor",
    label: "Doctor / Medical",
    category: "medical",
    translations: {
      en: "Please call a doctor",
      hi: "कृपया डॉक्टर को बुलाएं",
      te: "దయచేసి డాక్టర్‌ను పిలవండి"
    },
    description: "Dominant index and middle fingertips touching the radial artery/pulse of the non-dominant wrist.",
    hand_config: "two_handed",
    movement: "Two taps on the inner wrist pulse point.",
    difficulty: "Intermediate"
  },
  {
    id: "please",
    name: "Please",
    label: "Please / Request",
    category: "social",
    translations: {
      en: "Please",
      hi: "कृपया",
      te: "దయచేసి"
    },
    description: "Dominant open hand placed flat on center of chest making a clockwise circular smoothing gesture.",
    hand_config: "single_handed",
    movement: "Clockwise circular chest stroke.",
    difficulty: "Beginner"
  }
];

const API_BASE_URL = 'http://127.0.0.1:8000';
const WS_BASE_URL = 'ws://127.0.0.1:8000';
const HISTORY_STORAGE_KEY = 'signbridge_translation_history_v1';

export const apiClient = {
  async getHealth(): Promise<{ status: string; model_loaded: boolean; device: string }> {
    try {
      const res = await fetch(`${API_BASE_URL}/api/health`, { signal: AbortSignal.timeout(2500) });
      if (!res.ok) throw new Error('Health check non-200');
      return await res.json();
    } catch {
      return { status: 'offline', model_loaded: false, device: 'client-fallback' };
    }
  },

  async getVocabulary(): Promise<ISLSign[]> {
    try {
      const res = await fetch(`${API_BASE_URL}/api/vocabulary`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        return data.vocabulary;
      }
    } catch {
      // Fallback
    }
    return FALLBACK_VOCABULARY;
  },

  async getModelInfo(): Promise<ModelMetadata | null> {
    try {
      const res = await fetch(`${API_BASE_URL}/api/model/info`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Fallback
    }
    return null;
  },

  async predictFrame(payload: {
    left_hand?: any[];
    right_hand?: any[];
    pose?: any[];
  }): Promise<PredictionResult> {
    const res = await fetch(`${API_BASE_URL}/api/predict/frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Frame prediction failed');
    return await res.json();
  },

  async resetPrediction(): Promise<void> {
    try {
      await fetch(`${API_BASE_URL}/api/predict/reset`, { method: 'POST' });
    } catch {
      // ignore
    }
  },

  async evaluatePractice(payload: {
    target_sign: string;
    left_hand?: any[];
    right_hand?: any[];
    pose?: any[];
  }): Promise<PracticeEvaluation> {
    const res = await fetch(`${API_BASE_URL}/api/practice/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Practice evaluation failed');
    return await res.json();
  },

  connectWebSocket(
    onMessage: (data: any) => void,
    onError?: (err: any) => void,
    onClose?: () => void
  ): WebSocket {
    const ws = new WebSocket(`${WS_BASE_URL}/ws/stream`);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (err) {
        console.error('WS Parse error', err);
      }
    };
    if (onError) ws.onerror = onError;
    if (onClose) ws.onclose = onClose;
    return ws;
  },

  // Local Storage Translation History
  getHistory(): TranslationRecord[] {
    try {
      const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (!raw) return [];
      return JSON.parse(raw);
    } catch {
      return [];
    }
  },

  saveHistory(records: TranslationRecord[]) {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(records));
    } catch (e) {
      console.warn('LocalStorage save failed', e);
    }
  },

  addHistoryRecord(record: Omit<TranslationRecord, 'id' | 'timestamp'>): TranslationRecord[] {
    const current = this.getHistory();
    const newRecord: TranslationRecord = {
      ...record,
      id: 'tx_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7),
      timestamp: Date.now()
    };
    // Keep max 100 recent entries
    const updated = [newRecord, ...current].slice(0, 100);
    this.saveHistory(updated);
    return updated;
  },

  clearHistory() {
    try {
      localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch {
      // ignore
    }
  }
};
