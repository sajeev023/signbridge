const VOCABULARY = [
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
      en: "Yes / I agree",
      hi: "हाँ / मैं सहमत हूँ",
      te: "అవును / నేను అంగీకరిస్తున్నాను"
    },
    description: "Dominant hand formed into a closed fist nodding up and down from the wrist rhythmically like a nodding head.",
    hand_config: "single_handed",
    movement: "Vertical wrist flexing nod twice.",
    difficulty: "Beginner"
  },
  {
    id: "no",
    name: "No",
    label: "No / Disagree",
    category: "negation",
    translations: {
      en: "No / I disagree",
      hi: "नहीं / मैं असहमत हूँ",
      te: "కాదు / నేను అంగీకరించను"
    },
    description: "Dominant hand open with palm outward, waving firmly side-to-side horizontally in front of the body with head shake.",
    hand_config: "single_handed",
    movement: "Horizontal side-to-side oscillating wave.",
    difficulty: "Beginner"
  },
  {
    id: "water",
    name: "Water",
    label: "Water / Thirsty",
    category: "daily_needs",
    translations: {
      en: "I need drinking water",
      hi: "मुझे पीने का पानी चाहिए",
      te: "నాకు త్రాగునీరు కావాలి"
    },
    description: "Dominant hand cupped into a loose drinking vessel shape, brought up towards the mouth with a gentle tilting motion.",
    hand_config: "single_handed",
    movement: "Hand elevates towards mouth and tilts backwards.",
    difficulty: "Beginner"
  },
  {
    id: "food",
    name: "Food",
    label: "Food / Hungry",
    category: "daily_needs",
    translations: {
      en: "I need food / I am hungry",
      hi: "मुझे भोजन चाहिए / मुझे भूख लगी है",
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

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  res.status(200).json({
    count: VOCABULARY.length,
    vocabulary: VOCABULARY
  });
}
