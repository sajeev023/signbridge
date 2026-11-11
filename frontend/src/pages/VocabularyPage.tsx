import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Volume2, 
  BookOpen, 
  Tag, 
  GraduationCap, 
  CheckCircle2, 
  Hand, 
  Layers 
} from 'lucide-react';
import { ISLSign, SupportedLanguage } from '../types';
import { apiClient, FALLBACK_VOCABULARY } from '../services/apiClient';
import { speechService } from '../services/speechService';

interface VocabularyPageProps {
  onPracticeSign: (signId: string) => void;
  selectedLanguage: SupportedLanguage;
  isMuted: boolean;
}

export const VocabularyPage: React.FC<VocabularyPageProps> = ({
  onPracticeSign,
  selectedLanguage,
  isMuted
}) => {
  const [signs, setSigns] = useState<ISLSign[]>(FALLBACK_VOCABULARY);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  useEffect(() => {
    apiClient.getVocabulary().then((res) => {
      if (res && res.length > 0) setSigns(res);
    });
  }, []);

  const categories = [
    { id: 'all', label: 'All Signs' },
    { id: 'greetings', label: 'Greetings' },
    { id: 'social', label: 'Social' },
    { id: 'emergency', label: 'Emergency' },
    { id: 'daily_needs', label: 'Daily Needs' },
    { id: 'medical', label: 'Medical' },
  ];

  const filteredSigns = signs.filter((s) => {
    const matchesCategory = selectedCategory === 'all' || s.category === selectedCategory;
    const matchesSearch = 
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.translations.en.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.translations.hi.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-brand-400" />
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            Indian Sign Language Vocabulary Explorer
          </h1>
        </div>
        <p className="text-sm text-slate-400 mt-1 max-w-3xl">
          Morphologically verified dictionary of 10 essential communicative and emergency ISL gestures,
          grounded in the national standards of the Indian Sign Language Research and Training Centre (ISLRTC).
        </p>
      </div>

      {/* Search and Filters Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-navy-900 border border-navy-800 p-4 rounded-2xl shadow-subtle">
        {/* Search Input */}
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search sign, English or Hindi meaning..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-navy-950 border border-navy-800 rounded-xl pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-brand-500 transition-colors"
          />
        </div>

        {/* Category Pills */}
        <div className="flex flex-wrap gap-1.5 w-full sm:w-auto">
          {categories.map((cat) => (
            <button
              key={cat.id}
              onClick={() => setSelectedCategory(cat.id)}
              className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-all ${
                selectedCategory === cat.id
                  ? 'bg-brand-600 text-white font-semibold shadow-sm'
                  : 'bg-navy-950 text-slate-400 hover:text-white border border-navy-800'
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>
      </div>

      {/* Vocabulary Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredSigns.map((sign) => (
          <div
            key={sign.id}
            className="bg-navy-900 border border-navy-800 hover:border-navy-700 rounded-2xl p-6 transition-all shadow-subtle hover:shadow-glow flex flex-col justify-between"
          >
            <div>
              {/* Top Meta Badges */}
              <div className="flex items-center justify-between mb-3">
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-brand-950 text-brand-300 border border-brand-800">
                  {sign.category.toUpperCase()}
                </span>
                <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400">
                  <span className="px-2 py-0.5 rounded bg-navy-950 border border-navy-800">
                    {sign.hand_config === 'two_handed' ? 'Two Hands' : 'One Hand'}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-navy-950 border border-navy-800 text-sky-400">
                    {sign.difficulty}
                  </span>
                </div>
              </div>

              {/* Sign Title */}
              <h3 className="text-xl font-bold text-white tracking-tight">
                {sign.name}
              </h3>
              <p className="text-xs text-brand-300 font-medium mt-0.5">
                {sign.label}
              </p>

              {/* Multi-Lingual Box */}
              <div className="space-y-1.5 mt-4 p-3 bg-navy-950/80 rounded-xl border border-navy-850 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-500 font-mono">English:</span>
                  <span className="font-medium text-slate-200">{sign.translations.en}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-500 font-mono">Hindi (हिंदी):</span>
                  <span className="font-medium text-slate-200">{sign.translations.hi}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-500 font-mono">Telugu (తెలుగు):</span>
                  <span className="font-medium text-slate-200">{sign.translations.te}</span>
                </div>
              </div>

              {/* Description */}
              <div className="mt-4">
                <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1">
                  Biomechanical Execution:
                </span>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {sign.description}
                </p>
              </div>

              <div className="mt-2 text-[11px] text-slate-400 bg-navy-950/40 p-2 rounded-lg border border-navy-850/60">
                <strong className="text-slate-300">Kinematic Stroke:</strong> {sign.movement}
              </div>
            </div>

            {/* Bottom Actions */}
            <div className="mt-6 pt-4 border-t border-navy-800 flex items-center justify-between gap-2">
              <button
                onClick={() => speechService.speak(sign.translations[selectedLanguage] || sign.translations.en, selectedLanguage)}
                disabled={isMuted}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 hover:text-white bg-navy-950 hover:bg-navy-850 border border-navy-800 transition-all"
                title="Play Audio Pronunciation"
              >
                <Volume2 className="w-3.5 h-3.5 text-brand-400" />
                <span>Hear Audio</span>
              </button>

              <button
                onClick={() => onPracticeSign(sign.id)}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold text-white bg-brand-600 hover:bg-brand-500 shadow-glow transition-all"
              >
                <GraduationCap className="w-3.5 h-3.5" />
                <span>Practice Form</span>
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
