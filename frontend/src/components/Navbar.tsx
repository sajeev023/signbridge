import React from 'react';
import { 
  Radio, 
  Volume2, 
  VolumeX, 
  BookOpen, 
  History, 
  Cpu, 
  GraduationCap, 
  Info,
  Sparkles,
  BarChart3,
  ShieldCheck,
  Camera
} from 'lucide-react';
import { SupportedLanguage } from '../types';

interface NavbarProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  selectedLanguage: SupportedLanguage;
  onLanguageChange: (lang: SupportedLanguage) => void;
  isBackendHealthy: boolean;
  isMuted: boolean;
  onToggleMute: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentPage,
  onNavigate,
  selectedLanguage,
  onLanguageChange,
  isBackendHealthy,
  isMuted,
  onToggleMute
}) => {
  const desktopNavItems = [
    { id: 'landing', label: 'Home', icon: Sparkles },
    { id: 'translator', label: 'Live Translator', icon: Camera },
    { id: 'vocabulary', label: 'ISL Vocabulary', icon: BookOpen },
    { id: 'learning', label: 'Practice Studio', icon: GraduationCap },
    { id: 'history', label: 'History', icon: History },
    { id: 'research', label: 'Research & Specs', icon: BarChart3 },
    { id: 'privacy', label: 'Privacy', icon: ShieldCheck },
  ];

  const mobileBottomItems = [
    { id: 'landing', label: 'Home', icon: Sparkles },
    { id: 'translator', label: 'Translate', icon: Camera },
    { id: 'vocabulary', label: 'Vocab', icon: BookOpen },
    { id: 'learning', label: 'Practice', icon: GraduationCap },
    { id: 'history', label: 'History', icon: History },
  ];

  return (
    <>
      {/* Top Header */}
      <header className="sticky top-0 z-40 bg-navy-950/90 backdrop-blur-md border-b border-navy-800/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14 sm:h-16">
            {/* Brand Logo */}
            <div 
              onClick={() => onNavigate('landing')} 
              className="flex items-center gap-2.5 cursor-pointer group"
            >
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-600 via-sky-500 to-indigo-400 p-[1px] shadow-glow">
                <div className="w-full h-full bg-navy-900 rounded-[11px] flex items-center justify-center">
                  <span className="font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-sky-300 text-sm sm:text-base tracking-tighter">
                    SB
                  </span>
                </div>
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="font-extrabold text-base sm:text-lg tracking-tight text-white group-hover:text-brand-400 transition-colors">
                    SIGNBRIDGE
                  </span>
                  <span className="px-1.5 py-0.2 text-[9px] font-bold bg-brand-950 text-brand-300 border border-brand-800 rounded tracking-wider uppercase">
                    ISL AI
                  </span>
                </div>
              </div>
            </div>

            {/* Desktop Navigation Links */}
            <nav className="hidden lg:flex items-center gap-1">
              {desktopNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentPage === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => onNavigate(item.id)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl transition-all ${
                      isActive
                        ? 'bg-brand-600 text-white shadow-glow'
                        : 'text-slate-300 hover:text-white hover:bg-navy-850'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>

            {/* Utility Controls: Language Selector & Mute */}
            <div className="flex items-center gap-2">
              {/* Language Selector */}
              <div className="flex items-center bg-navy-900 border border-navy-800 rounded-xl p-0.5 text-xs shadow-subtle">
                {(['en', 'hi', 'te'] as SupportedLanguage[]).map((l) => (
                  <button
                    key={l}
                    onClick={() => onLanguageChange(l)}
                    className={`px-2 py-1 rounded-lg text-[11px] font-bold uppercase transition-all ${
                      selectedLanguage === l
                        ? 'bg-brand-600 text-white shadow-sm'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {l}
                  </button>
                ))}
              </div>

              {/* Sound Toggle */}
              <button
                onClick={onToggleMute}
                className={`p-2 rounded-xl border transition-all ${
                  isMuted
                    ? 'bg-rose-950/60 border-rose-800 text-rose-400'
                    : 'bg-navy-900 border-navy-800 text-slate-300 hover:text-white hover:border-navy-700'
                }`}
                title={isMuted ? 'Unmute Audio' : 'Mute Audio'}
              >
                {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4 text-brand-400" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Bottom Navigation Bar (Thumb Friendly) */}
      <nav className="lg:hidden fixed bottom-0 left-0 right-0 z-50 bg-navy-950/95 backdrop-blur-xl border-t border-navy-800/80 px-2 py-1.5 flex justify-around shadow-2xl safe-area-pb">
        {mobileBottomItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`flex flex-col items-center justify-center py-1 px-3 rounded-xl transition-all ${
                isActive
                  ? 'text-brand-400 font-bold scale-105'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Icon className={`w-5 h-5 ${isActive ? 'text-brand-400' : 'text-slate-400'}`} />
              <span className="text-[10px] mt-0.5 tracking-tight">{item.label}</span>
            </button>
          );
        })}
      </nav>
    </>
  );
};
