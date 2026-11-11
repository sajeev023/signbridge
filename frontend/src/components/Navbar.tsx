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
  Sparkles
} from 'lucide-react';
import { SupportedLanguage } from '../types';
import { speechService } from '../services/speechService';

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
  const navItems = [
    { id: 'landing', label: 'Overview', icon: Sparkles },
    { id: 'translator', label: 'Live Translator', icon: Radio },
    { id: 'history', label: 'History', icon: History },
    { id: 'vocabulary', label: 'ISL Vocabulary', icon: BookOpen },
    { id: 'learning', label: 'Practice Studio', icon: GraduationCap },
    { id: 'technical', label: 'Technical Spec', icon: Cpu },
    { id: 'about', label: 'About', icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 bg-navy-950/85 backdrop-blur-md border-b border-navy-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo */}
          <div 
            onClick={() => onNavigate('landing')} 
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-700 via-brand-600 to-sky-400 p-[1px] shadow-glow">
              <div className="w-full h-full bg-navy-900 rounded-[11px] flex items-center justify-center">
                <span className="font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-sky-300 text-lg tracking-tighter">
                  SB
                </span>
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-tight text-white group-hover:text-brand-400 transition-colors">
                  SIGNBRIDGE
                </span>
                <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-brand-950 text-brand-300 border border-brand-800/80 rounded tracking-wider uppercase">
                  ISL AI
                </span>
              </div>
              <p className="text-[10px] text-slate-400 tracking-wide font-mono hidden sm:block">
                Indian Sign Language Neural Bridge
              </p>
            </div>
          </div>

          {/* Navigation links */}
          <nav className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentPage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
                    isActive
                      ? 'bg-brand-600 text-white shadow-sm'
                      : 'text-slate-300 hover:text-white hover:bg-navy-850'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Utility controls: Language, Mute, Status */}
          <div className="flex items-center gap-3">
            {/* System Status Pill */}
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-navy-900 border border-navy-800 text-[11px]">
              <span className={`w-2 h-2 rounded-full ${isBackendHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
              <span className="text-slate-300 font-mono text-[10px]">
                {isBackendHealthy ? 'AI Engine Ready' : 'Client Mode'}
              </span>
            </div>

            {/* Language Selector */}
            <div className="flex items-center bg-navy-900 border border-navy-800 rounded-lg p-0.5 text-xs">
              <button
                onClick={() => onLanguageChange('en')}
                className={`px-2 py-1 rounded-md font-medium transition-all ${
                  selectedLanguage === 'en'
                    ? 'bg-brand-600 text-white font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="English Speech Output"
              >
                EN
              </button>
              <button
                onClick={() => onLanguageChange('hi')}
                className={`px-2 py-1 rounded-md font-medium transition-all ${
                  selectedLanguage === 'hi'
                    ? 'bg-brand-600 text-white font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Hindi (हिंदी) Speech Output"
              >
                HI
              </button>
              <button
                onClick={() => onLanguageChange('te')}
                className={`px-2 py-1 rounded-md font-medium transition-all ${
                  selectedLanguage === 'te'
                    ? 'bg-brand-600 text-white font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Telugu (తెలుగు) Speech Output"
              >
                TE
              </button>
            </div>

            {/* Audio Mute/Unmute */}
            <button
              onClick={onToggleMute}
              className={`p-2 rounded-lg border transition-all ${
                isMuted
                  ? 'bg-rose-950/40 border-rose-800/80 text-rose-400'
                  : 'bg-navy-900 border-navy-800 text-slate-300 hover:text-white hover:border-navy-700'
              }`}
              title={isMuted ? 'Unmute Audio' : 'Mute Audio'}
            >
              {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Nav strip */}
      <div className="lg:hidden flex overflow-x-auto border-t border-navy-850 px-2 py-1 gap-1 scrollbar-none bg-navy-950/95">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-xs whitespace-nowrap rounded-md font-medium ${
                isActive ? 'bg-brand-600 text-white' : 'text-slate-400'
              }`}
            >
              <Icon className="w-3 h-3" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </header>
  );
};
