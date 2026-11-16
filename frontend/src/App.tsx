import React, { useState, useEffect, Suspense, lazy } from 'react';
import { Navbar } from './components/Navbar';
import { SupportedLanguage } from './types';
import { apiClient } from './services/apiClient';
import { speechService } from './services/speechService';
import { ShieldCheck, GitBranch, Loader2 } from 'lucide-react';

// Lazy-loaded page components for optimal bundle splitting
const LandingPage = lazy(() => import('./pages/LandingPage').then(m => ({ default: m.LandingPage })));
const LiveISLAlphabetRecognizer = lazy(() => import('./pages/LiveISLAlphabetRecognizer').then(m => ({ default: m.LiveISLAlphabetRecognizer })));
const TranslatorWorkspace = lazy(() => import('./pages/TranslatorWorkspace').then(m => ({ default: m.TranslatorWorkspace })));
const HistoryPage = lazy(() => import('./pages/HistoryPage').then(m => ({ default: m.HistoryPage })));
const VocabularyPage = lazy(() => import('./pages/VocabularyPage').then(m => ({ default: m.VocabularyPage })));
const LearningPage = lazy(() => import('./pages/LearningPage').then(m => ({ default: m.LearningPage })));
const ResearchDashboardPage = lazy(() => import('./pages/ResearchDashboardPage').then(m => ({ default: m.ResearchDashboardPage })));
const TechnicalPage = lazy(() => import('./pages/TechnicalPage').then(m => ({ default: m.TechnicalPage })));
const PrivacyPage = lazy(() => import('./pages/PrivacyPage').then(m => ({ default: m.PrivacyPage })));
const AboutPage = lazy(() => import('./pages/AboutPage').then(m => ({ default: m.AboutPage })));

const VALID_PAGES = [
  'landing',
  'recognizer',
  'translator',
  'history',
  'vocabulary',
  'learning',
  'research',
  'technical',
  'privacy',
  'about'
];

function getInitialPage(): string {
  if (typeof window === 'undefined') return 'landing';
  const hash = window.location.hash.replace('#', '').trim();
  if (VALID_PAGES.includes(hash)) return hash;
  const params = new URLSearchParams(window.location.search);
  const p = params.get('page');
  if (p && VALID_PAGES.includes(p)) return p;
  return 'landing';
}

function PageLoadingFallback() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="flex flex-col items-center gap-4">
        <Loader2 className="w-10 h-10 text-brand-400 animate-spin" />
        <p className="text-slate-400 text-sm font-medium">Loading workspace module...</p>
      </div>
    </div>
  );
}

export function App() {
  const [currentPage, setCurrentPage] = useState<string>(getInitialPage);
  const [selectedLanguage, setSelectedLanguage] = useState<SupportedLanguage>('en');
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean>(false);
  const [practiceTargetSign, setPracticeTargetSign] = useState<string>('A');

  const navigateTo = (page: string) => {
    if (!VALID_PAGES.includes(page)) return;
    setCurrentPage(page);
    window.location.hash = page;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Sync hash changes on browser Back/Forward
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '').trim();
      if (VALID_PAGES.includes(hash)) {
        setCurrentPage(hash);
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Check backend health on mount and periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await apiClient.getHealth();
        setIsBackendHealthy(health.status === 'healthy' && health.model_loaded);
      } catch {
        setIsBackendHealthy(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleMute = () => {
    const nextState = !isMuted;
    setIsMuted(nextState);
    speechService.setMuted(nextState);
  };

  const handleOpenPracticeForSign = (signId: string) => {
    setPracticeTargetSign(signId);
    navigateTo('learning');
  };

  return (
    <div className="min-h-screen bg-navy-950 text-slate-100 flex flex-col font-sans selection:bg-brand-500 selection:text-white">
      {/* Top Navigation */}
      <Navbar
        currentPage={currentPage}
        onNavigate={navigateTo}
        selectedLanguage={selectedLanguage}
        onLanguageChange={setSelectedLanguage}
        isBackendHealthy={isBackendHealthy}
        isMuted={isMuted}
        onToggleMute={handleToggleMute}
      />

      {/* Main Page Content */}
      <main className="flex-1 pb-20 lg:pb-8">
        <Suspense fallback={<PageLoadingFallback />}>
          {currentPage === 'landing' && (
            <LandingPage onNavigate={navigateTo} />
          )}
          {currentPage === 'recognizer' && (
            <LiveISLAlphabetRecognizer onNavigate={navigateTo} />
          )}
          {currentPage === 'translator' && (
            <TranslatorWorkspace
              selectedLanguage={selectedLanguage}
              onLanguageChange={setSelectedLanguage}
              isMuted={isMuted}
              onToggleMute={handleToggleMute}
            />
          )}
          {currentPage === 'history' && (
            <HistoryPage
              selectedLanguage={selectedLanguage}
              isMuted={isMuted}
            />
          )}
          {currentPage === 'vocabulary' && (
            <VocabularyPage
              onPracticeSign={handleOpenPracticeForSign}
              selectedLanguage={selectedLanguage}
              isMuted={isMuted}
            />
          )}
          {currentPage === 'learning' && (
            <LearningPage
              initialSignId={practiceTargetSign}
              selectedLanguage={selectedLanguage}
              isMuted={isMuted}
              onNavigate={navigateTo}
            />
          )}
          {currentPage === 'research' && (
            <ResearchDashboardPage />
          )}
          {currentPage === 'technical' && (
            <TechnicalPage />
          )}
          {currentPage === 'privacy' && (
            <PrivacyPage />
          )}
          {currentPage === 'about' && (
            <AboutPage />
          )}
        </Suspense>
      </main>

      {/* Footer */}
      <footer className="border-t border-navy-850 bg-navy-950/90 py-8 px-4 sm:px-6 lg:px-8 text-xs text-slate-400 mb-16 lg:mb-0">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-lg bg-brand-600 flex items-center justify-center font-bold text-white text-[10px]">
              SB
            </div>
            <div>
              <span className="font-semibold text-slate-200">SIGNBRIDGE</span>
              <span className="text-slate-500 ml-2">v2.1.0 • Coordinate AI Production Edition</span>
            </div>
          </div>

          <p className="text-center sm:text-left text-slate-500">
            Engineered in alignment with official ISLRTC standards. Sub-2ms on-device coordinate inference.
          </p>

          <div className="flex items-center gap-4">
            <button
              onClick={() => navigateTo('privacy')}
              className="text-slate-400 hover:text-white transition-colors cursor-pointer"
            >
              Privacy Charter
            </button>
            <button
              onClick={() => navigateTo('research')}
              className="text-slate-400 hover:text-white transition-colors cursor-pointer"
            >
              Research & Metrics
            </button>
            <button
              onClick={() => navigateTo('about')}
              className="text-slate-400 hover:text-white transition-colors cursor-pointer"
            >
              About
            </button>
            <span className="flex items-center gap-1 text-slate-400">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span>Zero Data Retention</span>
            </span>
            <a
              href="https://github.com/sajeev023/signbridge"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-slate-400 hover:text-white transition-colors"
            >
              <GitBranch className="w-3.5 h-3.5" />
              <span>GitHub</span>
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
