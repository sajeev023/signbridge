import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LandingPage } from './pages/LandingPage';
import { TranslatorWorkspace } from './pages/TranslatorWorkspace';
import { HistoryPage } from './pages/HistoryPage';
import { VocabularyPage } from './pages/VocabularyPage';
import { LearningPage } from './pages/LearningPage';
import { ResearchDashboardPage } from './pages/ResearchDashboardPage';
import { TechnicalPage } from './pages/TechnicalPage';
import { PrivacyPage } from './pages/PrivacyPage';
import { AboutPage } from './pages/AboutPage';
import { SupportedLanguage } from './types';
import { apiClient } from './services/apiClient';
import { speechService } from './services/speechService';
import { ShieldCheck, GitBranch } from 'lucide-react';

export function App() {
  const [currentPage, setCurrentPage] = useState<string>('landing');
  const [selectedLanguage, setSelectedLanguage] = useState<SupportedLanguage>('en');
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean>(false);
  const [practiceTargetSign, setPracticeTargetSign] = useState<string>('namaste');

  // Check backend health on mount and periodically
  useEffect(() => {
    const checkHealth = async () => {
      const health = await apiClient.getHealth();
      setIsBackendHealthy(health.status === 'healthy' && health.model_loaded);
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
    setCurrentPage('learning');
  };

  return (
    <div className="min-h-screen bg-navy-950 text-slate-100 flex flex-col font-sans selection:bg-brand-500 selection:text-white">
      {/* Top Navigation */}
      <Navbar
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        selectedLanguage={selectedLanguage}
        onLanguageChange={setSelectedLanguage}
        isBackendHealthy={isBackendHealthy}
        isMuted={isMuted}
        onToggleMute={handleToggleMute}
      />

      {/* Main Page Content */}
      <main className="flex-1">
        {currentPage === 'landing' && (
          <LandingPage onNavigate={setCurrentPage} />
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
      </main>

      {/* Footer */}
      <footer className="border-t border-navy-850 bg-navy-950/90 py-8 px-4 sm:px-6 lg:px-8 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-lg bg-brand-600 flex items-center justify-center font-bold text-white text-[10px]">
              SB
            </div>
            <div>
              <span className="font-semibold text-slate-200">SIGNBRIDGE</span>
              <span className="text-slate-500 ml-2">v2.1.0 • Release Candidate</span>
            </div>
          </div>

          <p className="text-center sm:text-left text-slate-500">
            Engineered in alignment with official ISLRTC standards. Designed for assistive social impact.
          </p>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setCurrentPage('privacy')}
              className="text-slate-400 hover:text-white transition-colors"
            >
              Privacy Charter
            </button>
            <button
              onClick={() => setCurrentPage('research')}
              className="text-slate-400 hover:text-white transition-colors"
            >
              Research & Metrics
            </button>
            <span className="flex items-center gap-1 text-slate-400" title="Webcam in RAM; uploads purged immediately post-inference">
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
              <span>GitHub Repo</span>
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
