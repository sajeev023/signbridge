import React, { useState, useEffect } from 'react';
import { 
  History, 
  Trash2, 
  Download, 
  Volume2, 
  Search, 
  Copy, 
  Check, 
  ShieldCheck, 
  Tag, 
  Clock 
} from 'lucide-react';
import { TranslationRecord, SupportedLanguage } from '../types';
import { apiClient } from '../services/apiClient';
import { speechService } from '../services/speechService';

interface HistoryPageProps {
  selectedLanguage: SupportedLanguage;
  isMuted: boolean;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({
  selectedLanguage,
  isMuted
}) => {
  const [records, setRecords] = useState<TranslationRecord[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    setRecords(apiClient.getHistory());
  }, []);

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear your local translation history?')) {
      apiClient.clearHistory();
      setRecords([]);
    }
  };

  const handleDeleteItem = (id: string) => {
    const updated = apiClient.deleteHistoryRecord(id);
    setRecords(updated);
  };

  const handleExportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(records, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `signbridge_translations_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const filteredRecords = records.filter((r) => {
    return (
      r.translation_text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.category.toLowerCase().includes(searchQuery.toLowerCase())
    );
  });

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Page Title & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-navy-850">
        <div>
          <div className="flex items-center gap-2">
            <History className="w-6 h-6 text-brand-400" />
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              Translation History
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Local browser logs of confident Indian Sign Language predictions and spoken outputs.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {records.length > 0 && (
            <>
              <button
                onClick={handleExportJSON}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-navy-900 hover:bg-navy-850 border border-navy-800 rounded-lg transition-all"
                title="Export history as JSON file"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export JSON</span>
              </button>

              <button
                onClick={handleClearHistory}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-rose-300 hover:text-rose-200 bg-rose-950/60 hover:bg-rose-900 border border-rose-800 rounded-lg transition-all"
                title="Clear local history"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear History</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Privacy Guarantee Banner */}
      <div className="flex items-start gap-3 p-4 rounded-2xl bg-navy-900/60 border border-navy-800 text-xs text-slate-300">
        <ShieldCheck className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong>Privacy Guarantee:</strong> All translation logs are stored entirely in your local browser storage.
          Camera frames and video streams are processed live in-memory and are never recorded, saved, or uploaded to any server.
        </p>
      </div>

      {/* Search Input */}
      {records.length > 0 && (
        <div className="relative max-w-sm">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search translation logs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-navy-900 border border-navy-800 rounded-xl pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-brand-500 transition-colors"
          />
        </div>
      )}

      {/* History Records List */}
      {records.length === 0 ? (
        <div className="bg-navy-900/50 border border-navy-800 rounded-2xl p-12 text-center text-slate-400">
          <History className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <h3 className="text-base font-semibold text-slate-200">No Translations Recorded</h3>
          <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto">
            When you run the Live Translator, confident translations will automatically appear here with replayable audio.
          </p>
        </div>
      ) : filteredRecords.length === 0 ? (
        <p className="text-xs text-slate-500 italic py-6 text-center">
          No records match your search query "{searchQuery}".
        </p>
      ) : (
        <div className="space-y-3">
          {filteredRecords.map((rec) => (
            <div
              key={rec.id}
              className="bg-navy-900 border border-navy-800 hover:border-navy-750 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-all shadow-subtle"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-base font-bold text-white tracking-tight">
                    {rec.translation_text}
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-brand-950 text-brand-300 border border-brand-800 uppercase">
                    {rec.language}
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono text-slate-400 bg-navy-950 border border-navy-800">
                    {rec.category}
                  </span>
                  {rec.recognition_mode && (
                    <span className={`px-2 py-0.5 rounded text-[9px] font-mono uppercase ${
                      rec.recognition_mode === 'video_upload'
                        ? 'text-brand-300 bg-brand-950 border border-brand-800'
                        : rec.recognition_mode === 'live_camera'
                        ? 'text-emerald-300 bg-emerald-950 border border-emerald-800'
                        : 'text-amber-300 bg-amber-950 border border-amber-800'
                    }`}>
                      {rec.recognition_mode === 'video_upload' ? 'Upload' : rec.recognition_mode === 'live_camera' ? 'Camera' : 'Demo'}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3 text-slate-500" />
                    {new Date(rec.timestamp).toLocaleString()}
                  </span>
                  <span>•</span>
                  <span className="text-emerald-400">
                    {Math.round(rec.confidence * 100)}% Match
                  </span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2 self-end sm:self-center">
                <button
                  onClick={() => speechService.speak(rec.translation_text, rec.language)}
                  disabled={isMuted}
                  className="p-2 rounded-lg bg-navy-950 hover:bg-navy-850 text-slate-300 hover:text-white border border-navy-800 transition-all"
                  title="Replay Voice Audio"
                >
                  <Volume2 className="w-4 h-4 text-brand-400" />
                </button>

                <button
                  onClick={() => handleCopy(rec.translation_text, rec.id)}
                  className="p-2 rounded-lg bg-navy-950 hover:bg-navy-850 text-slate-300 hover:text-white border border-navy-800 transition-all"
                  title="Copy Translation Text"
                >
                  {copiedId === rec.id ? (
                    <Check className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>

                <button
                  onClick={() => handleDeleteItem(rec.id)}
                  className="p-2 rounded-lg bg-navy-950 hover:bg-rose-950/60 text-slate-400 hover:text-rose-300 border border-navy-800 hover:border-rose-800 transition-all"
                  title="Delete this record"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
