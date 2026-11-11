import { SupportedLanguage } from '../types';

class SpeechService {
  private synth: SpeechSynthesis | null = null;
  private voices: SpeechSynthesisVoice[] = [];
  private isMuted: boolean = false;
  private currentUtterance: SpeechSynthesisUtterance | null = null;
  private onVoicesLoadedCallbacks: Array<() => void> = [];

  constructor() {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      this.synth = window.speechSynthesis;
      this.loadVoices();
      if (this.synth.onvoiceschanged !== undefined) {
        this.synth.onvoiceschanged = () => this.loadVoices();
      }
    }
  }

  private loadVoices() {
    if (!this.synth) return;
    this.voices = this.synth.getVoices();
    this.onVoicesLoadedCallbacks.forEach((cb) => cb());
  }

  public onVoicesReady(callback: () => void) {
    if (this.voices.length > 0) {
      callback();
    } else {
      this.onVoicesLoadedCallbacks.push(callback);
    }
  }

  public getAvailableVoices(): SpeechSynthesisVoice[] {
    return this.voices;
  }

  public setMuted(muted: boolean) {
    this.isMuted = muted;
    if (muted && this.synth) {
      this.synth.cancel();
    }
  }

  public getIsMuted(): boolean {
    return this.isMuted;
  }

  public speak(
    text: string,
    language: SupportedLanguage = 'en',
    rate: number = 0.95,
    pitch: number = 1.0,
    volume: number = 1.0
  ): boolean {
    if (!this.synth || this.isMuted || !text.trim()) {
      return false;
    }

    // Cancel any active utterance
    this.synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = rate;
    utterance.pitch = pitch;
    utterance.volume = volume;

    // Language targeting
    const langCodeMap: Record<SupportedLanguage, string[]> = {
      en: ['en-IN', 'en-GB', 'en-US', 'en'],
      hi: ['hi-IN', 'hi'],
      te: ['te-IN', 'te']
    };

    const targetLocales = langCodeMap[language] || ['en-US'];

    // Find best matching voice
    let selectedVoice: SpeechSynthesisVoice | undefined;
    for (const locale of targetLocales) {
      selectedVoice = this.voices.find(
        (v) => v.lang.toLowerCase() === locale.toLowerCase() || v.lang.toLowerCase().startsWith(locale.toLowerCase())
      );
      if (selectedVoice) break;
    }

    if (selectedVoice) {
      utterance.voice = selectedVoice;
      utterance.lang = selectedVoice.lang;
    } else {
      utterance.lang = targetLocales[0];
    }

    this.currentUtterance = utterance;
    this.synth.speak(utterance);
    return true;
  }

  public stop() {
    if (this.synth) {
      this.synth.cancel();
    }
  }
}

export const speechService = new SpeechService();
