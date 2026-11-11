import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

class WebMediaPipeService {
  private handLandmarker: HandLandmarker | null = null;
  private isInitializing = false;

  public async init(): Promise<boolean> {
    if (this.handLandmarker) return true;
    if (this.isInitializing) return false;
    this.isInitializing = true;

    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numHands: 2
      });
      this.isInitializing = false;
      return true;
    } catch (gpuErr) {
      console.warn('MediaPipe GPU initialization failed, attempting CPU fallback:', gpuErr);
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
        this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'CPU'
          },
          runningMode: 'VIDEO',
          numHands: 2
        });
        this.isInitializing = false;
        return true;
      } catch (cpuErr) {
        console.error('MediaPipe CPU initialization failed:', cpuErr);
        this.isInitializing = false;
        return false;
      }
    }
  }

  public detect(videoElement: HTMLVideoElement, timestamp: number) {
    if (!this.handLandmarker || videoElement.readyState < 2) return null;
    try {
      return this.handLandmarker.detectForVideo(videoElement, timestamp);
    } catch {
      return null;
    }
  }
}

export const webMediaPipe = new WebMediaPipeService();
