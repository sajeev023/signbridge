export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  res.status(200).json({
    status: "healthy",
    version: "2.1.0",
    model_loaded: true,
    device: "edge-cloud",
    supported_vocabulary_count: 10,
    confidence_threshold: 0.75,
    features: {
      video_upload: true,
      live_camera: true,
      demo_mode: true,
      practice_studio: true,
      offline_inference: true
    },
    timestamp: Date.now() / 1000
  });
}
