export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  res.status(200).json({
    model_name: "SignBridge-BiLSTM-ISL",
    version: "2.1.0",
    architecture: {
      type: "1D-Conv + 2-Layer BiLSTM + Temporal Attention",
      num_features: 153,
      sequence_length: 30,
      conv_channels: 128,
      lstm_hidden_dim: 256,
      lstm_layers: 2,
      num_classes: 10,
      parameters: 796426,
      bidirectional: true
    },
    metrics: {
      synthetic_accuracy: 1.0,
      mean_cpu_latency_ms: 1.78,
      p95_cpu_latency_ms: 1.95,
      throughput_inferences_per_sec: 563.0
    },
    confidence_threshold: 0.75
  });
}
