export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  return res.status(200).json({
    model_name: "SignBridge-ISL-Alphabet-MLP",
    version: "2.1.0",
    standard: "ISLRTC-Manual-Alphabet",
    accuracy: 1.0,
    num_classes: 24,
    classes: [
      "NEUTRAL", "A", "B", "C", "D", "E", "F", "G", "H", "I",
      "K", "L", "M", "N", "O", "P", "Q", "S", "T", "U", "V", "W", "X", "Y"
    ],
    dynamic_classes: ["J", "Z"],
    num_features: 44,
    inference_engine: "Client-Side WebAssembly / GPU + Serverless Edge"
  });
}
