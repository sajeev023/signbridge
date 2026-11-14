import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { alphabetInferenceEngine } from '../../../frontend/src/services/alphabetInferenceEngine';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const inputPath = path.resolve(__dirname, 'canonical_parity_input.json');
const outputPath = path.resolve(__dirname, 'cross_runtime_parity_results.json');

const rawData = fs.readFileSync(inputPath, 'utf-8');
const inputData = JSON.parse(rawData);

const tolerance = 1e-4;
const results: Record<string, any> = {};

let totalFeaturesTested = 0;
let totalPassed = 0;
let maxGlobalDiff = 0.0;

for (const [letter, data] of Object.entries<any>(inputData)) {
  const lms = data.landmarks;
  const pyFeats: number[] = data.python_features;
  const tsFeats: number[] = alphabetInferenceEngine.extractSingleHandFeatures(lms);

  const diffs: number[] = [];
  let letterPassed = true;
  let maxLetterDiff = 0.0;

  for (let i = 0; i < 42; i++) {
    const diff = Math.abs(pyFeats[i] - tsFeats[i]);
    diffs.push(diff);
    totalFeaturesTested++;
    if (diff > maxLetterDiff) maxLetterDiff = diff;
    if (diff > maxGlobalDiff) maxGlobalDiff = diff;

    if (diff <= tolerance) {
      totalPassed++;
    } else {
      letterPassed = false;
    }
  }

  results[letter] = {
    passed: letterPassed,
    max_difference: maxLetterDiff,
    within_tolerance: maxLetterDiff <= tolerance,
    feature_diffs: diffs,
    python_sample: pyFeats.slice(0, 5),
    typescript_sample: tsFeats.slice(0, 5)
  };
}

const summary = {
  timestamp: new Date().toISOString(),
  tolerance,
  total_features_tested: totalFeaturesTested,
  total_features_passed: totalPassed,
  feature_agreement_rate: (totalPassed / totalFeaturesTested) * 100.0,
  max_global_difference: maxGlobalDiff,
  all_letters_parity_passed: totalPassed === totalFeaturesTested,
  per_letter_results: results
};

fs.writeFileSync(outputPath, JSON.stringify(summary, null, 2), 'utf-8');
console.log(`Parity test complete:`);
console.log(`Features tested: ${totalFeaturesTested}`);
console.log(`Agreement rate: ${(totalPassed / totalFeaturesTested) * 100.0}%`);
console.log(`Max global difference: ${maxGlobalDiff.toExponential(4)}`);
console.log(`All letters pass within tolerance (${tolerance}): ${summary.all_letters_parity_passed}`);
