import { decodeWavFile} from './audio.js';
import { softmax, runOnnxCombinedClassifier } from './onnx.js';
import { makePolarChart } from './chart.js';

window.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const output = document.getElementById("output");
  const runBtn = document.getElementById("runBtn");
  let selectedFile = null;

  fileInput.addEventListener("change", (event) => {
    selectedFile = event.target.files[0];
    output.innerText = selectedFile ? "File loaded. Click 'Run Essentia'." : "❌ No file selected.";
  });

  runBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      output.innerText = "❌ Please select a .wav file first.";
      return;
    }
    output.innerText = "🔄 Loading and decoding WAV file...";
    try {
      const { signal, sampleRate } = await decodeWavFile(selectedFile);

      const logits = await runOnnxCombinedClassifier(signal)

      const probabilities = softmax(logits);

      let maxLogit = -Infinity;
      let predictedClass = -1;
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxLogit) {
          maxLogit = logits[i];
          predictedClass = i;
        }
      }

      const ctxPolar = document.getElementById('polarChart').getContext('2d');
      makePolarChart(probabilities, ctxPolar);

      let text = `
Output shape: [${logits.length}]
Raw logits: [${Array.from(logits).map(x => x.toFixed(3)).join(", ")}]
Probabilities: [${probabilities.map(p => p.toFixed(3)).join(", ")}]
Predicted Class: ${predictedClass} (Confidence: ${(probabilities[predictedClass] * 100).toFixed(2)}%)
      `;
      output.innerText = text;
    } catch (err) {
      output.innerText = "❌ Error: " + err.message;
      console.error(err);
    }
  });
});