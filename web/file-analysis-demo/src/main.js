import { decodeAndResampleWavFile, chunkWaveform} from './audio.js';
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
      // const { signal, sampleRate } = await decodeWavFile(selectedFile);

      const signal = await decodeAndResampleWavFile(selectedFile);
      
      const chunks = chunkWaveform(signal);

      output.innerText = `🧠 Running ONNX classifier...`;
      
      const logitsArray = await runOnnxCombinedClassifier(chunks, 32, async (batchIdx, totalBatches) => {
        console.log(`Running batch ${batchIdx + 1} of ${totalBatches}`);
        output.innerText = `🧠 Running ONNX classifier... Batch ${batchIdx + 1} of ${totalBatches}`;
        await new Promise(r => setTimeout(r, 0)); // Let the browser update the UI
      });

      // Each row in logitsArray corresponds to one chunk's logits
      const numClasses = logitsArray.length / chunks.length;
      const predictions = [];

      for (let i = 0; i < chunks.length; i++) {
        const logits = logitsArray.slice(i * numClasses, (i + 1) * numClasses);
        const probs = softmax(logits);
        predictions.push(probs);
      }

      // After predictions are computed
      let text = `✅ Predictions for ${predictions.length} chunks:\n`;
      predictions.forEach((probs, i) => {
        const maxIdx = probs.indexOf(Math.max(...probs));
        const confidence = (probs[maxIdx] * 100).toFixed(1);
        text += `Chunk ${i + 1}: Class ${maxIdx} (Confidence: ${confidence}%)\n`;
        // Optionally, show all probabilities:
        // text += `  Probabilities: [${probs.map(p => p.toFixed(2)).join(", ")}]\n`;
      });
      output.innerText = text;

      const ctxPolar = document.getElementById('polarChart').getContext('2d');
      makePolarChart(predictions[0], ctxPolar); //just show the first chunk's probabilities for now

//       let text = `
// Output shape: [${logits.length}]
// Raw logits: [${Array.from(logits).map(x => x.toFixed(3)).join(", ")}]
// Probabilities: [${probabilities.map(p => p.toFixed(3)).join(", ")}]
// Predicted Class: ${predictedClass} (Confidence: ${(probabilities[predictedClass] * 100).toFixed(2)}%)
//       `;
//       output.innerText = text;
    } catch (err) {
      output.innerText = "❌ Error: " + err.message;
      console.error(err);
    }
  });
});