import { decodeAndResampleWavFile, chunkWaveform} from './audio.js';
import { softmax, runOnnxCombinedClassifier } from './onnx.js';
import { makePolarChart, makeLineChart, labelColors } from './chart.js';
import { makeWaveform } from './waveform.js';
import { MicRecorder } from './mic.js';


window.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const output = document.getElementById("output");
  const runBtn = document.getElementById("runBtn");
  const recordBtn = document.getElementById("recordBtn");
  const stopBtn = document.getElementById("stopBtn");
  const audioPlayer = document.getElementById("audioPlayer");
  let selectedFile = null;

  const mic = new MicRecorder(audioPlayer, output);

  recordBtn.addEventListener("click", async () => {
    recordBtn.disabled = true;
    stopBtn.disabled = false;
    await mic.start();
  });

  stopBtn.addEventListener("click", () => {
    mic.stop();
    recordBtn.disabled = false;
    stopBtn.disabled = true;
  });

  fileInput.addEventListener("change", (event) => {
    selectedFile = event.target.files[0];
    mic.clear(); // Clear mic recording if new file is chosen
    output.innerText = selectedFile ? "File loaded. Click 'Run Analysis'." : "❌ No file selected.";
    audioPlayer.src = selectedFile ? URL.createObjectURL(selectedFile) : "";
  });

  runBtn.addEventListener("click", async () => {

    let audioSource = null;
    if (mic.getBlob()) {
      audioSource = mic.getBlob();
      output.innerText = "🔄 Decoding mic recording...";
    } else if (selectedFile) {
      audioSource = selectedFile;
      output.innerText = "🔄 Loading and decoding WAV file...";
    } else {
      output.innerText = "❌ Please select a .wav file or record audio first.";
      return;
    }

    try {
      const signal = await decodeAndResampleWavFile(audioSource);
      
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
      let maxIdx = 0;
      predictions.forEach((probs, i) => {
        maxIdx = probs.indexOf(Math.max(...probs));
        const confidence = (probs[maxIdx] * 100).toFixed(1);
        text += `Chunk ${i + 1}: Class ${maxIdx} (Confidence: ${confidence}%)\n`;
        // Optionally, show all probabilities:
        // text += `  Probabilities: [${probs.map(p => p.toFixed(2)).join(", ")}]\n`;
      });
      output.innerText = text;
      
      let predictedLabelColor = labelColors[maxIdx];
      makeWaveform(selectedFile, predictedLabelColor);

      const ctxPolar = document.getElementById('polarChart').getContext('2d');
      const ctxLine = document.getElementById('lineChart').getContext('2d');
      makePolarChart(predictions[0], ctxPolar); //just show the first chunk's probabilities for now
      makeLineChart(predictions, ctxLine)
    } catch (err) {
      output.innerText = "❌ Error: " + err.message;
      console.error(err);
    }
  });
});