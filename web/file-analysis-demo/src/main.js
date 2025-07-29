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
  const playBtn = document.getElementById("playBtn");
  const pauseBtn = document.getElementById("pauseBtn");
  let selectedFile = null;

  const mic = new MicRecorder(output);

  // --- Button Event Listeners ---

  playBtn.addEventListener('click', () => {
    if (window.wavesurfer) {
      window.wavesurfer.play();
    }
  });

  pauseBtn.addEventListener('click', () => {
    if (window.wavesurfer) {
      window.wavesurfer.pause();
    }
  });
  
  recordBtn.addEventListener("click", async () => {
    // Clear any previous file selection or waveform
    selectedFile = null;
    fileInput.value = '';
    if (window.wavesurfer) {
      window.wavesurfer.destroy();
    }
    mic.clear();

    recordBtn.disabled = true;
    stopBtn.disabled = false;
    await mic.start();
  });

  stopBtn.addEventListener("click", async () => {
    const recordedBlob = await mic.stop();
    if (recordedBlob) {
      selectedFile = recordedBlob;
      makeWaveform(selectedFile); // Create waveform immediately
    }
    recordBtn.disabled = false;
    stopBtn.disabled = true;
  });

  fileInput.addEventListener("change", (event) => {
    selectedFile = event.target.files[0];
    mic.clear(); // Clear mic recording if new file is chosen
    if (selectedFile) {
        output.innerText = "File loaded. Click 'Run Analysis'.";
        makeWaveform(selectedFile); // Create waveform immediately
    } else {
        output.innerText = "❌ No file selected.";
    }
  });

  runBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      output.innerText = "❌ Please select a .wav file or record audio first.";
      return;
    }

    const audioSource = selectedFile; // This is our definitive audio source (File or Blob)

    try {
      output.innerText = "🔄 Loading and decoding audio file...";
      const signal = await decodeAndResampleWavFile(audioSource);
      
      const chunks = chunkWaveform(signal);

      output.innerText = `🧠 Running ONNX classifier...`;
      
      const logitsArray = await runOnnxCombinedClassifier(chunks, 32, async (batchIdx, totalBatches) => {
        console.log(`Running batch ${batchIdx + 1} of ${totalBatches}`);
        output.innerText = `🧠 Running ONNX classifier... Batch ${batchIdx + 1} of ${totalBatches}`;
        await new Promise(r => setTimeout(r, 0)); // Let the browser update the UI
      });

      const numClasses = logitsArray.length / chunks.length;
      const predictions = [];

      for (let i = 0; i < chunks.length; i++) {
        const logits = logitsArray.slice(i * numClasses, (i + 1) * numClasses);
        const probs = softmax(logits);
        predictions.push(probs);
      }

      let text = `✅ Predictions for ${predictions.length} chunks:\n`;
      let maxIdx = 0;
      predictions.forEach((probs, i) => {
        maxIdx = probs.indexOf(Math.max(...probs));
        const confidence = (probs[maxIdx] * 100).toFixed(1);
        text += `Chunk ${i + 1}: Class ${maxIdx} (Confidence: ${confidence}%)\n`;
      });
      output.innerText = text;
      
      let predictedLabelColor = labelColors[maxIdx];
      makeWaveform(audioSource, predictedLabelColor);

      const ctxPolar = document.getElementById('polarChart').getContext('2d');
      const ctxLine = document.getElementById('lineChart').getContext('2d');
      makePolarChart(predictions[0], ctxPolar); 
      makeLineChart(predictions, ctxLine)
    } catch (err) {
      output.innerText = "❌ Error: " + err.message;
      console.error(err);
    }
  });
});