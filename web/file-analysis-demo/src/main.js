import { decodeAndResampleWavFile, chunkWaveform, HOP_DURATION} from './audio.js';
import { softmax, runOnnxCombinedClassifier } from './onnx.js';
import { makePolarChart, makeLineChart, labelColors } from './chart.js';
import { makeWaveform, showPredictionsOnWaveform } from './waveform.js';
import { MicRecorder } from './mic.js';
import { renderLegend } from './chart.js';


window.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  fileInput.value = ''; // Clear any previous file selected
  const output = document.getElementById("output");
  const runBtn = document.getElementById("runBtn");
  const recordBtn = document.getElementById("recordBtn");
  const stopBtn = document.getElementById("stopBtn");
  const playBtn = document.getElementById("playBtn");
  const pauseBtn = document.getElementById("pauseBtn");
  let selectedFile = null;

  const mic = new MicRecorder(output);

  /**
   * Links waveform playback position to the polar chart, updating it dynamically.
   * @param {Array<Array<number>>} predictions - The array of prediction probabilities.
   * @param {number} hopDuration - The time duration between the start of each chunk.
   */
  function setupPlaybackAnalysis(predictions, hopDuration) {
      if (!window.wavesurfer) return;

      const ctxPolar = document.getElementById('polarChart').getContext('2d');
      let lastUpdatedChunkIndex = -1;

      const updateChartForTime = (time) => {
          const chunkIndex = Math.floor(time / hopDuration);

          // Update the chart only if the chunk is valid and has changed
          if (chunkIndex >= 0 && chunkIndex < predictions.length && chunkIndex !== lastUpdatedChunkIndex) {
              const predictionForChunk = predictions[chunkIndex];
              makePolarChart(predictionForChunk, ctxPolar);
              lastUpdatedChunkIndex = chunkIndex;
          }
      };

      // When a new waveform is loaded, WaveSurfer's .destroy() method is called,
      // which automatically removes all event listeners, preventing duplicates.
      
      // Fired continuously during playback
      window.wavesurfer.on('audioprocess', (currentTime) => {
          updateChartForTime(currentTime);
      });

      // Fired when the user seeks to a new position
      window.wavesurfer.on('seek', (progress) => {
          const newTime = progress * window.wavesurfer.getDuration();
          updateChartForTime(newTime);
          // When seeking while paused, audioprocess doesn't fire, so we need this.
      });
  }


  // --- Event Listeners ---
  playBtn.addEventListener('click', () => {
    if (window.wavesurfer) window.wavesurfer.play();
  });

  pauseBtn.addEventListener('click', () => {
    if (window.wavesurfer) window.wavesurfer.pause();
  });
  
  recordBtn.addEventListener("click", async () => {
    // Clear any previous file selection or waveform
    selectedFile = null;
    fileInput.value = '';
    if (window.wavesurfer) window.wavesurfer.destroy();
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

      let text = `✅ Analysis complete for ${predictions.length} audio segments.\n`;
      text += "See results on the waveform and charts below.";
      output.innerText = text;

      // --- Render Legend ---
      renderLegend('chartLegend');
      
      // --- Create timestamps and update visuals ---
      const timestamps = predictions.map((_, i) => (i * HOP_DURATION).toFixed(2));
      
      const ctxPolar = document.getElementById('polarChart').getContext('2d');
      const ctxLine = document.getElementById('lineChart').getContext('2d');
      
      // The polar chart now defaults to the first chunk's data on load
      makePolarChart(predictions[0], ctxPolar);
      makeLineChart(predictions, timestamps, ctxLine);
      showPredictionsOnWaveform(predictions, HOP_DURATION, labelColors);

      // Set up the dynamic updates for playback
      setupPlaybackAnalysis(predictions, HOP_DURATION);

    } catch (err) {
      output.innerText = "❌ Error: " + err.message;
      console.error(err);
    }
  });
});