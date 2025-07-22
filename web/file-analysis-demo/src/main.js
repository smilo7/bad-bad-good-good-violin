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
      // Decode WAV file
      const arrayBuffer = await selectedFile.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const signal = audioBuffer.getChannelData(0); // mono
      const sampleRate = audioBuffer.sampleRate;

      // Initialize Essentia
      const wasmModule = await EssentiaWASM();
      const extractor = new EssentiaExtractor(wasmModule);

      // Parameters for framing
      const frameSize = 1024;
      const hopSize = 512;
      const melBandsCount = 128;
      const melSpectrogram = [];

      // Frame-by-frame mel spectrum extraction
      for (let i = 0; i + frameSize <= signal.length; i += hopSize) {
        const frame = signal.slice(i, i + frameSize);
        const melBands = extractor.melSpectrumExtractor(frame, sampleRate);
        melSpectrogram.push(melBands); // single channel
      }

      // Display melspectrogram result
      let text = `Processed ${melSpectrogram.length} frames at ${sampleRate} Hz\n`;
      for (let i = 0; i < Math.min(5, melSpectrogram.length); i++) {
        const bands = melSpectrogram[i].slice(0, 5).map(x => x.toFixed(2)).join(", ");
        text += `Frame ${i + 1}: ${bands}\n`;
      }

      // Format into ONNX input
      const height = melBandsCount;
      const width = 128;
      const paddedSpectrogram = [];
      for (let i = 0; i < width; i++) {
        if (i < melSpectrogram.length) {
          paddedSpectrogram.push(melSpectrogram[i]);
        } else {
          paddedSpectrogram.push(new Array(height).fill(1e-10)); // pad with small value
        }
      }

      // Compute log-mel
      const logMelSpectrogram = [];
      for (let x = 0; x < width; x++) {
        const logFrame = [];
        for (let y = 0; y < height; y++) {
          let val = paddedSpectrogram[x][y];
          if (!isFinite(val) || val <= 0) val = 1e-10;
          const logVal = Math.log10(val);
          logFrame.push(logVal);
        }
        logMelSpectrogram.push(logFrame);
      }

      // Flatten for mean/std calculation
      const allLogVals = logMelSpectrogram.flat();
      const mean = allLogVals.reduce((a, b) => a + b, 0) / allLogVals.length;
      const std = Math.sqrt(allLogVals.reduce((a, b) => a + (b - mean) ** 2, 0) / allLogVals.length);

      // Normalize using mean/std
      const flatData = new Float32Array(1 * 1 * height * width);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const logVal = logMelSpectrogram[x][y];
          flatData[y * width + x] = (logVal - mean) / (std + 1e-6);
        }
      }

      // Prepare ONNX input tensor
      const inputTensor = new ort.Tensor("float32", flatData, [1, 1, height, width]);

      // Run ONNX model
      const session = await ort.InferenceSession.create("../../models/onnx/simple_cnn.onnx");
      const feeds = { input: inputTensor };
      const results = await session.run(feeds);
      const outputTensor = results["output"];
      const logits = outputTensor.data;

      // Softmax function
      function softmax(logits) {
        const maxLogit = Math.max(...logits);
        const exps = logits.map(x => Math.exp(x - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sumExps);
      }

      const probabilities = softmax(logits);

      // Find predicted class
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


			console.log('chart rendered.');

      text += `
			Output shape: [${outputTensor.dims.join(", ")}]
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