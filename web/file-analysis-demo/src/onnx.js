export function prepareOnnxInput(melSpectrogram, melBandsCount = 128, width = 128) {
  const height = melBandsCount;
  // Pad or trim spectrogram
  const paddedSpectrogram = [];
  for (let i = 0; i < width; i++) {
    if (i < melSpectrogram.length) {
      paddedSpectrogram.push(melSpectrogram[i]);
    } else {
      paddedSpectrogram.push(new Array(height).fill(1e-10));
    }
  }
  // Log-mel and normalization
  const logMelSpectrogram = [];
  for (let x = 0; x < width; x++) {
    const logFrame = [];
    for (let y = 0; y < height; y++) {
      let val = paddedSpectrogram[x][y];
      if (!isFinite(val) || val <= 0) val = 1e-10;
      logFrame.push(Math.log10(val));
    }
    logMelSpectrogram.push(logFrame);
  }
  const allLogVals = logMelSpectrogram.flat();
  const mean = allLogVals.reduce((a, b) => a + b, 0) / allLogVals.length;
  const std = Math.sqrt(allLogVals.reduce((a, b) => a + (b - mean) ** 2, 0) / allLogVals.length);
  const flatData = new Float32Array(1 * 1 * height * width);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const logVal = logMelSpectrogram[x][y];
      flatData[y * width + x] = (logVal - mean) / (std + 1e-6);
    }
  }
  return flatData;
}

export async function runOnnx(flatData, melBandsCount = 128, width = 128) {
  const inputTensor = new ort.Tensor("float32", flatData, [1, 1, melBandsCount, width]);
  const session = await ort.InferenceSession.create("models/onnx/simple_cnn.onnx");
  const feeds = { input: inputTensor };
  const results = await session.run(feeds);
  return results["output"].data;
}

export function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sumExps);
}

export async function runOnnxMelSpectrogramModel(waveformArray) {
    const session = await ort.InferenceSession.create('models/onnx/mel_spectrogram_model.onnx');
    const input = new ort.Tensor('float32', waveformArray, [1, waveformArray.length]);

    const feeds = { waveform: input };
    const results = await session.run(feeds);
    const melSpec = results.mel_spec.data;
    return melSpec;
}

export async function runOnnxCombinedClassifier(waveformArray) {
  const session = await ort.InferenceSession.create('models/onnx/audio_to_class.onnx');
  const input = new ort.Tensor('float32', waveformArray, [1, waveformArray.length]);

  const feeds = { waveform: input };
  const results = await session.run(feeds);
  const scores = results.class_scores.data;  // [logit_0, logit_1, ..., logit_n]
  return scores;
}