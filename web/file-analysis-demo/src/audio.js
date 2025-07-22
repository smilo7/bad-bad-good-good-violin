export async function decodeWavFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  return {
    signal: audioBuffer.getChannelData(0),
    sampleRate: audioBuffer.sampleRate
  };
}

export async function extractMelSpectrogram(signal, sampleRate, frameSize = 1024, hopSize = 512, melBandsCount = 128) {
  const wasmModule = await EssentiaWASM();
  const extractor = new EssentiaExtractor(wasmModule);
  const melSpectrogram = [];
  for (let i = 0; i + frameSize <= signal.length; i += hopSize) {
    const frame = signal.slice(i, i + frameSize);
    const melBands = extractor.melSpectrumExtractor(frame, sampleRate);
    melSpectrogram.push(melBands);
  }
  return melSpectrogram;
}