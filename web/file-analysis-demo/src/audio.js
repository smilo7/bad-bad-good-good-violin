export const CHUNK_DURATION = 0.5;
export const HOP_DURATION = 0.25;


export async function decodeAndResampleWavFile(file, targetSampleRate = 48000) {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);

  if (decoded.sampleRate === targetSampleRate) {
    return decoded.getChannelData(0); // mono
  }

  // Resample
  const offlineCtx = new OfflineAudioContext(1, decoded.length * targetSampleRate / decoded.sampleRate, targetSampleRate);
  const source = offlineCtx.createBufferSource();
  source.buffer = decoded;
  source.connect(offlineCtx.destination);
  source.start(0);

  const rendered = await offlineCtx.startRendering();
  return rendered.getChannelData(0); // mono
}

export function chunkWaveform(signal, sampleRate = 48000, chunkDuration = CHUNK_DURATION, hopDuration = HOP_DURATION) {
  const chunkSize = Math.floor(chunkDuration * sampleRate); // e.g., 24000
  const hopSize = Math.floor(hopDuration * sampleRate);     // e.g., 4800 for 0.1s hopDuration

  const chunks = [];
  for (let start = 0; start <= signal.length - hopSize; start += hopSize) {
    const end = start + chunkSize;
    const chunk = new Float32Array(chunkSize);

    if (end <= signal.length) {
      // Normal full chunk
      chunk.set(signal.slice(start, end));
    } else {
      // Pad last chunk with zeros
      const slice = signal.slice(start);
      chunk.set(slice, 0);
      // rest is already zero-filled by default
    }

    chunks.push(chunk);
  }

  return chunks;
}

export async function extractMelSpectrogram(signal, sampleRate, frameSize = 1024, hopSize = 512, melBandsCount = 128) {
  const wasmModule = await EssentiaWASM();
  const extractor = new EssentiaExtractor(wasmModule);

  // Clone the default profile and override melBandsCount
  const config = JSON.parse(JSON.stringify(extractor.profile));
  config.MelBands.numberBands = melBandsCount;

  const melSpectrogram = [];
  for (let i = 0; i + frameSize <= signal.length; i += hopSize) {
    const frame = signal.slice(i, i + frameSize);
    const melBands = extractor.melSpectrumExtractor(frame, sampleRate, false, config);
    melSpectrogram.push(melBands);
  }
  return melSpectrogram;
}