export async function decodeWavFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

	// we should resample if it is not 48000 Hz
	if (audioBuffer.sampleRate !== 48000) {
		const resampledBuffer = audioCtx.createBuffer(
			audioBuffer.numberOfChannels,
			audioBuffer.length * 48000 / audioBuffer.sampleRate,
			48000
		);
		for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
			const channelData = audioBuffer.getChannelData(channel);
			resampledBuffer.copyToChannel(channelData, channel, 0);
		}
  }

  return {
    signal: audioBuffer.getChannelData(0),
    sampleRate: audioBuffer.sampleRate
  };
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