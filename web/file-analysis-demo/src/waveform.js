import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js'

let wavesurfer = null;
export function makeWaveform(wavFile, predictedLabelColor) {
    if (wavesurfer) {
        wavesurfer.destroy();
    }

    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: 	predictedLabelColor,
        progressColor: '#383351',
        url: URL.createObjectURL(wavFile),
      })
      
      wavesurfer.on('interaction', () => {
        wavesurfer.play();
        console.log('do something with line chart');
    })

    wavesurfer.on('interaction', () => {
        wavesurfer.play();
    })
}