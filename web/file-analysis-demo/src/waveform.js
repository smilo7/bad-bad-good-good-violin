import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js'

export function makeWaveform(wavFile, waveColor = '#383351') {
    if (window.wavesurfer) {
        window.wavesurfer.destroy();
    }

    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    
    // Reset button states for new waveform
    playBtn.style.display = 'inline-block';
    playBtn.disabled = true;
    pauseBtn.style.display = 'none';

    window.wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: 	waveColor,
        progressColor: '#5a5570',
        url: URL.createObjectURL(wavFile),
        height: 128,
        barWidth: 2,
        barGap: 1,
        barRadius: 2
      });
      
    // When the waveform is ready, enable the play button
    window.wavesurfer.on('ready', () => {
        playBtn.disabled = false;
    });

    // When playback starts, show the pause button
    window.wavesurfer.on('play', () => {
        playBtn.style.display = 'none';
        pauseBtn.style.display = 'inline-block';
    });
    
    // When playback is paused, show the play button
    window.wavesurfer.on('pause', () => {
        playBtn.style.display = 'inline-block';
        pauseBtn.style.display = 'none';
    });
    
    // When the track finishes, reset to play state and rewind
    window.wavesurfer.on('finish', () => {
        playBtn.style.display = 'inline-block';
        pauseBtn.style.display = 'none';
        window.wavesurfer.seekTo(0);
    });
    
    // Clicking on the waveform will also play it
    window.wavesurfer.on('interaction', () => {
        window.wavesurfer.play();
    });
}