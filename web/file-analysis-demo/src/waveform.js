import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js'
import TimelinePlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/timeline.esm.js'
import RegionsPlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.esm.js'

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

    // Create an instance of the Regions plugin
    const regions = RegionsPlugin.create();

    window.wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: 	waveColor,
        progressColor: '#5a5570',
        url: URL.createObjectURL(wavFile),
        height: 128,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        plugins: [
            TimelinePlugin.create({
                container: '#wave-timeline',
                timeInterval: 1,            // Granularity of ticks
                primaryLabelInterval: 1,       // Major label every 1 second            
            }),
            regions
        ]
    });
    // Store regions instance for later access
    window.wavesurfer.regions = regions;
      
    window.wavesurfer.on('ready', () => { playBtn.disabled = false; });
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
    window.wavesurfer.on('interaction', () => { window.wavesurfer.play(); });
}

export function showPredictionsOnWaveform(predictions, hopDuration, classColors) {
    if (!window.wavesurfer || !window.wavesurfer.regions) return;

    const regionsPlugin = window.wavesurfer.regions;
    regionsPlugin.clearRegions();

    predictions.forEach((probs, i) => {
        const maxIdx = probs.indexOf(Math.max(...probs));
        const color = classColors[maxIdx]; // e.g., 'rgba(255, 99, 132, 1)'
        
        // Create a semi-transparent version of the color for the region fill
        const regionColor = color.replace(', 1)', ', 0.3)');

        regionsPlugin.addRegion({
            start: i * hopDuration,
            end: (i * hopDuration) + hopDuration,
            color: regionColor,
            drag: false,
            resize: false,
        });
    });
}