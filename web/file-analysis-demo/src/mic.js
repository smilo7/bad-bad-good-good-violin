export class MicRecorder {
  constructor(audioPlayer, output) {
    this.audioPlayer = audioPlayer;
    this.output = output;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.recordedBlob = null;
  }

  async start() {
    this.audioChunks = [];
    this.recordedBlob = null;
    this.output.innerText = "🎤 Recording...";
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.mediaRecorder = new MediaRecorder(stream);

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.audioChunks.push(e.data);
    };

    this.mediaRecorder.onstop = () => {
      this.recordedBlob = new Blob(this.audioChunks, { type: "audio/wav" });
      this.audioPlayer.src = URL.createObjectURL(this.recordedBlob);
      this.output.innerText = "Recording complete. Click 'Run Essentia' to analyze.";
    };

    this.mediaRecorder.start();
  }

  stop() {
    if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
      this.mediaRecorder.stop();
      this.output.innerText = "⏹️ Stopped recording.";
    }
  }

  getBlob() {
    return this.recordedBlob;
  }

  clear() {
    this.recordedBlob = null;
    this.audioChunks = [];
  }
}