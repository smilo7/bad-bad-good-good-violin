export class MicRecorder {
  constructor(output) {
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

    this.mediaRecorder.start();
  }

  stop() {
    return new Promise((resolve) => {
      if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
        this.mediaRecorder.onstop = () => {
          this.recordedBlob = new Blob(this.audioChunks, { type: "audio/wav" });
          this.output.innerText = "Recording complete. Click 'Run Analysis' to analyze.";
          resolve(this.recordedBlob);
        };
        this.mediaRecorder.stop();
        this.output.innerText = "⏹️ Stopped recording.";
      } else {
        resolve(null);
      }
    });
  }

  getBlob() {
    return this.recordedBlob;
  }

  clear() {
    this.recordedBlob = null;
    this.audioChunks = [];
  }
}