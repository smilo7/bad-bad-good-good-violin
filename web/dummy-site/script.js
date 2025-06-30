// Function to create a dummy input tensor for demonstration
// In a real application, this would be your preprocessed mel-spectrogram data.
function createDummyInput(height, width) {
    const data = new Float32Array(1 * 1 * height * width); // Batch, Channels, Height, Width
    for (let i = 0; i < data.length; i++) {
        data[i] = Math.random() * 2 - 1; // Random values between -1 and 1
    }
    // ONNX Runtime expects a flat array, and you define the shape separately.
    return new ort.Tensor('float32', data, [1, 1, height, width]);
}

async function runOnnxInference() {
    const outputDiv = document.getElementById('result');
    outputDiv.textContent = 'Loading model and running inference...';

    try {
        // 1. Create an ONNX Runtime session
        // Make sure the path to your .onnx file is correct.
        const session = await ort.InferenceSession.create('../../models/onnx/simple_cnn.onnx');

        // 2. Prepare the input data
        // Your model expects a 4D tensor: [batch_size, channels, height, width]
        // For your SmallerCNN, it's [1, 1, 128, 128] for a single grayscale spectrogram.
        const inputHeight = 128;
        const inputWidth = 128;
        const inputTensor = createDummyInput(inputHeight, inputWidth);

        // ONNX Runtime expects inputs as an object where keys match the input names
        // defined during export (e.g., "input").
        const feeds = { 'input': inputTensor };

        // 3. Run inference
        const results = await session.run(feeds);

        // 4. Process the output
        // The output names also match what you defined during export (e.g., "output").
        const outputTensor = results['output'];
        // outputTensor.data is a Float32Array containing your model's raw output (logits).
        // You'll likely want to apply a softmax if your model outputs logits for classification.

        // For demonstration, let's find the predicted class (assuming it's a classification model)
        const logits = outputTensor.data;
        let maxLogit = -Infinity;
        let predictedClass = -1;
        for (let i = 0; i < logits.length; i++) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
                predictedClass = i;
            }
        }

        outputDiv.textContent = `
        Model loaded and inference completed!
        Output shape: [${outputTensor.dims.join(', ')}]
        Raw logits: [${Array.from(logits.map(x => x.toFixed(4))).join(', ')}]
        Predicted Class (argmax): ${predictedClass}
        `;

    } catch (e) {
        outputDiv.textContent = `Error during ONNX inference: ${e.message}`;
        console.error(e);
    }
}

// Attach the function to a button click
document.addEventListener('DOMContentLoaded', () => {
    const runButton = document.getElementById('runInferenceButton');
    runButton.addEventListener('click', runOnnxInference);
});