// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// see also advanced usage of importing ONNX Runtime Web:
// https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/importing_onnxruntime-web
// ES Module import syntax
import * as ort from 'onnxruntime-web';

/**
 * Gathers system and browser information for performance logging.
 * @returns {Promise<object>} A promise that resolves to an object with system information.
 */
async function getSystemInfo() {
    const info = {};
    // Modern approach: User-Agent Client Hints API
    if (navigator.userAgentData) {
        info.platform = navigator.userAgentData.platform;
        info.brands = navigator.userAgentData.brands.map(b => `${b.brand} ${b.version}`).join(', ');
        try {
            const highEntropyValues = await navigator.userAgentData.getHighEntropyValues(['architecture']);
            info.architecture = highEntropyValues.architecture;
        } catch (e) {
            info.architecture = `Could not retrieve: ${e.message}`;
        }
    } else {
        // Fallback for older browsers
        info.userAgent = navigator.userAgent;
        info.platform = navigator.platform;
    }
    return info;
}


/**
 * Loads an image from a URL and returns its ImageData.
 * Uses OffscreenCanvas if available, with a fallback to a regular canvas.
 * @param {string} url The URL of the image to load.
 * @returns {Promise<ImageData>} A promise that resolves to the ImageData of the loaded image.
 */
async function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "Anonymous"; // Handle potential CORS issues
    image.onload = () => {
      let canvas;
      // Check for OffscreenCanvas support
      if (typeof OffscreenCanvas !== 'undefined') {
        canvas = new OffscreenCanvas(image.width, image.height);
      } else {
        canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
      }
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0);
      resolve(ctx.getImageData(0, 0, image.width, image.height));
    };
    image.onerror = reject;
    image.src = url;
  });
}

/**
 * Converts RGB ImageData to a grayscale Float32Array.
 * @param {ImageData} imageData The input image data.
 * @returns {Float32Array} A flat array of grayscale values (normalized to 0-1).
 */
function rgb2gray(imageData) {
    const { data, width, height } = imageData;
    const grayData = new Float32Array(width * height);
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        // Using luminosity method for grayscale conversion
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        grayData[i / 4] = gray / 255.0; // Normalize to [0, 1]
    }
    return grayData;
}

/**
 * Creates an ONNX tensor from grayscale image data.
 * @param {Float32Array} grayData The grayscale image data.
 * @param {number[]} dims The dimensions of the tensor [batch, channels, height, width].
 * @returns {ort.Tensor} The created tensor.
 */
function defineTensorInput(grayData, dims) {
    return new ort.Tensor('float32', grayData, dims);
}

/**
 * Runs inference on the ONNX session.
 * @param {ort.InferenceSession} session The ONNX inference session.
 * @param {ort.Tensor} inputTensor The input tensor.
 * @returns {Promise<ort.InferenceSession.OnnxValueMapType>} A promise that resolves to the output of the session.
 */
async function runSession(session, inputTensor) {
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    return await session.run(feeds);
}


/**
 * Creates and initializes an ONNX Runtime Inference Session.
 * @returns {Promise<ort.InferenceSession>} A promise that resolves to the created session.
 */
async function startSession(provider) {
    // Create a new session and load the specific model.
    return await ort.InferenceSession.create('./data/superpoint_1637x2048.onnx', { executionProviders: [provider] });
}

/**
 * Triggers a download of the provided data as a JSON file.
 * @param {object} data The data to be downloaded.
 * @param {string} filename The name of the file to be downloaded.
 */
function downloadJson(data, filename = 'performance.json') {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * The main entry point of the application.
 */
async function main() {
    const resultsData = {};
    try {
        resultsData.systemInfo = await getSystemInfo();
        resultsData.performance = {};
        const perf = resultsData.performance;

        let startTime = performance.now();
        
        const session = await startSession('wasm');
        perf.sessionCreation = performance.now() - startTime;
        console.log('ONNX session started successfully.', session);

        const imageUrl = 'data/pinball_1024x1024.jpg';
        
        startTime = performance.now();
        const imageData = await loadImage(imageUrl);
        perf.imageLoading = performance.now() - startTime;
        console.log('Image loaded successfully.', imageData);

        startTime = performance.now();
        const grayData = rgb2gray(imageData);
        perf.grayscaleConversion = performance.now() - startTime;

        const dims = [1, 1, imageData.height, imageData.width];
        
        startTime = performance.now();
        const inputTensor = defineTensorInput(grayData, dims);
        perf.tensorCreation = performance.now() - startTime;
        
        startTime = performance.now();
        const results = await runSession(session, inputTensor);
        perf.inference = performance.now() - startTime;
        console.log('Inference results:', results);

        perf.totalTime = Object.values(perf).reduce((a, b) => a + b, 0);

        console.log('Results Data:', resultsData);
        downloadJson(resultsData, `performance_${new Date().toISOString()}.json`);
        
    } catch (e) {
        console.error(`An error occurred in the main function: ${e}`);
        resultsData.error = e.message;
        downloadJson(resultsData, `performance_error_${new Date().toISOString()}.json`);
    }
}

main();
