// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// see also advanced usage of importing ONNX Runtime Web:
// https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/importing_onnxruntime-web
// ES Module import syntax
import * as ort from 'onnxruntime-web';
//  Enable SIMD (if supported)
ort.env.wasm.simd = true;
// Use proxy worker (required for multithreading)
ort.env.wasm.proxy = true;

// Number of threads (limit to avoid oversubscription)
const hw = navigator.hardwareConcurrency || 4;
ort.env.wasm.numThreads = Math.min(8, hw);


/**
 * Gather system and browser information for performance logging.
 * @returns {Promise<Object>} Promise that resolves to an object with possible fields:
 *  - platform {string}
 *  - brands {string} (e.g. "Chromium 120, Google Chrome 120")
 *  - architecture {string}
 *  - userAgent {string} (fallback)
 * @throws {Error} If userAgentData.getHighEntropyValues fails.
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
 * Load an image from a URL and return it as an HTMLImageElement.
 * @param {string} url URL of the image to load.
 * @returns {Promise<HTMLImageElement>} Promise that resolves to the loaded image element.
 * @rejects {Event|Error} In case of loading errors.
 * @notes The image element has `crossOrigin` set to "Anonymous".
 */
async function loadImageElement(url) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.crossOrigin = "Anonymous";
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = url;
    });
}


/**
 * Convert an HTMLImageElement to ImageData.
 * @param {HTMLImageElement} image Input image element.
 * @returns {ImageData} Extracted ImageData.
 * @throws {Error} If 2D context is not available.
 * @notes Uses OffscreenCanvas when available, otherwise a DOM canvas is created.
 */
function imageToImageData(image) {
    let canvas;
    if (typeof OffscreenCanvas !== 'undefined') {
        canvas = new OffscreenCanvas(image.width, image.height);
    } else {
        canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
    }
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    return ctx.getImageData(0, 0, image.width, image.height);
}


/**
 * Convert RGB ImageData to a grayscale Float32Array.
 * @param {ImageData} imageData Source ImageData in RGBA format.
 * @returns {Float32Array} Flat Float32Array of normalized grayscale values in [0, 1],
 *                        length = width * height.
 * @notes Uses luminosity formula: 0.299 * R + 0.587 * G + 0.114 * B.
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
 * Create an ONNX tensor from grayscale image data.
 * @param {Float32Array} grayData Normalized grayscale data.
 * @param {number[]} dims Tensor dimensions, e.g. [1, 1, height, width].
 * @returns {ort.Tensor} ONNX tensor (dtype 'float32').
 */
function defineTensorInput(grayData, dims) {
    return new ort.Tensor('float32', grayData, dims);
}

/**
 * Run inference on an ONNX session.
 * @param {ort.InferenceSession} session Initialized ONNX InferenceSession.
 * @param {ort.Tensor} inputTensor Input tensor.
 * @returns {Promise<Object<string, ort.Tensor>>} Promise resolving to an output map
 *                                              (e.g. { semi: ort.Tensor, desc: ort.Tensor }).
 * @throws {Error} If session.run fails.
 */
async function runSession(session, inputTensor) {
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    return await session.run(feeds);
}


/**
 * Create and initialize an ONNX Runtime Inference Session.
 * @param {string} modelPath Relative path or URL to the ONNX model file.
 * @param {string} provider Execution provider name to use (e.g. 'wasm' or 'webgl').
 * @returns {Promise<ort.InferenceSession>} Promise that resolves to the ready session.
 * @throws {Error} If model loading or initialization fails.
 * @notes Passes options: { executionProviders: [provider], graphOptimizationLevel: 'all' }.
 */
async function startSession(modelPath, provider) {
    // Create a new session and load the specific model.
    return await ort.InferenceSession.create(modelPath, { executionProviders: [provider], graphOptimizationLevel: 'all' });
}

/**
 * Draw the original image and keypoints on the canvas from the model's heatmap output.
 * @param {HTMLCanvasElement} canvas Canvas to draw on.
 * @param {HTMLImageElement} image Original image (used for size/background).
 * @param {ort.Tensor} heatmapTensor Output 'semi' tensor from the model. Expect dims [1, C, H, W].
 * @param {number} [confidenceThreshold=0.015] Confidence threshold for drawing points.
 * @notes Performs an implicit "depth-to-space" arrangement: assumes the first 64 channels map
 *        to an 8x8 sub-pixel grid per heatmap cell.
 */
function drawKeypoints(canvas, image, heatmapTensor) {
    const ctx = canvas.getContext('2d');
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);

    const data = heatmapTensor.data;
    const dims = heatmapTensor.dims;
    const [channelCount, heatmapHeight, heatmapWidth] = [dims[1], dims[2], dims[3]];
    const confidenceThreshold = 0.015; // This is a crucial parameter to tune.

    // The model outputs a heatmap with 65 channels. The first 64 are for keypoints in an 8x8 grid.
    // We need to perform a "depth to space" operation to create a full-size heatmap.
    const fullSizeHeatmap = new Float32Array(image.width * image.height);
    const cellSize = 8;

    for (let c = 0; c < channelCount - 1; c++) { // Iterate through the 64 keypoint channels
        const subPixelY = Math.floor(c / cellSize);
        const subPixelX = c % cellSize;

        for (let y = 0; y < heatmapHeight; y++) {
            for (let x = 0; x < heatmapWidth; x++) {
                const heatmapIndex = (c * heatmapHeight * heatmapWidth) + (y * heatmapWidth) + x;
                const score = data[heatmapIndex];

                const finalX = x * cellSize + subPixelX;
                const finalY = y * cellSize + subPixelY;

                const fullMapIndex = finalY * image.width + finalX;
                fullSizeHeatmap[fullMapIndex] = score;
            }
        }
    }

    // Now, iterate through the full-size heatmap and draw points above the threshold.
    ctx.fillStyle = 'green';
    for (let i = 0; i < fullSizeHeatmap.length; i++) {
        if (fullSizeHeatmap[i] > confidenceThreshold) {
            const x = i % image.width;
            const y = Math.floor(i / image.width);
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI); // Draw a circle of radius 2
            ctx.fill();
        }
    }
}


/**
 * Trigger download of the provided data as a JSON file.
 * @param {Object} data Object to serialize as JSON.
 * @param {string} [filename='performance.json'] File name for the download.
 * @returns {void}
 * @notes Uses Blob and URL.createObjectURL and revokes the URL after download.
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
 * Main entry point of the application.
 * @returns {Promise<void>} Promise that resolves when the main flow completes.
 * @throws {Error} Any errors are caught internally and saved into the performance JSON.
 * @notes Records timings (ms) for: session creation, image loading, grayscale conversion,
 *        tensor creation and inference. Timings are expressed in milliseconds.
 */
async function main() {
    const resultsData = {};
    const modelPath = './data/superpoint_quantized.onnx';
    const imageUrl = 'data/pinball_1024x1024.jpg';

    try {
        resultsData.systemInfo = await getSystemInfo();
        resultsData.runContext = {
            model: modelPath.split('/').pop(),
            image: imageUrl.split('/').pop()
        };
        resultsData.performance = {};
        const perf = resultsData.performance;

        let startTime = performance.now();
        
        const session = await startSession(modelPath, 'wasm');
        perf.sessionCreation = performance.now() - startTime;
        console.log('ONNX session started successfully.', session);
        
        startTime = performance.now();
        const image = await loadImageElement(imageUrl);
        perf.imageLoading = performance.now() - startTime;
        console.log('Image loaded successfully.', image);

        const imageData = imageToImageData(image);

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

        const canvas = document.getElementById('output-canvas');
        // The model output is a map with 'semi' and 'desc'. 'semi' is the heatmap.
        const heatmapTensor = results['semi'];
        
        console.log('Heatmap Tensor:', heatmapTensor);
        drawKeypoints(canvas, image, heatmapTensor);

        console.log('Results Data:', resultsData);
        downloadJson(resultsData, `performance_${new Date().toISOString()}.json`);
        
    } catch (e) {
        console.error(`An error occurred in the main function: ${e}`);
        resultsData.error = e.message;
        downloadJson(resultsData, `performance_error_${new Date().toISOString()}.json`);
    }
}

main();
