// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// see also advanced usage of importing ONNX Runtime Web:
// https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/importing_onnxruntime-web
// ES Module import syntax
import * as ort from 'onnxruntime-web';
// onnxruntime-web >=1.17 ships a single ort-wasm-simd-threaded.wasm build, so SIMD is
// always used when the browser supports it - there is no non-SIMD binary to opt into.

// Use proxy worker (required for multithreading)
ort.env.wasm.proxy = true;

// Number of threads (limit to avoid oversubscription).
// Actual multithreading only kicks in when the page is cross-origin isolated
// (COOP/COEP response headers -> SharedArrayBuffer available). If it isn't,
// ORT silently falls back to 1 thread regardless of this value.
const hw = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
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
 * Report the facts that determine whether SIMD/multithreaded WASM is actually
 * active, since neither is directly observable from the InferenceSession itself.
 * @returns {Object} crossOriginIsolated, hardwareConcurrency, requestedThreads,
 *                    and simdLikelySupported (WebAssembly SIMD feature-detect).
 */
async function getWasmRuntimeInfo() {
    let simdLikelySupported = false;
    try {
        // Minimal valid WASM module containing a v128.const SIMD instruction.
        // WebAssembly.validate returns false if the engine can't parse SIMD opcodes.
        const simdTestBytes = Uint8Array.from([
            0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
            10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
        ]);
        simdLikelySupported = WebAssembly.validate(simdTestBytes);
    } catch (e) {
        simdLikelySupported = false;
    }

    return {
        crossOriginIsolated: typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : null,
        hardwareConcurrency: navigator.hardwareConcurrency || null,
        requestedThreads: ort.env.wasm.numThreads,
        simdLikelySupported
    };
}

// Note: with `ort.env.wasm.proxy = true`, ORT fetches/compiles the .wasm binary inside a
// dedicated Worker, which has its own isolated Performance timeline - the main document's
// `performance.getEntriesByType('resource')` never sees that request, so per-binary load
// timing isn't observable from here. `perf.sessionCreation` below already spans the full
// round trip (fetch + compile + thread-pool init inside the worker) and is the reliable figure.

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
 * Decode the model's heatmap tensor into full-resolution corner coordinates.
 * Pure computation, no canvas access - this "depth to space" unfold + threshold is the
 * actual corner-localization step, kept separate from rendering so its cost (the thing
 * people mean by "corner detection time") can be measured on its own, apart from the
 * neural net's forward pass (see `runSession`/`perf.inference`) and canvas drawing.
 * @param {ort.Tensor} heatmapTensor Output 'semi' tensor from the model. Expect dims [1, C, H, W].
 * @param {number} imageWidth Full-resolution image width.
 * @param {number} imageHeight Full-resolution image height.
 * @param {number} [confidenceThreshold=0.015] Minimum score to keep a point. Crucial parameter to tune.
 * @returns {Array<{x: number, y: number, score: number}>} Detected corner points.
 */
function extractKeypoints(heatmapTensor, imageWidth, imageHeight, confidenceThreshold = 0.015) {
    const data = heatmapTensor.data;
    const dims = heatmapTensor.dims;
    const [channelCount, heatmapHeight, heatmapWidth] = [dims[1], dims[2], dims[3]];

    // The model outputs a heatmap with 65 channels. The first 64 are for keypoints in an 8x8 grid.
    // We need to perform a "depth to space" operation to create a full-size heatmap.
    const fullSizeHeatmap = new Float32Array(imageWidth * imageHeight);
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

                const fullMapIndex = finalY * imageWidth + finalX;
                fullSizeHeatmap[fullMapIndex] = score;
            }
        }
    }

    const keypoints = [];
    for (let i = 0; i < fullSizeHeatmap.length; i++) {
        const score = fullSizeHeatmap[i];
        if (score > confidenceThreshold) {
            keypoints.push({ x: i % imageWidth, y: Math.floor(i / imageWidth), score });
        }
    }
    return keypoints;
}

/**
 * Draw the original image and previously-extracted keypoints on the canvas.
 * @param {HTMLCanvasElement} canvas Canvas to draw on.
 * @param {HTMLImageElement} image Original image (used for size/background).
 * @param {Array<{x: number, y: number}>} keypoints Points from `extractKeypoints`.
 */
function renderKeypoints(canvas, image, keypoints) {
    const ctx = canvas.getContext('2d');
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);

    ctx.fillStyle = 'green';
    for (const { x, y } of keypoints) {
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI); // Draw a circle of radius 2
        ctx.fill();
    }
}


// Above this, an uploaded image's early conv activations (before the network's own
// 8x downsampling kicks in) scale with width * height * channels, and can exhaust
// WASM's linear memory. Not a correctness issue like alignment - a resource limit.
const MAX_UPLOAD_DIMENSION = 2048;

/**
 * Downscale an image if either dimension exceeds MAX_UPLOAD_DIMENSION, preserving
 * aspect ratio. Returns the original image unchanged if it's already within bounds.
 * @param {HTMLImageElement} image Source image.
 * @returns {HTMLImageElement|HTMLCanvasElement} The original image, or a canvas holding
 *          the downscaled copy.
 */
function capImageDimensions(image) {
    if (image.width <= MAX_UPLOAD_DIMENSION && image.height <= MAX_UPLOAD_DIMENSION) {
        return image;
    }
    const scale = MAX_UPLOAD_DIMENSION / Math.max(image.width, image.height);
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(image.width * scale);
    canvas.height = Math.round(image.height * scale);
    canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas;
}

/**
 * Run the full detection pipeline against an already-loaded image and render it.
 * Shared by the default-image run and the upload handler so both go through the
 * exact same code path (downscale cap -> grayscale -> tensor -> inference ->
 * corner extraction -> render).
 * @param {ort.InferenceSession} session Ready ONNX InferenceSession.
 * @param {HTMLImageElement} image Loaded image to process.
 * @param {HTMLCanvasElement} canvas Canvas to render keypoints onto.
 * @returns {Promise<Object>} { timings: {grayscaleConversion, tensorCreation, inference,
 *          keypointExtraction}, keypointCount, imageSize }.
 */
async function processImage(session, image, canvas) {
    const timings = {};
    const source = capImageDimensions(image);
    const imageData = imageToImageData(source);

    let startTime = performance.now();
    const grayData = rgb2gray(imageData);
    timings.grayscaleConversion = performance.now() - startTime;

    const dims = [1, 1, imageData.height, imageData.width];

    startTime = performance.now();
    const inputTensor = defineTensorInput(grayData, dims);
    timings.tensorCreation = performance.now() - startTime;

    startTime = performance.now();
    const results = await runSession(session, inputTensor);
    timings.inference = performance.now() - startTime;

    const heatmapTensor = results['semi'];

    startTime = performance.now();
    const keypoints = extractKeypoints(heatmapTensor, source.width, source.height);
    timings.keypointExtraction = performance.now() - startTime;

    renderKeypoints(canvas, source, keypoints);

    return { timings, keypointCount: keypoints.length, imageSize: `${source.width}x${source.height}` };
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
 * Wire the file input so users can run detection on their own image, reusing the
 * already-created session. Demonstrates that the canvas/pipeline size to whatever
 * image is loaded - the bundled demo image just happens to be square.
 * @param {ort.InferenceSession} session Ready ONNX InferenceSession.
 * @param {HTMLCanvasElement} canvas Canvas to render keypoints onto.
 * @returns {void}
 */
function setupImageUpload(session, canvas) {
    const uploadInput = document.getElementById('image-upload');
    const statusEl = document.getElementById('status');
    if (!uploadInput) return;

    uploadInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        uploadInput.disabled = true;
        if (statusEl) statusEl.textContent = 'Processing...';

        const objectUrl = URL.createObjectURL(file);
        try {
            const image = await loadImageElement(objectUrl);
            const { timings, keypointCount, imageSize } = await processImage(session, image, canvas);
            console.log(`Detected ${keypointCount} corners in uploaded image.`, timings);
            if (statusEl) {
                statusEl.textContent = `Uploaded image (${imageSize}): ${keypointCount} corners detected ` +
                    `(inference ${timings.inference.toFixed(1)} ms, extraction ${timings.keypointExtraction.toFixed(1)} ms).`;
            }
        } catch (e) {
            console.error('Error processing uploaded image:', e);
            if (statusEl) statusEl.textContent = `Error: ${e.message}`;
        } finally {
            URL.revokeObjectURL(objectUrl);
            uploadInput.disabled = false;
        }
    });

    uploadInput.disabled = false;
}

/**
 * Main entry point of the application.
 * @returns {Promise<void>} Promise that resolves when the main flow completes.
 * @throws {Error} Any errors are caught internally and saved into the performance JSON.
 * @notes Records timings (ms) for: session creation, image loading, grayscale conversion,
 *        tensor creation, inference (NN forward pass), and keypoint extraction (corner
 *        localization from the heatmap). Timings are expressed in milliseconds.
 */
async function main() {
    const resultsData = {};
    const modelPath = './data/superpoint_quantized.onnx';
    const imageUrl = 'data/pinball.jpg';
    const canvas = document.getElementById('output-canvas');
    const statusEl = document.getElementById('status');

    try {
        resultsData.systemInfo = await getSystemInfo();
        resultsData.wasmRuntimeInfo = await getWasmRuntimeInfo();
        console.log('WASM runtime info:', resultsData.wasmRuntimeInfo);
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

        const { timings, keypointCount, imageSize } = await processImage(session, image, canvas);
        Object.assign(perf, timings);
        perf.totalTime = Object.values(perf).reduce((a, b) => a + b, 0);
        resultsData.keypointCount = keypointCount;
        console.log(`Detected ${keypointCount} corners.`);

        console.log('Results Data:', resultsData);
        downloadJson(resultsData, `performance_${new Date().toISOString()}.json`);

        if (statusEl) statusEl.textContent = `Default image (${imageSize}): ${keypointCount} corners detected.`;
        setupImageUpload(session, canvas);

    } catch (e) {
        console.error(`An error occurred in the main function: ${e}`);
        resultsData.error = e.message;
        downloadJson(resultsData, `performance_error_${new Date().toISOString()}.json`);
        if (statusEl) statusEl.textContent = `Error: ${e.message}`;
    }
}

main();
