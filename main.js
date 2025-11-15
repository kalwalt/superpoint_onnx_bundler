// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// see also advanced usage of importing ONNX Runtime Web:
// https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/importing_onnxruntime-web
// ES Module import syntax
import * as ort from 'onnxruntime-web';

/**
 * Loads an image from a URL and returns its ImageData.
 * @param {string} url The URL of the image to load.
 * @returns {Promise<ImageData>} A promise that resolves to the ImageData of the loaded image.
 */
async function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "Anonymous"; // Handle potential CORS issues
    image.onload = () => {
      const canvas = new OffscreenCanvas(image.width, image.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0);
      resolve(ctx.getImageData(0, 0, image.width, image.height));
    };
    image.onerror = reject;
    image.src = url;
  });
}

/**
 * Creates and initializes an ONNX Runtime Inference Session.
 * @returns {Promise<ort.InferenceSession>} A promise that resolves to the created session.
 */
async function startSession() {
    // Create a new session and load the specific model.
    return await ort.InferenceSession.create('./superpoint_1637x2048.onnx', { executionProviders: ['wasm'] });
}

/**
 * The main entry point of the application.
 */
async function main() {
    try {
        const session = await startSession();
        console.log('ONNX session started successfully.', session);

        // You can now use the session for inference.
        // For example, load an image and run the model:
        // const imageData = await loadImage('your-image-url.jpg');
        // console.log('Image loaded successfully.', imageData);
        
        // Add your pre-processing, inference, and post-processing logic here.

    } catch (e) {
        console.error(`An error occurred in the main function: ${e}`);
    }
}

main();
