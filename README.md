# SuperPoint ONNX Bundler 🚀

This project demonstrates how to run the SuperPoint network for feature detection in the browser using the ONNX Runtime. The application is bundled for the web using Webpack.

For more information on the SuperPoint architecture, see the paper:
[SuperPoint: Self-Supervised Interest Point Detection and Description](https://doi.org/10.1109/CVPRW.2018.00060)

## 🛠️ Technologies Used

*   **ONNX Runtime Web**: To run the SuperPoint ONNX model in the browser.
*   **Webpack**: To bundle the application and its dependencies.
*   **Webpack Dev Server**: For a live-reloading development environment.
*   **JavaScript (ES Modules)**: For the main application logic.

## ⚙️ Setup and Installation

To get started, clone the repository and install the necessary dependencies using npm:

```bash
git clone <repository-url>
cd superpoint_infer_bundler
npm install
```

## ▶️ Running the Project

First, build the project. This will create the `dist` directory with the bundled application and copy the necessary `.wasm` files.

```bash
npm run build
```

Then, to start the development server, run the following command:

```bash
npm start
```

This will open a new browser tab with the application running at `http://localhost:8080`.

To preview the production build (`dist/`) instead, with the headers ONNX Runtime Web needs for multithreading:

```bash
npm run build
npm run preview
```

This opens `http://localhost:5000`.

## ⚡ SIMD & Multithreading

*   **SIMD** is always active when the browser's WebAssembly engine supports it: `onnxruntime-web` ships a single `ort-wasm-simd-threaded.wasm` binary (no separate SIMD/non-SIMD builds), so there's nothing to toggle.
*   **Multithreading** (`ort.env.wasm.numThreads` in `main.js`) only takes effect when the page is **cross-origin isolated**, which requires the response headers `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. Without them, ORT silently falls back to a single thread. `webpack-dev-server` sets these automatically (see `webpack.config.js`), and `npm run preview` (via `serve-dist.js`) replicates them for the production build - **whatever host serves `dist/` in production must set the same two headers**, or threading will silently degrade.
*   Running the app writes a `performance_*.json` file (see `downloadJson` in `main.js`) containing `wasmRuntimeInfo` (`crossOriginIsolated`, `hardwareConcurrency`, `requestedThreads`, `simdLikelySupported`) and `performance.sessionCreation` (ms), useful for confirming SIMD/threads are actually active and for measuring session init cost. Per-binary WASM fetch timing isn't observable from the main document because `ort.env.wasm.proxy = true` loads the binary inside a Worker with its own isolated Performance timeline - `sessionCreation` is the reliable end-to-end figure instead.

## 📦 Project Structure

*   `main.js`: The main entry point of the application. It handles loading the ONNX model, pre-processing the image data, and running the inference.
*   `webpack.config.js`: The configuration file for Webpack. It defines how the application is bundled and sets up the development server.
*   `serve-dist.js`: Minimal static server for previewing the production build with the COOP/COEP headers required for WASM multithreading.
*   `index.html`: The main HTML file that loads the bundled JavaScript application.
*   `package.json`: Lists the project's dependencies and defines the `npm` scripts.

## 🧠 ONNX Model Details

The bundled model (`data/superpoint_quantized.onnx`) is **dynamically quantized to `uint8`**, produced by [`superpoint_infer_engine`](https://github.com/kalwalt/superpoint_infer_engine/tree/feature-dynamic_axes)'s `convert_onnx.py` via `onnxruntime.quantization.quantize_dynamic`. Each conv layer is replaced with a `ConvInteger` op, with activations quantized at runtime via `DynamicQuantizeLinear` (no calibration data needed) - see that repo's Readme for the full mechanism.

### Quantized vs float32 benchmark (this runtime)

Measured with a standalone benchmark page (same 1024x1024 input, 2 warm-up + 8 timed `session.run()` calls per model, `onnxruntime-web` WASM backend, SIMD + 8 threads, `crossOriginIsolated: true`):

| | float32 | quantized (dynamic, `ConvInteger`) |
|---|---|---|
| Session creation | 442 ms | 101 ms |
| Inference (mean) | 1794.7 ms | 1474.4 ms |
| Inference (median) | 1804.3 ms | 1478.9 ms |

Quantization is a real but modest win here: **~18% faster inference**, plus a much faster session load (~4x smaller file: 1.3 MB vs 5.2 MB) - roughly 30% less total time per run including model load. That's well short of the 2-4x speedups int8 quantization gets on runtimes with fully-vectorized integer kernels, which suggests `ConvInteger` in onnxruntime-web's WASM backend gets *some* acceleration but likely not a fully SIMD-optimized path (inferred from the gap, not confirmed against ORT's WASM kernel source). This result is specific to the WASM/browser backend - it doesn't necessarily hold for the OpenVINO/Movidius targets that `superpoint_infer_engine` also produces artifacts for, since those don't consume this quantized graph.

## 🔮 Future Development

Here are some ideas for future improvements:

- [ ] **Implement Non-Maximum Suppression (NMS)**: To refine the keypoint detection by removing redundant, overlapping points and keeping only the most confident one in a local area.
- [ ] **Add a UI for Image Upload**: Allow users to upload their own images for feature detection.
- [ ] **Visualize Descriptors**: Add a feature to visualize the feature descriptors associated with each keypoint.
