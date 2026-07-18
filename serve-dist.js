// Minimal static server for previewing the production build (npm run build) with the
// Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy headers that ONNX Runtime Web
// needs for real multithreading (SharedArrayBuffer). webpack-dev-server sets these for
// `npm start`, but the plain `dist/` output has no server config of its own - whatever
// host serves it in production must set the same two headers, or threads silently drop to 1.
const http = require('http');
const fs = require('fs');
const path = require('path');

const root = __dirname;
const port = process.env.PORT || 5000;

const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.mjs': 'text/javascript',
    '.wasm': 'application/wasm',
    '.json': 'application/json',
    '.jpg': 'image/jpeg',
    '.png': 'image/png',
    '.onnx': 'application/octet-stream'
};

const server = http.createServer((req, res) => {
    const urlPath = req.url === '/' ? '/index.html' : req.url;
    const filePath = path.join(root, decodeURIComponent(urlPath.split('?')[0]));

    if (!filePath.startsWith(root)) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
    }

    fs.readFile(filePath, (err, data) => {
        if (err) {
            res.writeHead(404);
            res.end('Not found');
            return;
        }
        res.writeHead(200, {
            'Content-Type': mimeTypes[path.extname(filePath)] || 'application/octet-stream',
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp'
        });
        res.end(data);
    });
});

server.listen(port, () => {
    console.log(`Preview server running at http://localhost:${port} (COOP/COEP enabled)`);
});
