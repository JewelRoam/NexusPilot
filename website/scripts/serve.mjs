import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(new URL("..", import.meta.url).pathname, process.argv.includes("--dist") ? "dist" : ".");
const mime = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".jpg": "image/jpeg",
  ".mp4": "video/mp4"
};
createServer(async (req, res) => {
  try {
    const path = decodeURIComponent((req.url || "/").split("?")[0]);
    const filePath = resolve(root, `.${path === "/" ? "/index.html" : path}`);
    const data = await readFile(filePath);
    res.writeHead(200, { "Content-Type": mime[extname(filePath)] || "application/octet-stream", "Content-Length": data.length, "Cache-Control": "no-store" });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
}).listen(4173, () => console.log("NexusPilot website: http://localhost:4173"));
