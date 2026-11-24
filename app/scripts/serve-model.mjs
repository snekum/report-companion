// Serves converted MLC weights from ../model/mlc/<name>/ at
//   http://localhost:5300/<name>/resolve/main/<file>
// (the URL shape WebLLM expects), with CORS. For local testing only;
// hosting for the public app needs the owner's approval.
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { join, normalize } from "node:path";

const ROOT = new URL("../../model/mlc/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const PORT = Number(process.env.PORT ?? 5300);
createServer((req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  const m = decodeURIComponent(req.url.split("?")[0]).match(/^\/([^/]+)\/resolve\/main\/(.+)$/);
  if (!m) { res.writeHead(404).end(); return; }
  const file = normalize(join(ROOT, m[1], m[2]));
  if (!file.startsWith(normalize(ROOT))) { res.writeHead(403).end(); return; }
  try {
    const st = statSync(file);
    res.writeHead(200, { "Content-Length": st.size, "Content-Type": file.endsWith(".json") ? "application/json" : "application/octet-stream" });
    createReadStream(file).pipe(res);
  } catch { res.writeHead(404).end(); }
}).listen(PORT, () => console.log(`serving ${ROOT} on http://localhost:${PORT}/<name>/resolve/main/`));
