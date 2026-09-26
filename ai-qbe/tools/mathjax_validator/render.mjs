#!/usr/bin/env node
/**
 * Renders one or more LaTeX/mhchem snippets through real MathJax
 * (server-side, via mathjax-full's SVG output with a lite DOM adaptor --
 * no browser needed) and reports whether each one rendered cleanly.
 *
 * This is invoked as a subprocess by backend/validation/notation.py; it
 * is the one place in AI-QBE that answers "does this actually render in
 * MathJax" with the real library, rather than an approximation.
 *
 * Package selection: AllPackages minus 'noundefined'. 'noundefined'
 * makes MathJax silently render an unrecognized macro as inert red text
 * instead of raising a real error -- exactly the failure mode this
 * validator exists to catch (master prompt section 19: "unsupported
 * MathJax commands"), so it is deliberately excluded. Every other
 * package (mhchem, physics, ams matrices/environments, bbox, cancel,
 * etc.) stays enabled so legitimate academic notation isn't rejected.
 *
 * Usage: node render.mjs <<< '["x^2+y^2=r^2", "\\frac{a}{b"]'
 * Output (stdout): a JSON array, one result object per input, in order:
 *   {"ok": true}
 *   {"ok": false, "error": "Missing close brace"}
 */

import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { SVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import { AllPackages } from "mathjax-full/js/input/tex/AllPackages.js";

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const packages = AllPackages.filter((p) => p !== "noundefined");
const tex = new TeX({ packages });
const svgOutput = new SVG();
const doc = mathjax.document("", { InputJax: tex, OutputJax: svgOutput });

const ERROR_ATTR_RE = /data-mjx-error="([^"]*)"/;

function renderOne(latex) {
  try {
    const node = doc.convert(latex, { display: true });
    const html = adaptor.outerHTML(node);
    const match = html.match(ERROR_ATTR_RE);
    if (match) {
      return { ok: false, error: match[1] };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err && err.message ? err.message : String(err) };
  }
}

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => (data += chunk));
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}

const input = await readStdin();
let snippets;
try {
  snippets = JSON.parse(input);
} catch (err) {
  console.error(JSON.stringify({ fatal: `invalid input JSON: ${err.message}` }));
  process.exit(1);
}
if (!Array.isArray(snippets)) {
  console.error(JSON.stringify({ fatal: "input must be a JSON array of strings" }));
  process.exit(1);
}

const results = snippets.map((s) => renderOne(String(s)));
process.stdout.write(JSON.stringify(results));
