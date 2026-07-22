// Extracts the pure-math functions from _KROK4_JS (comparison_report.py) and runs them
// against Python-generated reference data (gen_krok4_crosscheck.py output) to verify the
// JS reimplementation matches the real server-side ring_extraction math.
// Usage: node krok4_crosscheck.js <comparison_report.py path> <reference.json path>
const fs = require('fs');

const pyFile = fs.readFileSync(process.argv[2] || '../../src/comparison_report.py', 'utf8');
const marker = '_KROK4_JS = r"""';
const startIdx = pyFile.indexOf(marker);
if (startIdx === -1) { console.error('could not find _KROK4_JS marker'); process.exit(1); }
const bodyStart = startIdx + marker.length;
const endIdx = pyFile.indexOf('"""', bodyStart);
if (endIdx === -1) { console.error('could not find closing triple-quote'); process.exit(1); }
let js = pyFile.slice(bodyStart, endIdx).trim();
// Strip the outer IIFE wrapper "(function(){ ... })();" so the function declarations
// land in this scope; stub `document` so the (now-unreachable, KROK4_DATA undefined)
// DOM-mounting tail at the end is a harmless no-op.
js = js.replace(/^\(function\(\)\{/, '').replace(/\}\)\(\);\s*$/, '');
global.document = { getElementById: () => null };
eval(js);

const data = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const d = data.interactive;
const p = data.params;

const nDirs = d.contour_pts.length;
const densPk = rayPeaks(d.density_profiles, d.n_samples, p.prom, d.density_min_distance, d.inner_margin, d.edge_margin);
const classPk = rayPeaks(d.classical_profiles, d.n_samples, p.prom, d.classical_min_distance, d.inner_margin, d.edge_margin);
const dclust = clusterByRadiusWithArcs(densPk, p.tol, nDirs, 2);
const cclust = clusterByRadiusWithArcs(classPk, p.tol, nDirs, 2);
const merged = mergeClusters(dclust, cclust, p.tol);
const chosen = dpSelectT(merged, p.age, p.gap);

console.log('js_chosen=' + JSON.stringify(chosen));
console.log('python_chosen=' + JSON.stringify(data.python_chosen));

const EPS = 1e-6;
const py = data.python_chosen;
let ok = py.length === chosen.length && py.every((t, i) => Math.abs(t - chosen[i]) < EPS);
console.log(ok ? 'MATCH: JS reproduces Python exactly' : 'MISMATCH');
process.exit(ok ? 0 : 1);
