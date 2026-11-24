// Run with: node app/test_dashboard.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "static/index.html"), "utf8");
const start = html.indexOf("async function pollRun()");
const end = html.indexOf("/**", start + 1);
assert(start >= 0 && end > start);
const pollSource = html.slice(start, end);

async function exercise(run) {
  const calls = [];
  const context = vm.createContext({
    run,
    calls,
    fetch: async () => ({ status: 200, ok: true, json: async () => ({ run }) }),
    showRun: () => calls.push("show"),
    loadDrones: async () => calls.push("drones"),
    loadChanges: async id => calls.push(["scan", id]),
    clearTimeout: () => {},
    setTimeout: (_, delay) => { calls.push(["timer", delay]); return 1; },
    document: { getElementById: () => ({ textContent: "" }) },
    window: { location: {} },
  });
  await vm.runInContext(`
    let pollTimer = null, linesShown = 0, shownRun = null, refreshedScanForRun = null;
    ${pollSource}
    (async () => { await pollRun(); await pollRun(); })();
  `, context);
  return JSON.parse(JSON.stringify(calls));
}

(async () => {
  assert.deepEqual(await exercise({ number: 1, status: "stopped", saved_scan_id: 3 }),
                   ["show", "drones", ["scan", 3], "show"]);
  assert.deepEqual(await exercise({ number: 2, status: "finished", saved_scan_id: null }),
                   ["show", "show"]);
  assert.deepEqual(await exercise({ number: 3, status: "stopping", saved_scan_id: null }),
                   ["show", ["timer", 1000], "show", ["timer", 1000]]);
  console.log("3 dashboard polling checks passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
