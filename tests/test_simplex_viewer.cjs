const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {setImmediate} = require("node:timers/promises");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname,
  "../experiments/nonergodic_mess3_reward_state_action_symmetry_cycle_5/simplex_viewer/viewer.js"), "utf8");

async function viewer() {
  const elements = new Map();
  const element = () => ({
    value: "", children: [], textContent: "", hidden: false, handlers: {},
    replaceChildren(...children) { this.children = children; },
    setAttribute() {},
    addEventListener(name, callback) { this.handlers[name] = callback; },
    on() {},
  });
  const get = id => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  for (const [id, value] of Object.entries({
    run: "0", mode: "sequence", layer: "post_final_norm", checkpoint: "final", sequence: "0", time: "0",
  })) get(id).value = value;
  const targets = Array.from({length: 128}, (_, t) => {
    const weight = 0.5 * 0.6 ** t;
    return [(1 - weight) / 3, (1 - weight) / 3, (1 - weight) / 3, weight / 2, weight / 3, weight / 6];
  });
  const prediction = {cloud: targets, sequences: [targets]};
  const metrics = {r_squared: 0.9, mse: 0.01, outside_simplex_fraction: 0.1, component_posterior: {r_squared: 0.8}};
  const run = {
    run_id: "fixture", agent_steps: 100, components: [{x: 0.2, alpha: 0.95}, {x: 0.466, alpha: 0.95}],
    sequences: [{id: 2, component: 0, targets, tokens: targets.map(() => 0), actions: targets.map(() => 0)}],
    cloud: {targets, rows: targets.map((_, i) => i)},
    predictions: {init: {post_final_norm: prediction}, final: {post_final_norm: prediction}},
    metrics: {init: {post_final_norm: metrics}, final: {post_final_norm: metrics}},
  };
  const plots = new Map();
  const context = vm.createContext({
    window: {SIMPLEX_DATA: {schema: 1, runs: [run]}},
    document: {getElementById: get, createElement: element},
    Plotly: {
      async react(id, traces, layout) { plots.set(id, {traces, layout}); },
      async relayout() {},
    },
    clearInterval() {},
  });
  vm.runInContext(source, context);
  await setImmediate();
  assert.equal(get("error").textContent, "");
  return {context, get, plots, targets};
}

test("hue distinguishes vertices and blends local states; weight controls depth", async () => {
  const {context} = await viewer();
  const color = point => context.beliefColor(point);
  assert.equal(color([1, 0, 0]), "rgb(229,83,94)");
  assert.equal(color([0, 1, 0]), "rgb(38,158,115)");
  assert.equal(color([0, 0, 1]), "rgb(60,111,214)");
  assert.equal(color([0.5, 0.5, 0]), "rgb(134,121,105)");
  assert.equal(color([0, 0, 0]), "rgb(40,40,40)");
  assert.notEqual(color([1, 0, 0]), color([0.25, 0, 0]));
  assert.notEqual(color([0, 0, 0]), color([0.25, 0, 0]));
});

test("bounded colors preserve signed, out-of-simplex geometry", async () => {
  const {context} = await viewer();
  const point = [1.4, -0.2, -0.1];
  const before = [...point];
  const trace = context.pointTrace([point], "raw", "#7257ba");
  assert.deepEqual(point, before);
  assert.equal(trace.x[0], 1.4);
  assert.equal(trace.y[0], -0.2);
  assert.equal(trace.marker.color[0], "rgb(229,83,94)");
  assert.equal(context.beliefColor([-1, 0, 0]), "rgb(40,40,40)");
  const plane = context.plane(-0.5);
  assert.deepEqual([...plane.x], [-0.5, 0, 0]);
  assert.equal(plane.vertexcolor.length, 3);
});

test("tiny masses remain nonzero in geometry while labels use fixed decimals", async () => {
  const {get, plots, targets} = await viewer();
  let previous;
  for (const t of [25, 30, 100, 127]) {
    get("time").value = String(t);
    get("time").handlers.input();
    await setImmediate();
    assert.equal(get("error").textContent, "");
    const {traces, layout} = plots.get("target-b");
    const current = traces.at(-1);
    const expected = targets[t].slice(3);
    assert.deepEqual([current.x[0], current.y[0], current.z[0]], expected);
    assert.ok(current.x[0] > 0);
    assert.notEqual(current.x[0], previous);
    previous = current.x[0];
    assert.equal(traces[3].x[0], expected.reduce((a, b) => a + b, 0));
    assert.equal(get("target-b-caption").textContent, "Posterior wB = 0.0000");
    assert.match(current.hovertemplate, /customdata\[0\]:\.4f/);
    assert.ok(layout.scene.xaxis.range[1] >= 1);
  }
});

test("checkpoint markers and plane colors agree across modes", async () => {
  const {get, plots, context} = await viewer();
  get("checkpoint").value = "both";
  for (const mode of ["sequence", "cloud"]) {
    get("mode").value = mode;
    await vm.runInContext("requestRender()", context);
    const traces = plots.get("probe-a").traces;
    const init = traces.find(t => t.name === "Initialization");
    const final = traces.find(t => t.name === "Final");
    assert.equal(init.marker.symbol, "diamond");
    assert.equal(final.marker.symbol, "circle");
    assert.deepEqual([...init.marker.color], [...final.marker.color]);
    if (mode === "sequence") {
      const targetPlane = plots.get("target-a").traces[3];
      assert.equal(new Set(targetPlane.vertexcolor).size, 3);
      assert.deepEqual([...targetPlane.vertexcolor], [...traces[3].vertexcolor]);
    }
  }
});
