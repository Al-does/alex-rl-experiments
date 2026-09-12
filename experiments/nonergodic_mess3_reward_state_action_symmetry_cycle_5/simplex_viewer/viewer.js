/* global Plotly */
"use strict";

const colors = {target: "#168379", init: "#bd7336", final: "#7257ba"};
const vertexColors = [[229, 83, 94], [38, 158, 115], [60, 111, 214]];
const ids = ["target-a", "target-b", "probe-a", "probe-b"];
const el = id => document.getElementById(id);
const initialCamera = {eye: {x: 1.55, y: 1.55, z: 1.25}};
let camera = initialCamera;
let syncingCamera = false;
let timer = null;
let rendering = false;
let pendingRender = false;

function block(row, component) {
  return row.slice(component * 3, component * 3 + 3);
}

function mass(row) {
  return row.reduce((a, b) => a + b, 0);
}

function beliefColor(point) {
  const positive = point.map(value => Math.max(0, value));
  const total = mass(positive);
  const depth = Math.sqrt(Math.max(0, Math.min(1, mass(point))));
  const rgb = [0, 1, 2].map(channel => {
    const hue = total > 0
      ? positive.reduce((sum, value, state) => sum + value / total * vertexColors[state][channel], 0)
      : 40;
    return Math.round(40 + depth * (hue - 40));
  });
  return `rgb(${rgb.join(",")})`;
}

function axes(points) {
  return {x: points.map(p => p[0]), y: points.map(p => p[1]), z: points.map(p => p[2])};
}

function plane(weight, color = null, opacity = 0.24) {
  return {
    type: "mesh3d", x: [weight, 0, 0], y: [0, weight, 0], z: [0, 0, weight],
    i: [0], j: [1], k: [2],
    ...(color ? {color} : {vertexcolor: [
      beliefColor([weight, 0, 0]), beliefColor([0, weight, 0]), beliefColor([0, 0, weight]),
    ]}),
    lighting: {ambient: 1, diffuse: 0, specular: 0, fresnel: 0},
    opacity, hoverinfo: "skip", showscale: false,
  };
}

function outline(weight, color, width = 2) {
  return {
    type: "scatter3d", mode: "lines",
    x: [weight, 0, 0, weight], y: [0, weight, 0, 0], z: [0, 0, weight, 0],
    line: {color, width}, hoverinfo: "skip", showlegend: false,
  };
}

function envelope() {
  return [
    plane(1, "#879b8e", 0.035), outline(1, "#b6c5ba", 1),
    {
      type: "scatter3d", mode: "lines", x: [1, 0, null, 0, 0, null, 0, 0],
      y: [0, 0, null, 1, 0, null, 0, 0], z: [0, 0, null, 0, 0, null, 1, 0],
      line: {color: "#ccd7ce", width: 1}, hoverinfo: "skip", showlegend: false,
    },
  ];
}

function pointTrace(points, name, color, {sequence = false, current = false, labels = [], symbol = "circle"} = {}) {
  return {
    type: "scatter3d", mode: sequence ? "lines+markers" : "markers", ...axes(points),
    name, customdata: points.map((p, i) => [mass(p), labels[i] ?? i]),
    marker: {
      color: points.map(beliefColor), symbol,
      size: current ? 7 : sequence ? 2.5 : 2, opacity: current ? 1 : 0.75,
      line: {color, width: current ? 2 : 0},
    },
    line: {color, width: 2}, showlegend: false,
    hovertemplate: `${name}<br>row / t = %{customdata[1]}<br>S0=%{x:.4f}<br>S1=%{y:.4f}<br>S2=%{z:.4f}<br>mass=%{customdata[0]:.4f}<extra></extra>`,
  };
}

function rangeFor(run, layer) {
  let low = 0, high = 1;
  for (const checkpoint of ["init", "final"]) {
    const predictions = run.predictions[checkpoint][layer];
    for (const row of [...predictions.cloud, ...predictions.sequences.flat()]) {
      for (const value of row) {
        low = Math.min(low, value); high = Math.max(high, value);
      }
      for (const component of [0, 1]) {
        const weight = mass(block(row, component));
        low = Math.min(low, weight); high = Math.max(high, weight);
      }
    }
  }
  return [low - 0.06, high + 0.06];
}

function layout(range) {
  const axis = title => ({
    title: {text: title, font: {size: 11}}, range, autorange: false,
    tickfont: {size: 9, color: "#718278"}, nticks: 5, showbackground: false,
    gridcolor: "#e7ede8", zerolinecolor: "#c6d2c9",
  });
  return {
    margin: {l: 0, r: 0, t: 0, b: 0}, paper_bgcolor: "white",
    showlegend: false, uirevision: "linked-camera",
    scene: {
      xaxis: axis("State 0 mass"), yaxis: axis("State 1 mass"), zaxis: axis("State 2 mass"),
      camera, aspectmode: "cube", dragmode: "orbit",
    },
  };
}

const number = value => value == null ? "undefined" : value.toFixed(3);
const percent = value => `${(100 * value).toFixed(1)}%`;

function fillSequences() {
  const run = window.SIMPLEX_DATA.runs[Number(el("run").value)];
  el("sequence").replaceChildren(...run.sequences.map((sequence, i) => {
    const option = document.createElement("option");
    option.value = i;
    option.textContent = `Episode ${sequence.id} · true component ${sequence.component === 0 ? "A" : "B"}`;
    return option;
  }));
  el("time").value = 0;
  fillTokens();
}

function fillTokens() {
  const run = window.SIMPLEX_DATA.runs[Number(el("run").value)];
  const sequence = run.sequences[Number(el("sequence").value)];
  el("tokens").replaceChildren(...sequence.tokens.map((token, t) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = t === 0 ? "B" : token;
    button.title = `t=${t}: ${t === 0 ? "BOS" : `token ${token}`}`;
    button.setAttribute("aria-label", button.title);
    button.onclick = () => { el("time").value = t; stop(); requestRender(); };
    return button;
  }));
}

function stop() {
  clearInterval(timer); timer = null;
  el("play").textContent = "Play";
  el("play").setAttribute("aria-label", "Play sequence");
}

async function render() {
  const run = window.SIMPLEX_DATA.runs[Number(el("run").value)];
  const layer = el("layer").value;
  const sequenceMode = el("mode").value === "sequence";
  const sequenceIndex = Number(el("sequence").value);
  const sequence = run.sequences[sequenceIndex];
  const time = Number(el("time").value);
  const choice = el("checkpoint").value;
  const checkpoints = choice === "both" ? ["init", "final"] : [choice];
  const range = rangeFor(run, layer);
  el("sequence-controls").hidden = !sequenceMode;
  el("run-meta").textContent = `${run.run_id} · ${run.agent_steps.toLocaleString()} training steps · ` +
    run.components.map((c, i) => `${i ? "B" : "A"}: x=${c.x.toFixed(3)}, α=${c.alpha}`).join(" / ");
  el("position").textContent = `${time} / 127`;
  el("time").setAttribute("aria-valuetext", `Position ${time} of 127`);
  if (sequenceMode) {
    const action = time === 0 ? "none" : sequence.actions[time - 1];
    el("step-meta").textContent = `t=${time} · visible token: ${time === 0 ? "BOS" : sequence.tokens[time]} · preceding executed action: ${action} · ` +
      (time === 127 ? "terminal observation; no next action" : `final-policy next action: ${sequence.actions[time]}`) +
      " · action 0=noop, 1=positive, 2=negative";
    [...el("tokens").children].forEach((button, t) => {
      button.className = t === time ? "active" : t > time ? "future" : "";
      button.setAttribute("aria-pressed", String(t === time));
    });
  }
  for (const checkpoint of ["init", "final"]) {
    const metrics = run.metrics[checkpoint][layer];
    el(`${checkpoint}-r2`).textContent = number(metrics.r_squared);
    el(`${checkpoint}-error`).textContent = `MSE ${metrics.mse.toExponential(2)} · ${percent(metrics.outside_simplex_fraction)} outside simplex`;
  }
  el("weight-r2").textContent = ["init", "final"].map(c => number(run.metrics[c][layer].component_posterior.r_squared)).join(" → ");
  el("plot-note").textContent = sequenceMode
    ? "Planes use current component mass. Trails include only positions up to t. Raw probe coordinates and overshoots remain visible."
    : `${run.cloud.targets.length.toLocaleString()} held-out points per component. Faint outlines mark the unit simplex and its sweep to the origin. Both checkpoints use the same histories and axes.`;
  const promises = [];
  for (let component = 0; component < 2; component++) {
    const suffix = component === 0 ? "a" : "b";
    const targetPoints = (sequenceMode ? sequence.targets : run.cloud.targets).map(row => block(row, component));
    const targetTraces = envelope();
    const probeTraces = envelope();
    if (sequenceMode) {
      const point = targetPoints[time], weight = mass(point);
      targetTraces.push(plane(weight), outline(weight, colors.target));
      targetTraces.push(pointTrace(targetPoints.slice(0, time + 1), "Bayesian", colors.target, {sequence: true}));
      targetTraces.push(pointTrace([point], "Bayesian · current", colors.target, {current: true, labels: [time]}));
      el(`target-${suffix}-caption`).textContent = `Posterior w${suffix.toUpperCase()} = ${weight.toFixed(4)}`;
    } else {
      targetTraces.push(pointTrace(targetPoints, "Bayesian", colors.target, {labels: run.cloud.rows}));
      el(`target-${suffix}-caption`).textContent = "Coordinates = posterior weight × local state belief";
    }
    const captions = [];
    for (const checkpoint of checkpoints) {
      const prediction = run.predictions[checkpoint][layer];
      const points = (sequenceMode ? prediction.sequences[sequenceIndex] : prediction.cloud).map(row => block(row, component));
      const name = checkpoint === "init" ? "Initialization" : "Final";
      const symbol = checkpoint === "init" ? "diamond" : "circle";
      if (sequenceMode) {
        const point = points[time], weight = mass(point);
        probeTraces.push(plane(weight, null, 0.16), outline(weight, colors[checkpoint]));
        probeTraces.push(pointTrace(points.slice(0, time + 1), name, colors[checkpoint], {sequence: true, symbol}));
        probeTraces.push(pointTrace([point], `${name} · current`, colors[checkpoint], {current: true, labels: [time], symbol}));
        captions.push(`${name} mass = ${weight.toFixed(4)}`);
      } else {
        probeTraces.push(pointTrace(points, name, colors[checkpoint], {labels: run.cloud.rows, symbol}));
      }
    }
    el(`probe-${suffix}-caption`).textContent = captions.join(" · ") || "Raw affine predictions · no projection onto the simplex";
    for (const [id, traces] of [[`target-${suffix}`, targetTraces], [`probe-${suffix}`, probeTraces]]) {
      promises.push(Plotly.react(id, traces, layout(range), {
        responsive: true, displaylogo: false, scrollZoom: true,
        modeBarButtonsToRemove: ["toImage"],
      }));
    }
  }
  await Promise.all(promises);
}

async function requestRender() {
  if (rendering) { pendingRender = true; return; }
  rendering = true;
  try {
    do { pendingRender = false; await render(); } while (pendingRender);
  } catch (error) {
    stop();
    el("error").hidden = false;
    el("error").textContent = `Could not render: ${error.message}`;
  } finally {
    rendering = false;
  }
}

async function start() {
  if (!window.SIMPLEX_DATA || window.SIMPLEX_DATA.schema !== 1) throw new Error("missing or unsupported viewer data");
  fillSequences();
  await requestRender();
  for (const id of ids) {
    el(id).on("plotly_relayout", async event => {
      if (!event["scene.camera"] || syncingCamera) return;
      syncingCamera = true;
      camera = event["scene.camera"];
      try {
        await Promise.all(ids.filter(other => other !== id).map(other => Plotly.relayout(other, {"scene.camera": camera})));
      } finally { syncingCamera = false; }
    });
  }
  for (const id of ["run", "mode", "layer", "checkpoint", "sequence"]) {
    el(id).addEventListener("change", () => {
      stop();
      if (id === "run") fillSequences();
      if (id === "sequence") { el("time").value = 0; fillTokens(); }
      requestRender();
    });
  }
  el("time").addEventListener("input", () => { stop(); requestRender(); });
  el("play").onclick = () => {
    if (timer) { stop(); return; }
    if (Number(el("time").value) === 127) el("time").value = 0;
    el("play").textContent = "Pause";
    el("play").setAttribute("aria-label", "Pause sequence");
    timer = setInterval(() => {
      if (rendering) return;
      const next = Number(el("time").value) + 1;
      el("time").value = Math.min(127, next);
      requestRender();
      if (next >= 127) stop();
    }, 350);
  };
  el("reset-camera").onclick = async () => {
    camera = initialCamera;
    syncingCamera = true;
    try { await Promise.all(ids.map(id => Plotly.relayout(id, {"scene.camera": camera}))); }
    finally { syncingCamera = false; }
  };
}

start().catch(error => {
  el("error").hidden = false;
  el("error").textContent = `Could not start: ${error.message}. Keep index.html, data.js and plotly.min.js together.`;
});
