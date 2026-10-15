"use strict";
const $ = id => document.getElementById(id);
let token = "", lease = "", state = null, connected = false, session = 0;
let held = null, heldEpoch = 0, manualBusy = false, frameURL = null;
let previewOK = false, lastStatus = 0, commandBusy = false;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
function message(text) { $("message").textContent = text; }
async function api(path, {method = "GET", body, control = lease, keepalive = false} = {}) {
  const response = await fetch(`/api/${path}`, {
    method, cache: "no-store", keepalive, signal: AbortSignal.timeout(1500),
    headers: {Authorization: `Bearer ${token}`, "X-Control-Lease": control,
      ...(body ? {"Content-Type": "application/json"} : {})},
    ...(body ? {body: JSON.stringify(body)} : {})
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : `Request failed (${response.status})`);
  }
  return response;
}
function render() {
  document.body.classList.toggle("paired", connected);
  const fresh = connected && Date.now() - lastStatus < 2000;
  const own = fresh && Boolean(lease);
  $("connection").textContent = fresh ? "Connected" : connected ? "Connection lost" : "Not connected";
  $("owner").textContent = own ? "You have control" : state?.control_held ? "Another controller" : "View only";
  $("take").disabled = !fresh || Boolean(lease) || Boolean(state?.control_held);
  $("start").disabled = !own || !state?.ready || !state?.detector_loaded || state?.tracking || commandBusy;
  $("left").disabled = $("right").disabled = !own || !state?.ready || state?.tracking;
  $("release").disabled = !own;
  $("stop").disabled = !connected;
  $("mode").textContent = state?.tracking ? "Tracking" : "Preview";
  $("camera").textContent = fresh ? state?.phase || "Starting" : "Offline";
  $("ball").textContent = fresh && state?.tracking ? state.target_detected ? "Detected" : "Searching" : "—";
  $("latency").textContent = fresh && state?.tracking && state.inference_ms != null ? `${Math.round(state.inference_ms)} ms` : "—";
  $("hardware").textContent = state ? state.dry_run ? "DRY RUN · No motor connected" : "Serial controller connected" : "No connection";
  $("requirements").textContent = state?.detector_loaded ? `Ready · ${state.policy === "actor" ? "Trained actor" : "Proportional"} camera control` : "Restart the server with --model to enable tracking. Preview and manual controls remain available.";
  $("stale").hidden = previewOK && fresh;
  if (state) {
    $("runtime").textContent = state.runtime;
    $("logs").textContent = state.logs.join("\n");
    if (state.error) message(`Service error: ${state.error}`);
  }
}
function cancelHold() {
  held = null; heldEpoch++;
  $("left").classList.remove("held"); $("right").classList.remove("held");
}
function loseControl(text) { lease = ""; cancelHold(); message(text); render(); }
async function stop() {
  cancelHold(); lease = ""; render();
  try {
    await api("control/command", {method: "POST", body: {action: "stop"}, control: ""});
    if (state) { state.tracking = false; state.control_held = false; state.target_detected = false; }
    render();
    message("Motion stopped. Take control again to resume.");
  } catch (error) { message(`Stop could not be confirmed: ${error.message}. Connection timeout will stop motion.`); }
}
async function release(keepalive = false) {
  const previous = lease;
  loseControl("Control released.");
  if (previous) {
    try { await api("control/release", {method: "POST", control: previous, keepalive}); }
    catch (_) { message("Release not confirmed; control will expire automatically."); }
  }
}
$("pair").addEventListener("submit", async event => {
  event.preventDefault(); token = $("key").value.trim();
  try {
    state = await (await api("status")).json();
    connected = true; session++; lastStatus = Date.now();
    $("pair").hidden = true; $("disconnect").hidden = false; $("key").value = "";
    message("Connected. Take control when you are ready."); render();
    statusLoop(session); previewLoop(session); heartbeatLoop(session);
  } catch (error) { token = ""; message(error.message); }
});
$("take").onclick = async () => {
  $("take").disabled = true;
  try {
    const result = await (await api("control/acquire", {method: "POST"})).json();
    lease = result.lease;
    if (document.hidden) await release();
    else message("You have control. Position the camera or start tracking.");
  } catch (error) { message(error.message); }
  render();
};
$("start").onclick = async () => {
  if (commandBusy || !lease) return;
  commandBusy = true; cancelHold(); render();
  try {
    await api("control/command", {method: "POST", body: {action: "start"}});
    if (state) state.tracking = true;
    message("Automatic tracking requested.");
  } catch (error) { message(error.message); }
  finally { commandBusy = false; render(); }
};
$("stop").onclick = stop;
$("release").onclick = () => release();
$("disconnect").onclick = async () => {
  await release(); connected = false; session++; token = ""; state = null; previewOK = false;
  $("pair").hidden = false; $("disconnect").hidden = true; $("preview").hidden = true;
  $("placeholder").hidden = false;
  if (frameURL) { URL.revokeObjectURL(frameURL); frameURL = null; }
  message("Disconnected."); render();
};
async function manualLoop(epoch, action, owner) {
  // Only one request can be in flight. A released gesture cannot queue more pulses.
  manualBusy = true;
  while (held === action && epoch === heldEpoch && lease === owner && !document.hidden) {
    try { await api("control/command", {method: "POST", body: {action}, control: owner}); }
    catch (error) { loseControl(error.message); }
    if (epoch !== heldEpoch || !held) break;
    await delay(80);
  }
  // Ordered after any last in-flight pulse, so release cannot be overtaken by it.
  try { await api("control/command", {method: "POST", body: {action: "manual-stop"}, control: owner}); }
  catch (_) { /* The bounded 200 ms pulse expires without a follow-up request. */ }
  finally { manualBusy = false; }
}
for (const action of ["left", "right"]) {
  const button = $(action);
  function begin() {
    if (button.disabled || held || manualBusy || !lease) return;
    held = action; const epoch = ++heldEpoch; button.classList.add("held");
    manualLoop(epoch, action, lease);
  }
  button.addEventListener("pointerdown", event => {
    event.preventDefault(); button.setPointerCapture(event.pointerId); begin();
  });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture", "blur"]) button.addEventListener(name, cancelHold);
  button.addEventListener("keydown", event => {
    if ([" ", "Enter"].includes(event.key)) { event.preventDefault(); if (!event.repeat) begin(); }
  });
  button.addEventListener("keyup", event => {
    if ([" ", "Enter"].includes(event.key)) { event.preventDefault(); cancelHold(); }
  });
  button.addEventListener("contextmenu", event => event.preventDefault());
}
async function statusLoop(id) {
  while (connected && session === id) {
    if (!document.hidden) {
      try {
        const result = await (await api("status")).json();
        if (session !== id) return;
        state = result; lastStatus = Date.now();
        if (lease && !state.control_held) loseControl("Control ended. Take control to resume.");
      } catch (error) { loseControl(`Connection interrupted: ${error.message}`); }
      render();
    }
    await delay(500);
  }
}
async function heartbeatLoop(id) {
  while (connected && session === id) {
    if (lease && !document.hidden) {
      const owner = lease;
      try { await api("control/heartbeat", {method: "POST", control: owner}); }
      catch (error) { if (lease === owner) loseControl(`Control connection lost: ${error.message}`); }
    }
    await delay(400);
  }
}
async function previewLoop(id) {
  while (connected && session === id) {
    if (!document.hidden) {
      try {
        const blob = await (await api("preview.jpg")).blob();
        if (session !== id) return;
        const url = URL.createObjectURL(blob);
        $("preview").src = url; $("preview").hidden = false; $("placeholder").hidden = true;
        if (frameURL) URL.revokeObjectURL(frameURL);
        frameURL = url; previewOK = true;
      } catch (_) { previewOK = false; }
      render();
    }
    await delay(125);
  }
}
document.addEventListener("visibilitychange", () => { if (document.hidden) release(true); });
window.addEventListener("pagehide", () => release(true));
window.addEventListener("blur", cancelHold);
window.addEventListener("offline", () => loseControl("Network disconnected. Motion stops when commands expire."));
