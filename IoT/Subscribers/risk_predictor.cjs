const mqtt = require("mqtt");
const BROKER = process.env.MQTT_BROKER || "mqtt://localhost:1883";
const client = mqtt.connect(BROKER);

const CENTER = {
  lat: parseFloat(process.env.CENTER_LAT || "26.2006"),
  lon: parseFloat(process.env.CENTER_LON || "92.9376")
};
const ROWS = parseInt(process.env.GRID_ROWS || "40", 10);
const COLS = parseInt(process.env.GRID_COLS || "40", 10);
const LAT_SPAN = parseFloat(process.env.LAT_SPAN || "0.12"); 
const LON_SPAN = parseFloat(process.env.LON_SPAN || "0.12"); 
const PUBLISH_INTERVAL_MS = parseInt(process.env.PUBLISH_MS || "1000", 10);
const FORECAST_STEPS = parseInt(process.env.FORECAST_STEPS || "3", 10);
const DIFF_COEFF = parseFloat(process.env.DIFF_COEFF || "0.25");
const DECAY = parseFloat(process.env.DECAY || "0.92");
const SMOOTH_ALPHA = parseFloat(process.env.SMOOTH_ALPHA || "0.6"); 
const RISK_THRESHOLD_ZONE = parseFloat(process.env.RISK_ZONE_THRESHOLD || "0.6");
const MIN_ZONE_CELLS = parseInt(process.env.MIN_ZONE_CELLS || "4", 10);
const ROUTE_RISK_WEIGHT = parseFloat(process.env.ROUTE_RISK_WEIGHT || "9.0");

const minLat = CENTER.lat - LAT_SPAN / 2;
const maxLat = CENTER.lat + LAT_SPAN / 2;
const minLon = CENTER.lon - LON_SPAN / 2;
const maxLon = CENTER.lon + LON_SPAN / 2;
const latStep = (maxLat - minLat) / ROWS;
const lonStep = (maxLon - minLon) / COLS;

let riskGrid = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
let sensorLastSeen = {}; 
let volunteerHist = {};
let volunteerCellVectors = Array.from({ length: ROWS }, () => Array.from({ length: COLS }, () => ({ vx: 0, vy: 0, count: 0 })));

let highRiskBuffer = []; 
const HIGH_RISK_BUFFER_MAX = 500;

/* MQTT topics to subscribe */
const SENSOR_TOPIC = "sensors/#";
const VOL_GPS_TOPIC = "volunteers/+/gps";
const VOL_VEC_TOPIC = "volunteers/+/vector";
const ROUTE_REQUEST_TOPIC = "routes/request";

function latLonToCell(lat, lon) {
  let i = Math.floor((lat - minLat) / latStep);
  let j = Math.floor((lon - minLon) / lonStep);
  if (i < 0) i = 0;
  if (i >= ROWS) i = ROWS - 1;
  if (j < 0) j = 0;
  if (j >= COLS) j = COLS - 1;
  return { i, j };
}
function cellCenter(i, j) {
  const lat = minLat + (i + 0.5) * latStep;
  const lon = minLon + (j + 0.5) * lonStep;
  return { lat, lon };
}

function normalizeSensorValue(type, value) {
  if (type === "rain") return Math.min(1, value / 100); 
  if (type === "water") return Math.min(1, value / 3);   
  if (type === "tremor") return Math.min(1, value / 1);  
  if (type === "sos") return 1.0;
  return Math.min(1, value / 100);
}

function updateCellRisk(i, j, obsNorm, alpha = SMOOTH_ALPHA) {
  const prev = riskGrid[i][j] || 0;
  const next = alpha * obsNorm + (1 - alpha) * prev;
  riskGrid[i][j] = Math.min(1, Math.max(0, next));
}

function predictNextGrid(grid, diffCoeff = DIFF_COEFF, decay = DECAY) {
  const rows = grid.length;
  const cols = grid[0].length;
  const next = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let neighSum = 0;
      let neighCount = 0;
      for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
          if (di === 0 && dj === 0) continue;
          const ni = i + di;
          const nj = j + dj;
          if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
            neighSum += grid[ni][nj];
            neighCount++;
          }
        }
      }
      const neighAvg = neighCount ? neighSum / neighCount : 0;
      next[i][j] = Math.min(1, decay * grid[i][j] + diffCoeff * neighAvg);
    }
  }
  return next;
}

function convexHull(points) {
  if (!points || points.length < 3) return points.slice();
  points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of points) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = points.length - 1; i >= 0; i--) {
    const p = points[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function astar(startCell, goalCell, gridRisk, weight = ROUTE_RISK_WEIGHT) {
  const rows = gridRisk.length, cols = gridRisk[0].length;
  function inBounds(i, j) { return i >= 0 && i < rows && j >= 0 && j < cols; }
  function heuristic(a, b) { // Euclidean in grid units
    const di = a.i - b.i, dj = a.j - b.j;
    return Math.sqrt(di * di + dj * dj);
  }
  const start = { i: startCell.i, j: startCell.j };
  const goal = { i: goalCell.i, j: goalCell.j };

  const openKey = (i, j) => `${i},${j}`;
  const neighbors = (node) => {
    const res = [];
    for (let di = -1; di <= 1; di++) {
      for (let dj = -1; dj <= 1; dj++) {
        if (di === 0 && dj === 0) continue;
        const ni = node.i + di, nj = node.j + dj;
        if (!inBounds(ni, nj)) continue;
        // diagonal cost slightly higher:
        const moveCost = (di === 0 || dj === 0) ? 1 : Math.SQRT2;
        res.push({ i: ni, j: nj, moveCost });
      }
    }
    return res;
  };

  const openSet = new Map(); 
  const pq = new MinHeap();
  const startKey = openKey(start.i, start.j);
  const startRisk = gridRisk[start.i][start.j];
  openSet.set(startKey, { f: heuristic(start, goal), g: 0, parent: null });
  pq.push({ key: startKey, f: heuristic(start, goal), i: start.i, j: start.j, g: 0 });

  while (!pq.isEmpty()) {
    const node = pq.pop();
    const key = node.key;
    const cur = openSet.get(key);
    if (!cur) continue;
    const [ci, cj] = key.split(",").map(Number);
    if (ci === goal.i && cj === goal.j) {

      const path = [];
      let curKey = key;
      while (curKey) {
        const [pi, pj] = curKey.split(",").map(Number);
        path.push({ i: pi, j: pj });
        const parent = openSet.get(curKey) && openSet.get(curKey).parent;
        curKey = parent;
      }
      path.reverse();
      return path;
    }

    for (const nb of neighbors({ i: ci, j: cj })) {
      const nKey = openKey(nb.i, nb.j);
      const risk = gridRisk[nb.i][nb.j];
      const traversalCost = nb.moveCost * (1 + weight * risk);
      const tentativeG = cur.g + traversalCost;
      const existing = openSet.get(nKey);
      if (!existing || tentativeG < existing.g) {
        const h = heuristic({ i: nb.i, j: nb.j }, goal);
        const f = tentativeG + h;
        openSet.set(nKey, { f, g: tentativeG, parent: key });
        pq.push({ key: nKey, f, i: nb.i, j: nb.j, g: tentativeG });
      }
    }
 
    openSet.delete(key);
  }


  return null;
}


class MinHeap {
  constructor() { this.arr = []; }
  push(x) { this.arr.push(x); this._siftUp(this.arr.length - 1); }
  pop() {
    if (!this.arr.length) return null;
    const top = this.arr[0];
    const last = this.arr.pop();
    if (this.arr.length) { this.arr[0] = last; this._siftDown(0); }
    return top;
  }
  isEmpty() { return this.arr.length === 0; }
  _siftUp(i) {
    while (i > 0) {
      const p = Math.floor((i - 1) / 2);
      if (this.arr[p].f <= this.arr[i].f) break;
      [this.arr[p], this.arr[i]] = [this.arr[i], this.arr[p]];
      i = p;
    }
  }
  _siftDown(i) {
    const n = this.arr.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.arr[l].f < this.arr[smallest].f) smallest = l;
      if (r < n && this.arr[r].f < this.arr[smallest].f) smallest = r;
      if (smallest === i) break;
      [this.arr[smallest], this.arr[i]] = [this.arr[i], this.arr[smallest]];
      i = smallest;
    }
  }
}


function handleSensorMessage(topic, msg) {
  try {
    const payload = JSON.parse(msg.toString());
    const lat = Number(payload.lat);
    const lon = Number(payload.lon);
    const type = payload.type || (payload.rainfall ? "rain" : (payload.water ? "water" : "unknown"));
    const rawVal = payload.value !== undefined ? Number(payload.value) : (payload.valueNorm !== undefined ? payload.valueNorm : null);
    const norm = payload.valueNorm !== undefined ? Number(payload.valueNorm) : normalizeSensorValue(type, rawVal || 0);

    const { i, j } = latLonToCell(lat, lon);
    updateCellRisk(i, j, norm, SMOOTH_ALPHA);


    sensorLastSeen[payload.nodeId || `${i}_${j}`] = { ts: payload.ts || Date.now(), type, value: rawVal, norm, lat, lon };

    if (norm >= RISK_THRESHOLD_ZONE) {
      const center = cellCenter(i,j);
      highRiskBuffer.push({ i, j, lat: center.lat, lon: center.lon, risk: norm, ts: Date.now(), detail: { type, rawVal } });
      if (highRiskBuffer.length > HIGH_RISK_BUFFER_MAX) highRiskBuffer.shift();
    }
  } catch (e) {
    console.warn("handleSensorMessage parse error:", e && e.message);
  }
}

function handleVolunteerGps(topic, msg) {
  try {
    const payload = JSON.parse(msg.toString());
    const vid = payload.id || payload.volId;
    if (!vid) return;
    const lat = Number(payload.lat), lon = Number(payload.lon), ts = payload.timestamp || payload.ts || Date.now();
    volunteerHist[vid] = volunteerHist[vid] || [];
    volunteerHist[vid].push({ lat, lon, ts });

    if (volunteerHist[vid].length > 5) volunteerHist[vid].shift();

    if (volunteerHist[vid].length >= 2) {
      const a = volunteerHist[vid][volunteerHist[vid].length - 2];
      const b = volunteerHist[vid][volunteerHist[vid].length - 1];
      const dx = b.lon - a.lon;
      const dy = b.lat - a.lat;
      const { i, j } = latLonToCell(b.lat, b.lon);
      const cellVec = volunteerCellVectors[i][j];
      cellVec.vx += dx;
      cellVec.vy += dy;
      cellVec.count += 1;
    }
  } catch (e) {
    console.warn("handleVolunteerGps parse error:", e && e.message);
  }
}

function handleVolunteerVector(topic, msg) {
  try {
    const payload = JSON.parse(msg.toString());
    const latLon = volunteerHist[payload.id] && volunteerHist[payload.id].slice(-1)[0];
    if (!latLon) return; 
    const { lat, lon } = latLon;
    const { i, j } = latLonToCell(lat, lon);
    const cellVec = volunteerCellVectors[i][j];
    cellVec.vx += Number(payload.dx || 0);
    cellVec.vy += Number(payload.dy || 0);
    cellVec.count += 1;
  } catch (e) {
    console.warn("handleVolunteerVector parse error:", e && e.message);
  }
}

function publishRiskGrid() {
  const cells = [];
  for (let i = 0; i < ROWS; i++) {
    for (let j = 0; j < COLS; j++) {
      const c = cellCenter(i, j);
      cells.push({ i, j, lat: c.lat, lon: c.lon, risk: Number(riskGrid[i][j].toFixed(4)) });
    }
  }
  client.publish("risk/grid", JSON.stringify({ gridId: "g1", ts: Date.now(), rows: ROWS, cols: COLS, cells }));
}

function publishVolFlow() {
  const cells = [];
  for (let i = 0; i < ROWS; i++) {
    for (let j = 0; j < COLS; j++) {
      const vec = volunteerCellVectors[i][j];
      if (vec.count > 0) {
        const avgVx = vec.vx / vec.count;
        const avgVy = vec.vy / vec.count;
        const mag = Math.sqrt(avgVx * avgVx + avgVy * avgVy);
        const c = cellCenter(i, j);
        cells.push({ i, j, lat: c.lat, lon: c.lon, vx: Number(avgVx.toFixed(6)), vy: Number(avgVy.toFixed(6)), magnitude: Number(mag.toFixed(6)), count: vec.count });
      }
      volunteerCellVectors[i][j] = { vx: 0, vy: 0, count: 0 };
    }
  }
  client.publish("viz/vol_flow", JSON.stringify({ ts: Date.now(), cells }));
}

function detectAndPublishZones() {
  if (!highRiskBuffer.length) return;
  const mapKey = (it) => `${it.i},${it.j}`;
  const cellMap = new Map();
  for (const e of highRiskBuffer) {
    const k = mapKey(e);
    if (!cellMap.has(k)) cellMap.set(k, { i: e.i, j: e.j, lat: e.lat, lon: e.lon, risk: e.risk, count: 0, details: [] });
    const entry = cellMap.get(k);
    entry.count += 1;
    entry.risk = Math.max(entry.risk, e.risk);
    entry.details.push(e.detail);
  }
  const entries = Array.from(cellMap.values()).filter(e => e.risk >= RISK_THRESHOLD_ZONE);
  if (entries.length < MIN_ZONE_CELLS) return;


  const visited = new Set();
  const clusters = [];
  function cellNeighbors(i, j) {
    const res = [];
    for (let di = -1; di <= 1; di++) {
      for (let dj = -1; dj <= 1; dj++) {
        if (di === 0 && dj === 0) continue;
        const ni = i + di, nj = j + dj;
        if (ni >= 0 && ni < ROWS && nj >= 0 && nj < COLS) res.push([ni, nj]);
      }
    }
    return res;
  }
  const entryMap = new Map(entries.map(e => [ `${e.i},${e.j}`, e ]));
  for (const e of entries) {
    const rootKey = `${e.i},${e.j}`;
    if (visited.has(rootKey)) continue;
  
    const q = [e];
    const comp = [];
    visited.add(rootKey);
    while (q.length) {
      const cur = q.shift();
      comp.push(cur);
      for (const [ni, nj] of cellNeighbors(cur.i, cur.j)) {
        const k = `${ni},${nj}`;
        if (visited.has(k)) continue;
        if (entryMap.has(k)) {
          visited.add(k);
          q.push(entryMap.get(k));
        }
      }
    }
    clusters.push(comp);
  }

  for (const cluster of clusters) {
    if (cluster.length < MIN_ZONE_CELLS) continue;
    const pts = cluster.map(c => [c.lat, c.lon]);
    const hull = convexHull(pts);

    const centroid = hull.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1]], [0, 0]).map(x => x / hull.length);
    const centroidObj = { lat: Number(centroid[0].toFixed(6)), lon: Number(centroid[1].toFixed(6)) };
    const maxRisk = Math.max(...cluster.map(c => c.risk));
    const hazards = [...new Set(cluster.flatMap(c => c.details.flatMap(d => d && d.type ? [d.type] : [])))];
    const payload = {
      hull,
      centroid: centroidObj,
      ts: Date.now(),
      riskDetail: { total: Number(maxRisk.toFixed(4)), hazards }
    };
    client.publish("risk/zone", JSON.stringify(payload));

    highRiskBuffer = highRiskBuffer.filter(h => Date.now() - h.ts < 1000 * 60 * 10); 
  }
}

function handleRouteRequest(topic, msg) {
  try {
    const req = JSON.parse(msg.toString());
    const origin = req.origin || req.originPoint || req.originLatLon;
    const start = latLonToCell(origin.lat, origin.lon);

    const candidates = [];
    for (let i = 0; i < ROWS; i++) {
      for (let j = 0; j < COLS; j++) {
        if (i === 0 || j === 0 || i === ROWS - 1 || j === COLS - 1) {
          candidates.push({ i, j, risk: riskGrid[i][j] });
        }
      }
    }
    candidates.sort((a, b) => a.risk - b.risk); 
    let path = null;
    for (const cand of candidates.slice(0, 10)) { 
      path = astar(start, { i: cand.i, j: cand.j }, riskGrid, ROUTE_RISK_WEIGHT);
      if (path && path.length) break;
    }
    if (!path) {
 
      path = simpleStraightPathToEdge(start);
    }
   
    const waypoints = path.map(cell => {
      const c = cellCenter(cell.i, cell.j);
      return { lat: Number(c.lat.toFixed(6)), lon: Number(c.lon.toFixed(6)), i: cell.i, j: cell.j };
    });
    const outTopic = `routes/${req.requestId || (`auto-${Date.now()}`)}`;
    const outPayload = { requestId: req.requestId || null, ts: Date.now(), origin, waypoints, meta: { reason: req.reason || "evacuation" } };
    client.publish(outTopic, JSON.stringify(outPayload));
  } catch (e) {
    console.warn("handleRouteRequest error:", e && e.message);
  }
}


function simpleStraightPathToEdge(start) {
  const { i, j } = start;
  const path = [{ i, j }];
  let ci = i, cj = j;
  const maxSteps = ROWS + COLS;
  for (let step = 0; step < maxSteps; step++) {
    if (ci === 0 || cj === 0 || ci === ROWS - 1 || cj === COLS - 1) break;
    const ni = ci < ROWS / 2 ? ci - 1 : ci + 1;
    const nj = cj < COLS / 2 ? cj - 1 : cj + 1;
    if (Math.abs(ni - ci) >= Math.abs(nj - cj)) ci = ni; else cj = nj;
    path.push({ i: ci, j: cj });
  }
  return path;
}


function periodicPublish() {
  publishRiskGrid();

  let forecastGrid = JSON.parse(JSON.stringify(riskGrid));
  const forecasts = [];
  for (let s = 0; s < FORECAST_STEPS; s++) {
    forecastGrid = predictNextGrid(forecastGrid, DIFF_COEFF, DECAY);
    const cells = [];
    for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) {
      const c = cellCenter(i, j);
      cells.push({ i, j, lat: c.lat, lon: c.lon, risk: Number(forecastGrid[i][j].toFixed(4)) });
    }
    forecasts.push({ step: s + 1, ts: Date.now(), cells });
  }
  client.publish("risk/forecast", JSON.stringify({ gridId: "g1", ts: Date.now(), forecasts }));

  publishVolFlow();

  detectAndPublishZones();
}

/* MQTT setup */
client.on("connect", () => {
  console.log("Risk predictor connected to broker:", BROKER);
  client.subscribe([SENSOR_TOPIC, VOL_GPS_TOPIC, VOL_VEC_TOPIC, ROUTE_REQUEST_TOPIC], (err) => {
    if (err) console.error("subscribe error:", err);
    else console.log("Subscribed to sensors & volunteers & routes/request");
  });
});

client.on("message", (topic, msg) => {
  if (topic.startsWith("sensors/")) {
    return handleSensorMessage(topic, msg);
  }
  if (topic.includes("/gps")) {
    return handleVolunteerGps(topic, msg);
  }
  if (topic.includes("/vector")) {
    return handleVolunteerVector(topic, msg);
  }
  if (topic === ROUTE_REQUEST_TOPIC) {
    return handleRouteRequest(topic, msg);
  }
});

setInterval(periodicPublish, PUBLISH_INTERVAL_MS);

process.on("SIGINT", () => {
  console.log("Risk predictor shutting down...");
  try { client.end(false, () => process.exit(0)); } catch (e) { process.exit(0); }
});

