/**
 * alert_service.cjs
 * Robust Alerting microservice:
 *  - subscribes to risk/grid, risk/forecast, sensors/#
 *  - deduplicates + rate-limits alerts per zone
 *  - sends SMS + Voice via Twilio (if credentials present)
 *  - always publishes an MQTT alert message on alerts/{zoneId}
 *  - triggers route requests on routes/request for evacuation guidance
 *
 * Usage:
 *   export TWILIO_SID=...
 *   export TWILIO_AUTH_TOKEN=...
 *   export TWILIO_FROM=+1xxxx
 *   node alert_service.cjs
 */

const mqtt = require("mqtt");
const twilio = require("twilio");

const BROKER = process.env.MQTT_BROKER || "mqtt://localhost:1883";
const client = mqtt.connect(BROKER);

// Twilio config (optional). If missing, service will only publish MQTT alerts.
const TWILIO_SID = process.env.TWILIO_SID || "";
const TWILIO_AUTH_TOKEN = process.env.TWILIO_AUTH_TOKEN || "";
const TWILIO_FROM = process.env.TWILIO_FROM || "";
const ENABLE_TWILIO = TWILIO_SID && TWILIO_AUTH_TOKEN && TWILIO_FROM;
const twilioClient = ENABLE_TWILIO ? twilio(TWILIO_SID, TWILIO_AUTH_TOKEN) : null;

// Subscription topics
const TOPICS = ["risk/grid", "risk/forecast", "sensors/#"];

// Rate limiting / dedupe
const MIN_ALERT_INTERVAL_MS = 1000 * 60 * 5; // 5 minutes per zone by default
const lastAlertAt = new Map(); // key => timestamp

// Map zone (rounded centroid) -> phone list (override for demo)
const ALERT_PHONE_MAP = {
  "24.8333,92.7789": [""]
};

// Helpers
function roundCoord(lat, lon, precision = 4) {
  return `${lat.toFixed(precision)},${lon.toFixed(precision)}`;
}
function makeZoneId(centroid) {
  // zone id is rounded centroid
  return roundCoord(centroid.lat, centroid.lon, 4).replace(/\./g, "_").replace(/,/g, "_");
}
function now() {
  return Date.now();
}

/** Haversine (meters) */
function haversine(aLat, aLon, bLat, bLon) {
  const R = 6371e3;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(bLat - aLat);
  const dLon = toRad(bLon - aLon);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(aLat)) *
      Math.cos(toRad(bLat)) *
      Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/** Build human-friendly alert text */
function buildAlertText(centroid, riskDetail) {
  const score = (riskDetail && typeof riskDetail.total === "number")
    ? riskDetail.total.toFixed(2)
    : (riskDetail && riskDetail.score ? riskDetail.score.toFixed(2) : "N/A");

  const hazardList = (riskDetail && riskDetail.hazards)
    ? riskDetail.hazards.join(", ")
    : (riskDetail && riskDetail.type ? riskDetail.type : "multiple hazards");

  const text = `ALERT — Risk ${score} around (${centroid.lat.toFixed(4)},${centroid.lon.toFixed(4)}). Hazards: ${hazardList}. Move to nearest safe zone. Follow official instructions.`;
  return text;
}

/** Publish machine-readable alert on MQTT so other services can react */
function publishMqttAlert(zoneId, centroid, riskDetail, severity = "high") {
  const topic = `alerts/${zoneId}`;
  const payload = {
    zoneId,
    ts: now(),
    centroid,
    severity,
    riskDetail,
    recommendedAction: "evacuate", // could be computed more finely
  };
  client.publish(topic, JSON.stringify(payload));
  return { topic, payload };
}

/** Publish a routes request so route engine can compute evacuations */
function publishRouteRequest(zoneId, centroid) {
  // routes/request payload contract:
  // { requestId, zoneId, ts, origin:{lat,lon}, reason }
  const req = {
    requestId: `req-${zoneId}-${Date.now()}`,
    zoneId,
    ts: now(),
    origin: centroid,
    reason: "evacuation_alert"
  };
  client.publish("routes/request", JSON.stringify(req));
  return req;
}

/** Send SMS + Voice using Twilio (if enabled) */
async function sendTwilioAlerts(phones = [], text = "") {
  if (!ENABLE_TWILIO) {
    console.log("Twilio not configured — skipping SMS/Call. MQTT alerts only.");
    return { sent: false, reason: "twilio-unconfigured" };
  }

  const results = [];
  for (const to of phones) {
    try {
      // SMS
      const msg = await twilioClient.messages.create({
        to,
        from: TWILIO_FROM,
        body: text,
      });

      // Voice call (simple TTS)
      const call = await twilioClient.calls.create({
        to,
        from: TWILIO_FROM,
        twiml: `<Response><Say voice="alice">${text}</Say></Response>`,
      });

      results.push({ to, smsSid: msg.sid, callSid: call.sid, ok: true });
    } catch (err) {
      console.error("Twilio send error for", to, err && err.message);
      results.push({ to, ok: false, error: err && err.message });
    }
  }
  return { sent: true, results };
}

/** Decide whether to emit an alert for a zone now */
function shouldEmitAlert(zoneKey) {
  const last = lastAlertAt.get(zoneKey) || 0;
  if (now() - last > MIN_ALERT_INTERVAL_MS) {
    lastAlertAt.set(zoneKey, now());
    return true;
  }
  return false;
}

/** Process incoming risk/zone-like messages */
async function handleRiskMessage(parsed) {
  // Accept either:
  //  - { centroid: {lat,lon}, riskDetail: { total, hazards: [...] } }
  //  - OR grid summaries with aggregated centroids (we adapt)
  let centroid = null;
  let riskDetail = null;

  if (parsed.centroid && parsed.riskDetail) {
    centroid = parsed.centroid;
    riskDetail = parsed.riskDetail;
  } else if (parsed.cells && Array.isArray(parsed.cells)) {
    // pick highest risk cell as centroid
    const top = parsed.cells.reduce((a, b) => (b.risk > (a.risk||0) ? b : a), {});
    centroid = { lat: top.lat, lon: top.lon };
    riskDetail = { total: top.risk, hazards: parsed.hazards || [] };
  } else {
    console.warn("Unknown risk message shape, ignoring:", Object.keys(parsed));
    return;
  }

  const zoneId = makeZoneId(centroid);

  if (!shouldEmitAlert(zoneId)) {
    console.log(`Alert suppressed for ${zoneId} (rate-limited)`);
    return;
  }

  // Build text and publish machine alert
  const text = buildAlertText(centroid, riskDetail);
  const mqttPublish = publishMqttAlert(zoneId, centroid, riskDetail, "high");

  // Trigger route engine to compute evacuation guidance
  const routeReq = publishRouteRequest(zoneId, centroid);

  // Determine phone list for this zone
  const roundedKey = roundCoord(centroid.lat, centroid.lon, 4);
  const phones = ALERT_PHONE_MAP[roundedKey] || [];

  // Send Twilio notifications (if configured); otherwise log for demo
  const twilioResult = await sendTwilioAlerts(phones, text);

  console.log("ALERT EMITTED:", {
    zoneId,
    centroid,
    mqttTopic: mqttPublish.topic,
    routeRequest: routeReq.requestId,
    twilio: twilioResult.sent ? "ok" : "skipped"
  });
}

/** Global message handler */
client.on("connect", () => {
  console.log("Alert service connected to MQTT broker:", BROKER);
  client.subscribe(TOPICS, { qos: 0 }, (err) => {
    if (err) console.error("Subscribe error:", err);
    else console.log("Subscribed to topics:", TOPICS.join(", "));
  });
});

client.on("message", (topic, msgBuf) => {
  try {
    const msg = JSON.parse(msgBuf.toString());

    // Route messages of interest to handler
    if (topic === "risk/zone" || topic === "risk/grid" || topic === "risk/forecast") {
      // risk aggregator / predictor will publish these forms
      handleRiskMessage(msg).catch((e) => console.error("handleRiskMessage error:", e));
      return;
    }

    // sensors/# can be used to raise micro-alerts (e.g., immediate SOS)
    if (topic.startsWith("sensors/")) {
      // Example: immediate SOS pattern detection
      if (msg.type === "sos" || (msg.type === "water" && msg.valueNorm >= 0.95)) {
        // create an ad-hoc centroid from sensor point
        const centroid = { lat: msg.lat, lon: msg.lon };
        const riskDetail = { total: msg.valueNorm, hazards: [msg.type] };
        handleRiskMessage({ centroid, riskDetail }).catch((e) => console.error("SOS handler error:", e));
      }
      return;
    }

    // ignore other messages
  } catch (err) {
    console.error("Failed to parse message:", err && err.message);
  }
});

/** Graceful shutdown */
process.on("SIGINT", () => {
  console.log("Alert service shutting down...");
  try { client.end(false, () => process.exit(0)); } catch (e) { process.exit(0); }
});
