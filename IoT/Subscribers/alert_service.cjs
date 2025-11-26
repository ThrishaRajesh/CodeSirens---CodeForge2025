const mqtt = require("mqtt");
const twilio = require("twilio");

const BROKER = process.env.MQTT_BROKER || "mqtt://localhost:1883";
const client = mqtt.connect(BROKER);

// Twilio config
const TWILIO_SID = process.env.TWILIO_SID || "";
const TWILIO_AUTH_TOKEN = process.env.TWILIO_AUTH_TOKEN || "";
const TWILIO_FROM = process.env.TWILIO_FROM || "";
const ENABLE_TWILIO = TWILIO_SID && TWILIO_AUTH_TOKEN && TWILIO_FROM;
const twilioClient = ENABLE_TWILIO ? twilio(TWILIO_SID, TWILIO_AUTH_TOKEN) : null;

const TOPICS = ["risk/grid", "risk/forecast", "sensors/#"];

const MIN_ALERT_INTERVAL_MS = 1000 * 60 * 5; // 5 minutes per zone by default
const lastAlertAt = new Map(); // key => timestamp

// Map zone 
const ALERT_PHONE_MAP = {
  "24.8333,92.7789": [""]
};

function roundCoord(lat, lon, precision = 4) {
  return `${lat.toFixed(precision)},${lon.toFixed(precision)}`;
}
function makeZoneId(centroid) {
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

function publishMqttAlert(zoneId, centroid, riskDetail, severity = "high") {
  const topic = `alerts/${zoneId}`;
  const payload = {
    zoneId,
    ts: now(),
    centroid,
    severity,
    riskDetail,
    recommendedAction: "evacuate", 
  };
  client.publish(topic, JSON.stringify(payload));
  return { topic, payload };
}


function publishRouteRequest(zoneId, centroid) {
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

/** Send SMS + Voice using Twilio */
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

function shouldEmitAlert(zoneKey) {
  const last = lastAlertAt.get(zoneKey) || 0;
  if (now() - last > MIN_ALERT_INTERVAL_MS) {
    lastAlertAt.set(zoneKey, now());
    return true;
  }
  return false;
}

async function handleRiskMessage(parsed) {
  let centroid = null;
  let riskDetail = null;

  if (parsed.centroid && parsed.riskDetail) {
    centroid = parsed.centroid;
    riskDetail = parsed.riskDetail;
  } else if (parsed.cells && Array.isArray(parsed.cells)) {

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

  const text = buildAlertText(centroid, riskDetail);
  const mqttPublish = publishMqttAlert(zoneId, centroid, riskDetail, "high");

  const routeReq = publishRouteRequest(zoneId, centroid);

  const roundedKey = roundCoord(centroid.lat, centroid.lon, 4);
  const phones = ALERT_PHONE_MAP[roundedKey] || [];

  const twilioResult = await sendTwilioAlerts(phones, text);

  console.log("ALERT EMITTED:", {
    zoneId,
    centroid,
    mqttTopic: mqttPublish.topic,
    routeRequest: routeReq.requestId,
    twilio: twilioResult.sent ? "ok" : "skipped"
  });
}

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

    if (topic === "risk/zone" || topic === "risk/grid" || topic === "risk/forecast") {
  
      handleRiskMessage(msg).catch((e) => console.error("handleRiskMessage error:", e));
      return;
    }
    
    if (topic.startsWith("sensors/")) {
      
      if (msg.type === "sos" || (msg.type === "water" && msg.valueNorm >= 0.95)) {
        const centroid = { lat: msg.lat, lon: msg.lon };
        const riskDetail = { total: msg.valueNorm, hazards: [msg.type] };
        handleRiskMessage({ centroid, riskDetail }).catch((e) => console.error("SOS handler error:", e));
      }
      return;
    }

  } catch (err) {
    console.error("Failed to parse message:", err && err.message);
  }
});

process.on("SIGINT", () => {
  console.log("Alert service shutting down...");
  try { client.end(false, () => process.exit(0)); } catch (e) { process.exit(0); }
});

