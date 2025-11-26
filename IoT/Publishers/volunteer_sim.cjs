const mqtt = require("mqtt");

const BROKER_URL = "mqtt://localhost:1883";
const PUBLISH_MS = 2000;
const MOVE_RANGE = 0.0007;    
const CENTER = { lat: 26.2006, lon: 92.9376 };  // Assam – Brahmaputra Basin

const client = mqtt.connect(BROKER_URL);
client.on("connect", () =>
  console.log(`Volunteer simulator connected → ${BROKER_URL}`)
);
client.on("error", (err) => { console.error(err); process.exit(1); });

/** Four volunteers*/
const volunteers = [
  { id: 1, lat: CENTER.lat, lon: CENTER.lon },
  { id: 2, lat: CENTER.lat + 0.0008, lon: CENTER.lon - 0.0004 },
  { id: 3, lat: CENTER.lat - 0.0009, lon: CENTER.lon + 0.0007 },
  { id: 4, lat: CENTER.lat + 0.0015, lon: CENTER.lon + 0.0003 }
];

const randomStep = (v) => v + (Math.random() - 0.5) * MOVE_RANGE;

/** Haversine distance (meters) */
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

function publishPositions() {
  volunteers.forEach((v) => {
    const prev = { lat: v.lat, lon: v.lon, t: v.t || Date.now() };

    v.lat = randomStep(v.lat);
    v.lon = randomStep(v.lon);
    v.t = Date.now();

    const dist = haversine(prev.lat, prev.lon, v.lat, v.lon);
    const dtSec = (v.t - prev.t) / 1000;
    const speed = dist / dtSec; // m/s

    const dx = v.lon - prev.lon;
    const dy = v.lat - prev.lat;

    /** ------------ MQTT Publish #1: Absolute GPS ------------ */
    client.publish(
      `volunteers/${v.id}/gps`,
      JSON.stringify({
        id: v.id,
        lat: +v.lat.toFixed(6),
        lon: +v.lon.toFixed(6),
        speed: +speed.toFixed(2),
        timestamp: v.t
      })
    );

    /** ------------ MQTT Publish #2: Velocity Vector ------------ */
    client.publish(
      `volunteers/${v.id}/vector`,
      JSON.stringify({
        id: v.id,
        dx: +dx.toFixed(6),
        dy: +dy.toFixed(6),
        speed: +speed.toFixed(2),
        magnitude: +Math.sqrt(dx * dx + dy * dy).toFixed(6),
        timestamp: v.t
      })
    );
  });

  console.log("Volunteer positions + vectors broadcast");
}

setInterval(publishPositions, PUBLISH_MS);
console.log("Volunteer simulation started…");

