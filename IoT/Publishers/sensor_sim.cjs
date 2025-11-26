const mqtt = require("mqtt");

const BROKER_URL = "mqtt://localhost:1883";
const GRID_SIZE = 50;
const CENTER = { lat: 26.2006, lon: 92.9376 };  
const JITTER = 0.05;          
const INTERVAL = 2000;        

const client = mqtt.connect(BROKER_URL);
client.on("connect", () =>
  console.log(`Sensor simulator connected → ${BROKER_URL}`)
);
client.on("error", (e) => {
  console.error("MQTT Connection Error:", e);
  process.exit(1);
});

// Generate simulated sensor nodes
const nodes = Array.from({ length: GRID_SIZE }, (_, i) => ({
  nodeId: `node-${i + 1}`,
  lat: CENTER.lat + (Math.random() - 0.5) * JITTER,
  lon: CENTER.lon + (Math.random() - 0.5) * JITTER,
}));

function normalizeRain(mm) {
  return Math.min(1, mm / 100); 
}
function normalizeWater(m) {
  return Math.min(1, m / 3);   
}

/**
 * Broadcast simulated sensor readings
 */
function pushData() {
  nodes.forEach((n) => {
    const rainfall = +(Math.random() * 100).toFixed(1); 
    const water = +(Math.random() * 3).toFixed(2);     

    const messages = [
      {
        type: "rain",
        value: rainfall,
        valueNorm: normalizeRain(rainfall),
      },
      {
        type: "water",
        value: water,
        valueNorm: normalizeWater(water),
      },
    ];

    messages.forEach((sensor) => {
      const payload = {
        nodeId: n.nodeId,
        ts: Date.now(),
        lat: n.lat,
        lon: n.lon,
        type: sensor.type,
        value: sensor.value,
        valueNorm: sensor.valueNorm,
      };

      client.publish(`sensors/${n.nodeId}`, JSON.stringify(payload));
    });
  });

  console.log("Sensor grid data broadcast");
}

setInterval(pushData, INTERVAL);
console.log("IoT Sensor simulation started…");

