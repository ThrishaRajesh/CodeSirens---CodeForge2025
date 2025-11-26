// server.js
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

// Serve static files from 'public'
app.use(express.static(path.join(__dirname, 'public')));

// Disaster route
app.get('/disasters', (req, res) => {
    const filePath = 'C:/Users/VISMAYA/Desktop/CodeForge/Disaster_Detection/disaster_reports.json'; // absolute path
    
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) return res.status(500).json({ error: 'Could not read disaster reports', details: err.message });
        try {
            const reports = JSON.parse(data);
            const locations = reports
                .filter(r => r.location && r.location.latitude && r.location.longitude)
                .map(r => ({
                    lat: r.location.latitude,
                    lng: r.location.longitude,
                    disaster_type: r.disaster_type,
                    risk_level: r.risk_level,
                    timestamp: r.timestamp
                }));
            res.json(locations);
        } catch (e) {
            res.status(500).json({ error: 'Invalid JSON', details: e.message });
        }
    });
});

// Socket.io for volunteer updates
io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);

    socket.on('volunteerUpdate', (data) => {
        socket.broadcast.emit('volunteerUpdate', data);
    });

    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
    });
});

// Start server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
