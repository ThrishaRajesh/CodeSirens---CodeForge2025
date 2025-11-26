const fs = require('fs');

const path = 'C:/Users/VISMAYA/Desktop/CodeForge/Disaster_Detection/disaster_reports.json';

fs.readFile(path, 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading file:', err);
    } else {
        console.log('File read successfully:', data.slice(0, 100), '...');
    }
});
