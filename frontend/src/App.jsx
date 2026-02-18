import { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([
    "INITIALIZING C.O.R.E. SYSTEMS...",
    "ESTABLISHING SECURE CONNECTION...",
    "ACCESS GRANTED: AGENT TINU."
  ]);

  // Fetch real data from your FastAPI backend
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/system/stats');
        const data = await response.json();
        setStats(data);
      } catch (error) {
        setLogs(prev => ["[ERROR] DATABASE CONNECTION FAILED", ...prev].slice(0, 15));
      }
    };

    // Fetch immediately, then update every 5 seconds
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  // Simulate a live terminal feed of network events
  useEffect(() => {
    const events = [
      "Running biometric analysis...",
      "Intercepting RTSP stream 172.16.0.151...",
      "Calibrating cosine distance thresholds...",
      "New frame captured. Extracting vectors...",
      "Matching embeddings against ChromaDB..."
    ];

    const logInterval = setInterval(() => {
      const randomEvent = events[Math.floor(Math.random() * events.length)];
      const timestamp = new Date().toLocaleTimeString();
      setLogs(prev => [`[${timestamp}] ${randomEvent}`, ...prev].slice(0, 15));
    }, 2500);

    return () => clearInterval(logInterval);
  }, []);

  return (
    <>
      <div className="crt-overlay"></div>
      <div className="scanline"></div>

      <div className="dashboard">
        {/* LEFT PANEL: Live Statistics */}
        <div className="panel">
          <h2>System Status</h2>
          
          <div className="data-row">
            <span>NETWORK:</span>
            <span className={stats?.status === 'ONLINE' ? 'highlight' : 'warning'}>
              {stats?.status || 'OFFLINE'}
            </span>
          </div>

          <div className="data-row">
            <span>UNIQUE SUSPECTS:</span>
            <span className="highlight">{stats?.unique_suspects || 0}</span>
          </div>

          <div className="data-row">
            <span>TOTAL CAPTURES:</span>
            <span className="highlight">{stats?.total_faces_captured || 0}</span>
          </div>

          <div className="data-row">
            <span>ACTIVE CAMERAS:</span>
            <span className="highlight">{stats?.active_cameras || 0}</span>
          </div>

          <h3 style={{ marginTop: '40px' }}>Uplink Status</h3>
          <div className="data-row">
            <span>DB LATENCY:</span>
            <span className="highlight">12ms</span>
          </div>
          <div className="data-row">
            <span>AI CORE:</span>
            <span className="highlight">ANTELOPE-V2 ACTIVE</span>
          </div>
        </div>

        {/* CENTER PANEL: The Radar */}
        <div className="panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ textAlign: 'center' }}>Global Tracking Array</h2>
          <div className="radar-container" style={{ flexGrow: 1 }}>
            <div className="radar">
              <div className="vertical"></div>
            </div>
          </div>
          <p style={{ textAlign: 'center', opacity: 0.7, letterSpacing: '4px' }}>
            SCANNING SECTORS...
          </p>
        </div>

        {/* RIGHT PANEL: Live Terminal Log */}
        <div className="panel">
          <h2>Event Feed</h2>
          <ul className="terminal-log">
            {logs.map((log, index) => (
              <li key={index} style={{ color: index === 0 ? 'var(--neon-green)' : 'inherit' }}>
                {log}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </>
  );
}

export default App;