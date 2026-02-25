import { useEffect, useState, useRef } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([
    "INITIALIZING C.O.R.E. SYSTEMS...",
    "ESTABLISHING SECURE CONNECTION...",
    "ACCESS GRANTED: AGENT TINU."
  ]);

  // Search States (Defaulting to IMG_SEARCH now)
  const [activeTab, setActiveTab] = useState('IMG_SEARCH'); 
  const [selectedFile, setSelectedFile] = useState(null);
  const [searchId, setSearchId] = useState("");
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const fileInputRef = useRef(null);

  // Fetch Stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/system/stats`);
        const data = await response.json();
        setStats(data);
      } catch (error) {
        setLogs(prev => ["[ERROR] DATABASE CONNECTION FAILED", ...prev].slice(0, 15));
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  // Simulate Feed Logs
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

  // Handlers
  const handleImageSearch = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    setIsSearching(true);
    setSearchResults(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE_URL}/api/investigate/search_by_image`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setSearchResults(data.sightings || []);
    } catch (err) {
      console.error(err);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleIdSearch = async (e) => {
    e.preventDefault();
    if (!searchId) return;
    setIsSearching(true);
    setSearchResults(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/investigate/person/${searchId}`);
      if (!res.ok) throw new Error("Not found");
      const data = await res.json();
      setSearchResults(data.timeline || []);
    } catch (err) {
      console.error(err);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const triggerFileInput = () => fileInputRef.current.click();

  const renderResults = () => {
    if (isSearching) return <div className="loading-text">[ ANALYZING BIOMETRIC DATA... ]</div>;
    if (searchResults && searchResults.length === 0) return <div className="warning">[ NO MATCHES FOUND IN DATABASE ]</div>;
    if (searchResults && searchResults.length > 0) {
      return (
        <div className="results-grid">
          {searchResults.map((sighting, idx) => (
            <div key={idx} className="result-card">
              <img 
                src={`${API_BASE_URL}${sighting.image_url}`} 
                alt="Target" 
                className="result-image"
                onError={(e) => e.target.src = 'https://via.placeholder.com/150/000000/00ff00?text=NO+IMAGE'}
              />
              <div className="result-info">
                <div className="data-row"><span>CAM:</span> <span className="highlight">{sighting.camera}</span></div>
                <div className="data-row"><span>DATE:</span> <span className="highlight">{sighting.timestamp.split(' ')[0]}</span></div>
                <div className="data-row"><span>TIME:</span> <span className="highlight">{sighting.timestamp.split(' ')[1]}</span></div>
                {sighting.match_score && (
                  <div className="data-row"><span>CONF:</span> <span className="highlight">{(sighting.match_score * 100).toFixed(1)}%</span></div>
                )}
              </div>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <>
      <div className="crt-overlay"></div>
      <div className="scanline"></div>

      <div className="dashboard">
        
        {/* PANEL 1: Live Statistics */}
        <div className="panel">
          <h2>System Status</h2>
          <div className="data-row">
            <span>NETWORK:</span>
            <span className={stats?.status === 'ONLINE' ? 'highlight' : 'warning'}>{stats?.status || 'OFFLINE'}</span>
          </div>
          <div className="data-row"><span>UNIQUE SUSPECTS:</span><span className="highlight">{stats?.unique_suspects || 0}</span></div>
          <div className="data-row"><span>TOTAL CAPTURES:</span><span className="highlight">{stats?.total_faces_captured || 0}</span></div>
          <div className="data-row"><span>ACTIVE CAMERAS:</span><span className="highlight">{stats?.active_cameras || 0}</span></div>
          
          <h3 style={{ marginTop: '40px' }}>Uplink Status</h3>
          <div className="data-row"><span>DB LATENCY:</span><span className="highlight">12ms</span></div>
          <div className="data-row"><span>AI CORE:</span><span className="highlight">ANTELOPE-V2 ACTIVE</span></div>
        </div>

        {/* PANEL 2: The Radar */}
        <div className="panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ textAlign: 'center' }}>Global Tracking Array</h2>
          <div className="radar-container" style={{ flexGrow: 1 }}>
            <div className="radar"><div className="vertical"></div></div>
          </div>
          <p style={{ textAlign: 'center', opacity: 0.7, letterSpacing: '4px' }}>SCANNING SECTORS...</p>
        </div>

        {/* PANEL 3: Dedicated Live Feed */}
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

        {/* PANEL 4: Investigation & Search (The previously empty space) */}
        <div className="panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2>Investigation</h2>
          
          <div className="tabs-container">
            <button className={`tab-btn ${activeTab === 'IMG_SEARCH' ? 'active-tab' : ''}`} onClick={() => {setActiveTab('IMG_SEARCH'); setSearchResults(null);}}>[ IMG_SEARCH ]</button>
            <button className={`tab-btn ${activeTab === 'ID_SEARCH' ? 'active-tab' : ''}`} onClick={() => {setActiveTab('ID_SEARCH'); setSearchResults(null);}}>[ ID_SEARCH ]</button>
          </div>

          <div className="tab-content" style={{ flexGrow: 1, overflowY: 'auto', marginTop: '15px' }}>
            
            {/* Image Search Form */}
            {activeTab === 'IMG_SEARCH' && (
              <div className="search-module">
                <p>UPLOAD SUSPECT BIOMETRIC DATA</p>
                <form onSubmit={handleImageSearch}>
                  <input type="file" accept="image/*" ref={fileInputRef} style={{ display: 'none' }} onChange={(e) => setSelectedFile(e.target.files[0])} />
                  <button type="button" className="sci-fi-input-btn" onClick={triggerFileInput}>
                    {selectedFile ? `> ${selectedFile.name}` : "> SELECT IMAGE FILE"}
                  </button>
                  <button type="submit" className="sci-fi-submit-btn" disabled={!selectedFile || isSearching}>[ INITIATE SCAN ]</button>
                </form>
                <div className="results-container">{renderResults()}</div>
              </div>
            )}

            {/* ID Search Form */}
            {activeTab === 'ID_SEARCH' && (
              <div className="search-module">
                <p>ENTER TARGET IDENTIFICATION</p>
                <form onSubmit={handleIdSearch}>
                  <input type="text" className="sci-fi-text-input" placeholder="person_xxxxxxxx" value={searchId} onChange={(e) => setSearchId(e.target.value)} />
                  <button type="submit" className="sci-fi-submit-btn" disabled={!searchId || isSearching}>[ PULL DOSSIER ]</button>
                </form>
                <div className="results-container">{renderResults()}</div>
              </div>
            )}

          </div>
        </div>

      </div>
    </>
  );
}

export default App;