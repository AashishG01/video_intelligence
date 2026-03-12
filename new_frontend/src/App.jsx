import React, { useState, useRef } from 'react';
import { 
  MonitorPlay, 
  UserSearch, 
  UploadCloud, 
  Search, 
  MapPin, 
  Clock, 
  AlertCircle,
  Loader2,
  X
} from 'lucide-react';

// ==========================================
// CONFIGURATION
// ==========================================
// Point this to your FastAPI backend address
const BACKEND_URL = "http://localhost:8000";

// Mock Live Feed (Since the backend doesn't have a live websocket yet)
const MOCK_LIVE_FEED = [
  { id: 1, cam: "Main Entrance", time: "Just now", img: "https://i.pravatar.cc/150?img=11" },
  { id: 2, cam: "Lobby", time: "1 min ago", img: "https://i.pravatar.cc/150?img=12" },
];

// ==========================================
// COMPONENT: Sighting Card (Now wired to Backend Data)
// ==========================================
const SightingCard = ({ data }) => {
  // Convert backend score (e.g. 0.45) to percentage
  const confidencePercent = (data.match_score * 100).toFixed(1);
  const isHighConf = data.match_score >= 0.45; // Anything over 45% is considered High Confidence for UI
  
  // The backend returns a relative path like "/images/person_id/file.jpg". We need the full URL.
  const fullImageUrl = `${BACKEND_URL}${data.image_url}`;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 flex items-center shadow-sm hover:shadow-md transition-shadow">
      <img src={fullImageUrl} alt="Subject Sighting" className="w-16 h-16 rounded-lg object-cover mr-4 border border-slate-100" />
      
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1">
          <h4 className="font-semibold text-slate-800">Subject #{data.person_id.substring(0, 8)}</h4>
          <span className={`px-2.5 py-1 text-xs font-medium rounded-full ${
            isHighConf ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
          }`}>
            {isHighConf ? `High Match (${confidencePercent}%)` : `Possible (${confidencePercent}%)`}
          </span>
        </div>
        <div className="flex items-center text-sm text-slate-500 space-x-4">
          <div className="flex items-center"><MapPin className="w-3.5 h-3.5 mr-1" /> {data.camera}</div>
          <div className="flex items-center"><Clock className="w-3.5 h-3.5 mr-1" /> {data.timestamp}</div>
        </div>
      </div>

      <button className="ml-6 px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors">
        Full Timeline
      </button>
    </div>
  );
};

// ==========================================
// COMPONENT: Investigator View
// ==========================================
const InvestigatorView = () => {
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  
  // File Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setHasSearched(false);
      setSearchResults([]);
      setErrorMsg(null);
    }
  };

  const clearSelection = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    setHasSearched(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSearch = async () => {
    if (!selectedFile) {
      setErrorMsg("Please upload a suspect photo first.");
      return;
    }

    setIsSearching(true);
    setErrorMsg(null);

    // Package the file for the FastAPI backend
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Sending to backend with your calibrated 0.57 threshold
      const response = await fetch(`${BACKEND_URL}/api/investigate/search_by_image?threshold=0.57`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Database search failed.");
      }

      const data = await response.json();
      setSearchResults(data.sightings || []);
      setHasSearched(true);

    } catch (err) {
      console.error(err);
      setErrorMsg(err.message);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto py-8 px-6">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-slate-800 mb-2">Subject Investigation</h2>
        <p className="text-slate-500">Upload a photo to locate a subject across all active cameras.</p>
      </div>

      {/* Upload & Preview Zone */}
      <input 
        type="file" 
        accept="image/*" 
        className="hidden" 
        ref={fileInputRef} 
        onChange={handleFileSelect} 
      />
      
      <div 
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl bg-white p-10 flex flex-col items-center justify-center text-center cursor-pointer mb-6 transition-all ${
          previewUrl ? 'border-blue-400 bg-blue-50/30' : 'border-slate-300 hover:bg-slate-50'
        }`}
      >
        {previewUrl ? (
          <div className="flex flex-col items-center">
            <div className="relative">
              <img src={previewUrl} alt="Preview" className="w-32 h-32 object-cover rounded-xl shadow-md border border-slate-200 mb-4" />
              <button 
                onClick={clearSelection}
                className="absolute -top-3 -right-3 bg-red-100 text-red-600 rounded-full p-1.5 hover:bg-red-200 transition-colors shadow-sm"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <p className="text-sm font-medium text-slate-700">{selectedFile.name}</p>
          </div>
        ) : (
          <>
            <UploadCloud className="w-12 h-12 text-blue-500 mb-4" />
            <h3 className="text-lg font-semibold text-slate-700 mb-1">Drag suspect photo here</h3>
            <p className="text-sm text-slate-400">or click to browse from your computer (JPG, PNG)</p>
          </>
        )}
      </div>

      {/* Error Message */}
      {errorMsg && (
        <div className="mb-6 p-4 bg-red-50 text-red-700 border border-red-200 rounded-lg flex items-center">
          <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
          <span className="font-medium text-sm">{errorMsg}</span>
        </div>
      )}

      {/* Search Bar */}
      <div className="flex items-center gap-4 mb-8">
        <select className="flex-1 bg-white border border-slate-200 text-slate-700 rounded-lg px-4 py-2.5 outline-none focus:border-blue-500">
          <option>All Cameras</option>
          {/* We can dynamically fetch camera names from /api/system/stats later */}
        </select>
        <button 
          onClick={handleSearch}
          disabled={isSearching || !selectedFile}
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-8 py-2.5 rounded-lg flex items-center transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSearching ? <Loader2 className="w-5 h-5 mr-2 animate-spin" /> : <Search className="w-5 h-5 mr-2" />}
          {isSearching ? 'Analyzing...' : 'Search'}
        </button>
      </div>

      {/* Results Area */}
      {isSearching && (
        <div className="text-center py-12">
          <Loader2 className="w-10 h-10 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-slate-500 font-medium">Scanning surveillance database for vector matches...</p>
        </div>
      )}

      {!isSearching && hasSearched && (
        <div className="space-y-4">
          {searchResults.length > 0 ? (
            <>
              <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center">
                <AlertCircle className="w-5 h-5 mr-2 text-blue-500" />
                Found {searchResults.length} possible sightings
              </h3>
              {searchResults.map((res, idx) => (
                <SightingCard key={idx} data={res} />
              ))}
            </>
          ) : (
            <div className="text-center py-12 bg-white rounded-2xl border border-slate-200">
              <Search className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-slate-800">No Matches Found</h3>
              <p className="text-slate-500">The system could not locate this individual in the database.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ==========================================
// COMPONENT: Live Monitor View
// ==========================================
const LiveMonitorView = () => {
  return (
    <div className="flex h-full">
      <div className="flex-1 p-6 overflow-auto">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-slate-800">Live Monitor</h2>
          <span className="flex items-center text-sm font-medium text-green-600 bg-green-50 px-3 py-1 rounded-full border border-green-200">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2"></span>
            System Active
          </span>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((cam) => (
            <div key={cam} className="bg-slate-800 rounded-xl aspect-video relative overflow-hidden group">
              <div className="absolute top-4 left-4 text-white text-sm font-medium bg-black/50 px-3 py-1 rounded-md backdrop-blur-sm">
                Camera {cam}
              </div>
              <div className="w-full h-full flex items-center justify-center text-slate-600">
                <MonitorPlay className="w-12 h-12 opacity-50" />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="w-80 bg-white border-l border-slate-200 flex flex-col h-full">
        <div className="p-4 border-b border-slate-100">
          <h3 className="font-semibold text-slate-800">Recent Activity</h3>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {MOCK_LIVE_FEED.map((feed) => (
            <div key={feed.id} className="flex items-center space-x-3 p-2 hover:bg-slate-50 rounded-lg transition-colors cursor-default">
              <img src={feed.img} alt="Capture" className="w-10 h-10 rounded-full border border-slate-200" />
              <div>
                <p className="text-sm font-medium text-slate-800">{feed.cam}</p>
                <p className="text-xs text-slate-500">{feed.time}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ==========================================
// MAIN APP COMPONENT
// ==========================================
export default function App() {
  const [currentView, setCurrentView] = useState('investigator'); // Default to investigator so you can test it immediately!

  return (
    <div className="flex h-screen w-full font-sans">
      <nav className="w-64 bg-white border-r border-slate-200 flex flex-col z-10">
        <div className="p-6 border-b border-slate-100">
          <div className="flex items-center text-blue-600 font-bold text-xl tracking-tight">
            <MonitorPlay className="w-6 h-6 mr-2" />
            C.O.R.E.
          </div>
          <div className="text-xs text-slate-400 mt-1 uppercase font-semibold tracking-wider">Surveillance Engine</div>
        </div>

        <div className="flex-1 py-4 px-3 space-y-1 bg-white">
          <button 
            onClick={() => setCurrentView('monitor')}
            className={`w-full flex items-center px-3 py-2.5 rounded-lg font-medium transition-colors ${
              currentView === 'monitor' 
                ? 'bg-blue-50 text-blue-700' 
                : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
            }`}
          >
            <MonitorPlay className="w-5 h-5 mr-3" />
            Live Monitor
          </button>
          <button 
            onClick={() => setCurrentView('investigator')}
            className={`w-full flex items-center px-3 py-2.5 rounded-lg font-medium transition-colors ${
              currentView === 'investigator' 
                ? 'bg-blue-50 text-blue-700' 
                : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
            }`}
          >
            <UserSearch className="w-5 h-5 mr-3" />
            Investigator
          </button>
        </div>
      </nav>

      <main className="flex-1 h-screen overflow-hidden bg-slate-50 relative">
        <div className="absolute inset-0 overflow-y-auto">
            {currentView === 'monitor' ? <LiveMonitorView /> : <InvestigatorView />}
        </div>
      </main>
    </div>
  );
}