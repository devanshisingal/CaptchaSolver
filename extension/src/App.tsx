import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { ShieldCheck, Activity, Search, AlertTriangle } from 'lucide-react';
import './index.css';

interface PredictionData {
  prediction: string;
  confidence: number;
  latency: number;
  dimensions: string;
  score: number;
}

function App() {
  const [data, setData] = useState<PredictionData | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadLatest = () => {
    chrome.storage.local.get(['latestPrediction'], (result) => {
      if (result.latestPrediction) {
        setData(result.latestPrediction);
      }
    });
  };

  useEffect(() => {
    loadLatest();
    // Listen for storage changes
    chrome.storage.onChanged.addListener((changes) => {
      if (changes.latestPrediction) {
        setData(changes.latestPrediction.newValue);
        setScanning(false);
      }
    });
  }, []);

  const handleScan = async () => {
    setScanning(true);
    setError(null);
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab.id) {
        chrome.tabs.sendMessage(tab.id, { action: 'SCAN_CAPTCHA' }, (response) => {
          if (chrome.runtime.lastError) {
             setError("Please refresh the page to inject the content script.");
             setScanning(false);
             return;
          }
          if (response && response.status === 'not_found') {
             setError("No CAPTCHA detected on this page.");
             setScanning(false);
          }
        });
      }
    } catch (e) {
      setError(String(e));
      setScanning(false);
    }
  };

  return (
    <div className="w-[340px] p-4 font-sans bg-background text-white min-h-[400px] flex flex-col relative overflow-hidden">
      {/* Decorative background blur */}
      <div className="absolute top-[-50px] right-[-50px] w-32 h-32 bg-primary/30 rounded-full blur-3xl"></div>
      <div className="absolute bottom-[-50px] left-[-50px] w-32 h-32 bg-accent/30 rounded-full blur-3xl"></div>

      <header className="flex items-center gap-2 mb-6 relative z-10">
        <ShieldCheck className="text-primary w-6 h-6" />
        <h1 className="text-lg font-semibold tracking-wide">CAPTCHA Vision</h1>
      </header>

      <div className="flex-1 relative z-10 flex flex-col gap-4">
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }} 
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-500/20 border border-red-500/50 p-3 rounded-lg flex items-start gap-2 text-sm text-red-200"
          >
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <p>{error}</p>
          </motion.div>
        )}

        <div className="glass-panel rounded-xl p-4 flex-1 flex flex-col items-center justify-center min-h-[200px]">
          {scanning ? (
            <div className="flex flex-col items-center text-primary">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
              >
                <Search className="w-8 h-8 mb-2" />
              </motion.div>
              <p className="text-sm font-medium animate-pulse">Scanning DOM...</p>
            </div>
          ) : data ? (
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="w-full"
            >
              <div className="text-center mb-4">
                <p className="text-xs text-gray-400 uppercase tracking-widest mb-1">Status</p>
                <p className="text-sm text-green-400 font-medium bg-green-400/10 inline-block px-3 py-1 rounded-full border border-green-400/20">
                  CAPTCHA Found
                </p>
              </div>

              <div className="bg-surface/50 rounded-lg p-4 border border-white/5 mb-4 text-center shadow-inner">
                <p className="text-xs text-gray-400 mb-1">Prediction</p>
                <p className="text-3xl font-bold tracking-widest text-white drop-shadow-md">
                  {data.prediction}
                </p>
                {data.confidence < 0.7 && (
                  <p className="text-xs text-amber-400 mt-2 flex items-center justify-center gap-1">
                    <AlertTriangle className="w-3 h-3" /> Low confidence
                  </p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-surface/30 p-2 rounded border border-white/5">
                  <p className="text-[10px] text-gray-400 uppercase">Confidence</p>
                  <p className="text-sm font-medium">{(data.confidence * 100).toFixed(1)}%</p>
                </div>
                <div className="bg-surface/30 p-2 rounded border border-white/5">
                  <p className="text-[10px] text-gray-400 uppercase">Solve Time</p>
                  <p className="text-sm font-medium">{data.latency}ms</p>
                </div>
                <div className="bg-surface/30 p-2 rounded border border-white/5">
                  <p className="text-[10px] text-gray-400 uppercase">Dimensions</p>
                  <p className="text-sm font-medium">{data.dimensions}</p>
                </div>
                <div className="bg-surface/30 p-2 rounded border border-white/5">
                  <p className="text-[10px] text-gray-400 uppercase">Preprocessed</p>
                  <p className="text-sm font-medium text-primary">YES</p>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="text-center text-gray-400">
              <Activity className="w-12 h-12 mx-auto mb-3 opacity-20" />
              <p className="text-sm">Ready to scan the current page for CAPTCHAs.</p>
            </div>
          )}
        </div>

        <button 
          onClick={handleScan}
          disabled={scanning}
          className="w-full py-3 rounded-lg bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity font-medium shadow-lg shadow-primary/20 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {data ? 'Analyze Again' : 'Scan Page'}
        </button>
      </div>
    </div>
  );
}

export default App;
