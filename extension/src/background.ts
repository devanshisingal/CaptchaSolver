// Background service worker

chrome.runtime.onMessage.addListener((request: any, sender: any, sendResponse: any) => {
    if (request.action === 'PREDICT_CAPTCHA') {
        const { base64, dimensions, score } = request.payload;
        
        const startTime = Date.now();
        
        // Call FastAPI backend
        fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64 })
        })
        .then(res => res.json())
        .then(data => {
            const latency = Date.now() - startTime;
            
            // Store results in chrome.storage for popup and analytics
            const resultData = {
                prediction: data.prediction,
                confidence: data.confidence,
                latency,
                dimensions,
                score,
                timestamp: Date.now()
            };
            
            chrome.storage.local.set({ latestPrediction: resultData });
            
            // Also update domain analytics
            const domain = sender.tab?.url ? new URL(sender.tab.url).hostname : 'unknown';
            updateAnalytics(domain, data.confidence);
            
            sendResponse({ success: true, data: resultData });
        })
        .catch(error => {
            console.error('Prediction failed:', error);
            sendResponse({ success: false, error: error.toString() });
        });
        
        return true; // Keep channel open
    }
});

function updateAnalytics(domain: string, confidence: number) {
    chrome.storage.local.get(['analytics'], (result: any) => {
        const analytics = result.analytics || {};
        
        if (!analytics[domain]) {
            analytics[domain] = { count: 0, totalConfidence: 0, avgConfidence: 0 };
        }
        
        analytics[domain].count += 1;
        analytics[domain].totalConfidence += confidence;
        analytics[domain].avgConfidence = analytics[domain].totalConfidence / analytics[domain].count;
        
        chrome.storage.local.set({ analytics });
    });
}
