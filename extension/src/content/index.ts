import { scanForCaptcha } from './scanner';

// Message listener to trigger scanning from popup
chrome.runtime.onMessage.addListener((request: any, sender: any, sendResponse: any) => {
    if (request.action === 'SCAN_CAPTCHA') {
        const result = scanForCaptcha();
        
        if (result && result.base64) {
            // Forward the base64 image to the background script for prediction
            chrome.runtime.sendMessage({
                action: 'PREDICT_CAPTCHA',
                payload: {
                    base64: result.base64,
                    dimensions: result.dimensions,
                    score: result.score
                }
            });
            sendResponse({ status: 'found', data: result });
        } else {
            sendResponse({ status: 'not_found' });
        }
    }
    return true; // Keep message channel open for async response
});
