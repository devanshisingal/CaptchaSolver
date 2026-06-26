import { scoreImage } from './scorer';
import { extractImageBase64 } from './extractor';

export interface ScanResult {
    imageSrc: string | null;
    base64: string | null;
    score: number;
    dimensions: string;
}

export function scanForCaptcha(): ScanResult | null {
    const images = Array.from(document.querySelectorAll('img'));
    
    let bestCandidate: HTMLImageElement | null = null;
    let highestScore = 0;

    for (const img of images) {
        const score = scoreImage(img);
        if (score > highestScore) {
            highestScore = score;
            bestCandidate = img;
        }
    }

    if (bestCandidate && highestScore > 30) {
        const base64 = extractImageBase64(bestCandidate);
        const dimensions = `${bestCandidate.width || bestCandidate.naturalWidth}x${bestCandidate.height || bestCandidate.naturalHeight}`;
        
        // Highlight it slightly for the demo
        bestCandidate.style.border = '2px solid #3b82f6';
        bestCandidate.style.boxShadow = '0 0 10px #3b82f6';

        return {
            imageSrc: bestCandidate.src,
            base64,
            score: highestScore,
            dimensions
        };
    }

    return null;
}
