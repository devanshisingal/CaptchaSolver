export function extractImageBase64(img: HTMLImageElement): string | null {
    try {
        const canvas = document.createElement('canvas');
        canvas.width = img.width || img.naturalWidth;
        canvas.height = img.height || img.naturalHeight;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;
        
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64
        return canvas.toDataURL('image/png');
    } catch (e) {
        console.error('Failed to extract image:', e);
        return null;
    }
}
