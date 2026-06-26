export function scoreImage(img: HTMLImageElement): number {
    let score = 0;
    
    // 1. Check dimensions (CAPTCHAs are usually small rectangles, e.g. 150x50, 200x60)
    const width = img.width || img.naturalWidth;
    const height = img.height || img.naturalHeight;
    
    if (width > 80 && width < 300 && height > 30 && height < 120) {
        score += 20;
    }
    
    // 2. Check attributes (class, alt, id, src)
    const attributesString = `${img.className} ${img.alt} ${img.id} ${img.src}`.toLowerCase();
    if (attributesString.includes('captcha')) {
        score += 30;
    }
    if (attributesString.includes('challenge') || attributesString.includes('security')) {
        score += 15;
    }

    // 3. Proximity to input fields
    // A captcha image is almost always near a text input
    let currentElement: HTMLElement | null = img;
    let foundInput = false;
    
    // Look up to 3 levels up in DOM and search for inputs
    for (let i = 0; i < 3; i++) {
        if (!currentElement || !currentElement.parentElement) break;
        currentElement = currentElement.parentElement;
        
        const inputs = currentElement.querySelectorAll('input[type="text"], input:not([type])');
        for (let j = 0; j < inputs.length; j++) {
            const input = inputs[j] as HTMLInputElement;
            const inputAttr = `${input.name} ${input.id} ${input.placeholder}`.toLowerCase();
            if (inputAttr.includes('captcha')) {
                score += 40; // High signal
                foundInput = true;
                break;
            }
        }
        if (foundInput) break;
    }
    
    if (!foundInput) {
        // Just being near any input gives a small boost
        if (currentElement && currentElement.querySelector('input')) {
            score += 10;
        }
    }

    return score;
}
