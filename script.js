// Load TensorFlow.js with WASM backend
tf.setBackend('wasm').then(() => {
    console.log("WASM backend is ready");
    main();  // Only call main after the WASM backend is confirmed
}).catch(error => {
    console.error("Failed to load WASM backend:", error);
});

async function loadImage() {
    const img = new Image();
    img.src = 'gray.png';
    await new Promise(resolve => img.onload = resolve);

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const data = new Uint8Array(img.width * img.height);

    // Convert RGBA image to grayscale using only one channel
    for (let i = 0; i < data.length; i++) {
        data[i] = imageData.data[i * 4]; // Use the red channel or a grayscale formula if needed
    }

    // Create a tensor with the correct shape
    return tf.tensor(data, [img.height, img.width], 'int32');
}


// Functions for thresholding, dilation, and erosion
function threshold(tensor, thresholdValue) {
    return tensor.greaterEqual(tf.scalar(thresholdValue)).toInt();
}

function dilation(tensor, size) {
    return tensor.dilation2d(tf.ones([size, size]), [1, 1, 1]);
}

function erosion(tensor, size) {
    return tensor.erosion2d(tf.ones([size, size]), [1, 1, 1]);
}

// Update display with new image processing results
function updateCanvas(tensor) {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    tensor.data().then(data => {
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < data.length; i++) {
            imageData.data[i * 4] = data[i];
            imageData.data[i * 4 + 1] = data[i];
            imageData.data[i * 4 + 2] = data[i];
            imageData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);
    });
}

// Main processing logic
async function main() {
    let tensor = await loadImage();
    document.getElementById('thresholdSlider').oninput = () => {
        const thresholdValue = +document.getElementById('thresholdSlider').value;
        const result = threshold(tensor, thresholdValue);
        updateCanvas(result);
    };

    document.getElementById('dilationSlider').oninput = () => {
        const size = +document.getElementById('dilationSlider').value;
        const result = dilation(tensor, size);
        updateCanvas(result);
    };

    document.getElementById('erosionSlider').oninput = () => {
        const size = +document.getElementById('erosionSlider').value;
        const result = erosion(tensor, size);
        updateCanvas(result);
    };
}

main();
