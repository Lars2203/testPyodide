// Set TensorFlow backend to WebAssembly for faster performance
tf.setBackend('wasm').then(() => {
    main();  // Run main only after the WASM backend is ready
});

// Load image and set up processing parameters
async function loadImage() {
    const img = new Image();
    img.src = 'gray.png';
    await new Promise(resolve => img.onload = resolve);

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    // Get image data and create a tensor using only one channel
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const data = new Uint8Array(imageData.width * imageData.height);

    // Extract only the grayscale data (usually the red channel)
    for (let i = 0; i < data.length; i++) {
        data[i] = imageData.data[i * 4]; // Use the red channel
    }

    // Create a tensor from the grayscale data
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
