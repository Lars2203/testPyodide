let originalImageTensor;
let currentImageTensor;
const inputCanvas = document.getElementById('inputCanvas');
const outputCanvas = document.getElementById('outputCanvas');
const thresholdSlider = document.getElementById('threshold');
const kernelSizeSlider = document.getElementById('kernelSize');
const thresholdValue = document.getElementById('thresholdValue');
const kernelSizeValue = document.getElementById('kernelSizeValue');

async function loadImage() {
    const img = new Image();
    img.src = 'gray.png';
    await img.decode();

    inputCanvas.width = img.width;
    inputCanvas.height = img.height;
    outputCanvas.width = img.width;
    outputCanvas.height = img.height;

    const ctx = inputCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    originalImageTensor = tf.tidy(() => {
        return tf.browser.fromPixels(imageData, 1)
            .toFloat()
            .div(tf.scalar(255));
    });
    currentImageTensor = originalImageTensor.clone();

    applyThreshold();
}

function applyThreshold() {
    const threshold = parseInt(thresholdSlider.value) / 100;
    thresholdValue.textContent = threshold.toFixed(2);

    tf.tidy(() => {
        const thresholded = currentImageTensor.greater(threshold).toFloat();
        displayTensor(thresholded);
    });
}

function createKernel(size) {
    return tf.tidy(() => tf.ones([size, size, 1, 1]).toFloat());
}

function applyDilation() {
    const kernelSize = parseInt(kernelSizeSlider.value);
    kernelSizeValue.textContent = kernelSize;
    
    tf.tidy(() => {
        const kernel = createKernel(kernelSize);
        const input = currentImageTensor.expandDims(0).expandDims(-1); // Expand to rank 4
        const dilated = tf.conv2d(input, kernel, 1, 'same');
        currentImageTensor = dilated.squeeze([0, -1]); // Squeeze back to rank 2
        displayTensor(currentImageTensor);
    });
}

function applyErosion() {
    const kernelSize = parseInt(kernelSizeSlider.value);
    kernelSizeValue.textContent = kernelSize;
    
    tf.tidy(() => {
        const kernel = createKernel(kernelSize);
        const input = currentImageTensor.expandDims(0).expandDims(-1); // Expand to rank 4
        const eroded = tf.conv2d(input, kernel.mul(-1).add(1), 1, 'same');
        currentImageTensor = eroded.squeeze([0, -1]); // Squeeze back to rank 2
        displayTensor(currentImageTensor);
    });
}

function resetImage() {
    currentImageTensor.dispose();
    currentImageTensor = originalImageTensor.clone();
    applyThreshold();
}

async function displayTensor(tensor) {
    const displayTensor = tf.tidy(() => {
        return tensor.clone().clipByValue(0, 1);
    });

    try {
        await tf.browser.toPixels(displayTensor, outputCanvas);
    } finally {
        displayTensor.dispose();
    }
}

thresholdSlider.addEventListener('input', applyThreshold);
kernelSizeSlider.addEventListener('input', () => {
    kernelSizeValue.textContent = kernelSizeSlider.value;
});

loadImage().catch(console.error);

window.addEventListener('unload', () => {
    if (originalImageTensor) originalImageTensor.dispose();
    if (currentImageTensor) currentImageTensor.dispose();
});