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
        return tf.browser.fromPixels(imageData, 3)
            .toFloat()
            .div(tf.scalar(255));
    });
    
    originalImageTensor = tf.image.rgbToGrayscale(originalImageTensor);
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
    return tf.tidy(() => tf.ones([size, size]));
}

async function applyDilation() {
    const kernelSize = parseInt(kernelSizeSlider.value);
    kernelSizeValue.textContent = kernelSize;

    tf.tidy(() => {
        // Prepare input tensor shape [batch, height, width, channels]
        const input = currentImageTensor.expandDims(0);

        // Create dilation kernel (ones for dilation)
        const kernel = tf.ones([kernelSize, kernelSize, 1, 1]);

        // Apply 2D convolution (dilation effect)
        const dilated = tf.conv2d(input, kernel, [1, 1], 'same');
        currentImageTensor = dilated.squeeze([0, -1]);
        displayTensor(currentImageTensor);
    });
}

async function applyErosion() {
    const kernelSize = parseInt(kernelSizeSlider.value);
    kernelSizeValue.textContent = kernelSize;

    tf.tidy(() => {
        // Prepare input tensor shape [batch, height, width, channels]
        const input = currentImageTensor.expandDims(0);

        // Create erosion kernel (ones for erosion)
        const kernel = tf.ones([kernelSize, kernelSize, 1, 1]);

        console.log(input);

        // Apply 2D convolution (erosion effect)
        const eroded = tf.conv2d(input, kernel, [1, 1], 'same');
        currentImageTensor = eroded.squeeze([0, -1]);
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
