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

    // Convert image to grayscale by averaging the RGB values
    const grayscaleData = new Uint8ClampedArray(imageData.width * imageData.height);

    for (let i = 0; i < imageData.data.length; i += 4) {
        // Get the RGB values
        const r = imageData.data[i];
        const g = imageData.data[i + 1];
        const b = imageData.data[i + 2];

        // Convert to grayscale using the average of RGB
        const gray = (r + g + b) / 3;

        // Store the grayscale value in the new array (using only 1 channel, grayscale)
        grayscaleData[i / 4] = gray;
    }

    // Create a new ImageData object with the grayscale values
    const grayscaleImageData = new ImageData(new Uint8ClampedArray(grayscaleData.buffer), img.width, img.height);

    // Update the input canvas with the grayscale image
    ctx.putImageData(grayscaleImageData, 0, 0);

    // Convert the grayscale image to tensor
    originalImageTensor = tf.tidy(() => {
        return tf.browser.fromPixels(grayscaleImageData, 1) // 1 indicates grayscale
            .toFloat()
            .div(tf.scalar(255));
    });
    currentImageTensor = originalImageTensor.clone();

    applyThreshold();
}

function applyThreshold() {
    console.log(tf.version);
    console.log(tf.getBackend()); 
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
        const input = currentImageTensor.expandDims(0).expandDims(-1);

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
        const input = currentImageTensor.expandDims(0).expandDims(-1);

        // Create erosion kernel (ones for erosion)
        const kernel = tf.ones([kernelSize, kernelSize, 1, 1]);

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
