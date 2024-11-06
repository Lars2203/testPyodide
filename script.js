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
    const grayscaleImageData = new Uint8ClampedArray(imageData.width * imageData.height);
    for (let i = 0; i < imageData.data.length; i += 4) {
        const r = imageData.data[i];
        const g = imageData.data[i + 1];
        const b = imageData.data[i + 2];
        const gray = 0.2989 * r + 0.587 * g + 0.114 * b;  // RGB to grayscale conversion
        grayscaleImageData[i / 4] = gray;  // Store grayscale value
    }

    const grayscaleImage = new ImageData(grayscaleImageData, imageData.width, imageData.height);
    ctx.putImageData(grayscaleImage, 0, 0);

    originalImageTensor = tf.tidy(() => {
        return tf.browser.fromPixels(grayscaleImage, 1)
            .toFloat()
            .div(tf.scalar(255));
    });
    currentImageTensor = originalImageTensor.clone();

    applyThreshold();
}

async function applyDilation() {
    const kernelSize = parseInt(kernelSizeSlider.value);
    kernelSizeValue.textContent = kernelSize;

    tf.tidy(() => {
        // Ensure the input tensor has the correct rank [batch, height, width, channels]
        const input = currentImageTensor.expandDims(0).expandDims(-1); // [1, height, width, 1]

        // Create dilation kernel (ones for dilation)
        const kernel = tf.ones([kernelSize, kernelSize, 1, 1]);

        // Apply 2D convolution (dilation effect)
        const dilated = tf.conv2d(input, kernel, [1, 1], 'same');
        currentImageTensor = dilated.squeeze([0, -1]); // Remove batch and channel dims
        displayTensor(currentImageTensor);
    });
}

async function applyErosion() {
    const kernelSize = parseInt(kernelSizeSlider.value);
    kernelSizeValue.textContent = kernelSize;

    tf.tidy(() => {
        // Ensure the input tensor has the correct rank [batch, height, width, channels]
        const input = currentImageTensor.expandDims(0).expandDims(-1); // [1, height, width, 1]

        // Create erosion kernel (ones for erosion)
        const kernel = tf.ones([kernelSize, kernelSize, 1, 1]);

        // Apply 2D convolution (erosion effect)
        const eroded = tf.conv2d(input, kernel, [1, 1], 'same');
        currentImageTensor = eroded.squeeze([0, -1]); // Remove batch and channel dims
        displayTensor(currentImageTensor);
    });
}
