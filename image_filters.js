import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@3.18.0/dist/tf.min.js';
import * as tfjsWasm from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.18.0/dist/tf-backend-wasm.min.js';

await tfjsWasm.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.18.0/dist/tfjs-backend-wasm.wasm');

export async function apply_filters(imagePath, threshold, kernelSize) {
  await loadPyodide(); // Load the Pyodide WebAssembly module

  // Load the grayscale image
  const img = await Pyodide.runPythonAsync(`
    import numpy as np
    from PIL import Image
    image = Image.open('${imagePath}')
    image = np.array(image.convert('L'))
    return image
  `);

  // Apply thresholding
  const thresholdedImg = await Pyodide.runPythonAsync(`
    import numpy as np
    image = np.array(${tf.tensor(img).dataSync()})
    image[image < ${threshold}] = 0
    image[image >= ${threshold}] = 255
    return image
  `);

  // Apply morphological operations
  const morphedImg = await Pyodide.runPythonAsync(`
    import numpy as np
    from scipy.ndimage.morphology import binary_opening, binary_closing
    image = np.array(${tf.tensor(thresholdedImg).dataSync()})
    kernel = np.ones((${kernelSize}, ${kernelSize}), np.uint8)
    image = binary_opening(image, kernel)
    image = binary_closing(image, kernel)
    return image
  `);

  // Convert the processed image back to a data URL
  const processedImageData = await Pyodide.runPythonAsync(`
    import numpy as np
    from PIL import Image
    import io
    import base64
    image = np.array(${tf.tensor(morphedImg).dataSync()}, dtype=np.uint8)
    pil_image = Image.fromarray(image)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")
  `);

  return `data:image/png;base64,${processedImageData}`;
}

async function loadPyodide() {
  if (!window.Pyodide) {
    await loadPyodideAndDependencies();
  }
}

async function loadPyodideAndDependencies() {
  const pyodide = await window.loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/'
  });
  await pyodide.loadPackage(['numpy', 'scipy']);
  window.Pyodide = pyodide;
}