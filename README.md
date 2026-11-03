# Neural Flows

An interactive 3D neural network visualizer built with Next.js, Three.js, and Tailwind CSS.

Neural Flows visualizes forward propagation through a multi-layer perceptron (MLP) trained to classify handwritten letters (A-Z, 26 classes). Inference runs entirely in the browser and requires no Python, PyTorch, or backend services.

---

## Features

- **Interactive 3D multi-layer architecture**
  - Input layer: 784 nodes ($28 \times 28$ input grid).
  - Hidden layer 1: 64 neurons (ReLU activation).
  - Hidden layer 2: 32 neurons (ReLU activation).
  - Output layer: 26 neurons (A-Z softmax probabilities) with 3D floating sprite labels.
- **Client-side inference**
  - Forward propagation is computed in JavaScript using typed arrays and rendered with WebGL.
  - Predictions update as the user draws.
- **Idle pulse animation**
  - With an empty canvas, a light pulse travels from the input layer across every connection to each layer in turn, until all 26 letter nodes flash. It loops, and stops as soon as you start drawing.
- **EMNIST-style input normalization**
  - Crops the drawn strokes to their bounding box, scales the long side to 24 px (keeping the aspect ratio) and centers it in the $28 \times 28$ frame, the way EMNIST images were prepared, so letter size and position don't hurt accuracy.
- **Synapse and neuron inspector**
  - Selecting a neuron or input pixel bar in the 3D scene isolates its incoming and outgoing synapses.
  - Selected connections are highlighted in blue (`#4da6ff`) with a radial selection glow.
  - An inspection card displays the pre-activation, activation, bias, and top synapse contributions.
- **Monochrome interface**
  - Minimal black-and-white UI.
  - Connection cylinders rest in a dim blue-gray and are colored during active drawing (green for excitatory, red for inhibitory).
  - Custom vector favicon.
- **26-letter preset palette**
  - A compact four-column grid of preset buttons for testing letters A through Z.

---

## Architecture and Model Specification

```
Input (784) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(26, Softmax)
```

| Layer | Dimensions | Activation | Outgoing Synapses |
|---|---|---|---|
| Input | 784 ($28 \times 28$) | Linear | 50,176 |
| Hidden 1 | 64 | ReLU | 2,048 |
| Hidden 2 | 32 | ReLU | 832 |
| Output | 26 | Softmax | N/A (predicted letter) |

Total: 53,178 parameters (53,056 weights + 122 biases).

Model weights and biases are bundled as JSON in `data/model_weights.json`.

---

## Development Overview

1. **Migration and modernization**
   - Migrated legacy vanilla Three.js scripts to a Next.js 15 application with TypeScript, using dynamic loading for server-side rendering compatibility.
2. **Alphabet expansion (0-9 to A-Z)**
   - Expanded the output space from 10 digits to 26 uppercase English letters.
   - Reworked the 3D sprite labels and probability distribution display accordingly.
3. **Training and accuracy**
   - Trained on the EMNIST Letters dataset with GPU data augmentation (rotation, scale, shear, shifts, stroke thickness, blur and binarized strokes) to match finger- and mouse-drawn input.
   - Added EMNIST-style bounding-box scaling and centering at inference time to compensate for differently sized and off-center hand-drawn input.
4. **Rendering optimization**
   - Uses `THREE.InstancedMesh` with dynamic buffer attributes for connection cylinders and neuron geometry, sustaining 60 FPS in standard browser viewports.
5. **Synapse isolation**
   - Implemented raycasting pointer handlers to select individual neurons in any layer.
   - Adjacent synapses of the selected neuron are highlighted in blue (`#4da6ff`).

---

## Getting Started

### Prerequisites

- Node.js 18 or later
- npm, pnpm, or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/AmirtterK/neural-flows.git
cd neural-flows

# Install dependencies
npm install

# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in a browser.

### Production Build

```bash
npm run build
npm run start
```

---

## Retraining the Model

A Google Colab notebook, `train_emnist_colab.ipynb`, is included in the repository. It trains the network on the full EMNIST Letters dataset (145,600 samples) and exports the updated weights file. Training on a free GPU takes a few minutes. Hyperparameters: AdamW (weight decay 1e-4), one-cycle learning rate schedule (max lr 0.004), batch size 256, 80 epochs, cross-entropy loss, GPU data augmentation, and a 5% validation split used to pick the best epoch. The notebook reports clean and augmented test accuracy and the most confused letters, and exports the normalization constants computed from the training data. EMNIST Letters merges upper- and lower-case forms into one class per letter.

---

## Tech Stack

- **Framework:** Next.js (App Router), React
- **3D graphics:** Three.js, OrbitControls
- **Styling:** Tailwind CSS
- **Language:** TypeScript

---

## License

Released under the MIT License.