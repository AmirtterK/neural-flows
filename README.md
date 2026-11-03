# Neural Flow 🧠✨

An interactive, high-performance **3D Neural Network Visualizer** built with **Next.js**, **Three.js**, and **Tailwind CSS**. 

**Neural Flow** visualizes real-time forward propagation through a multi-layer perceptron (MLP) trained to classify uppercase letters (**A–Z**, 26 classes) directly in the browser with zero Python, PyTorch, or backend requirements.

---

## ✨ Features

- **🌐 Interactive 3D Multi-Layer Architecture**
  - **Input Layer:** 784 nodes ($28 \times 28$ input retina).
  - **Hidden Layer 1:** 64 neurons (ReLU activation).
  - **Hidden Layer 2:** 32 neurons (ReLU activation).
  - **Output Layer:** 26 neurons (A–Z Softmax probabilities) with dynamic 3D floating sprite labels.
- **⚡ Client-Side WebGL Inference**
  - Instant forward propagation running entirely in WebGL and JavaScript via Web Workers/typed arrays.
  - Complete zero-latency letter prediction as you sketch.
- **🎯 Center-of-Mass Auto-Alignment**
  - Automatically calculates the center-of-mass $(\bar{x}, \bar{y})$ of drawn strokes and translates them to $(13.5, 13.5)$, dramatically increasing classification accuracy for off-center sketches.
- **🔍 Synapse & Neuron Inspector**
  - Click any neuron or input pixel bar in the 3D scene to isolate its incoming and outgoing synapses.
  - Connective lines illuminate in vibrant electric blue (`#4da6ff`) with radial selection glow.
  - Displays a centered, scroll-free inspection card showing exact pre-activation, activation, bias, and top synapse contributions.
- **🎨 Modern Monochrome Aesthetic**
  - Minimalist black-and-white UI designed for maximum clarity.
  - All connection cylinders remain black by default, illuminating dynamically during active drawing (green for excitatory, red for inhibitory).
  - Custom vector neural favicon for modern browser tabs.
- **🔤 26-Letter Preset Palette**
  - 4-column compact preset button grid for instant testing of all letters A through Z without scrolling.

---

## 🏗️ Architecture & Model Specs

```
Input (784) ──► Dense(64, ReLU) ──► Dense(32, ReLU) ──► Dense(26, Softmax)
```

| Layer | Dimensions | Activation | Synapses Out |
|---|---|---|---|
| **Input** | 784 ($28 \times 28$) | Linear | 50,176 |
| **Hidden 1** | 64 | ReLU | 2,048 |
| **Hidden 2** | 32 | ReLU | 832 |
| **Output** | 26 | Softmax | — (Predicted Letter) |

Model weights and biases are bundled directly as JSON in `data/model_weights.json`.

---

## 🛠️ How the Work Was Done

1. **Porting & Modernization**:
   - Upgraded from legacy vanilla Three.js scripts into a modern, SSR-safe Next.js 15 application using React dynamic loading and TypeScript.
2. **Alphabet Expansion (0–9 ➔ A–Z)**:
   - Expanded the network output space from 10 digits to 26 uppercase English letters.
   - Re-architected 3D sprite billboard typography and probability distributions.
3. **Weight Training & Accuracy Optimization**:
   - Trained on augmented letterforms with varying stroke thicknesses, dilations, and spatial offsets.
   - Introduced center-of-mass stroke normalization prior to inference to eliminate positional sensitivity.
4. **WebGL Rendering Optimization**:
   - Employs `THREE.InstancedMesh` with dynamic buffer attributes for thousands of connection cylinders and neuron geometry, achieving 60 FPS in standard browser viewports.
5. **Interactive Synapse Isolation**:
   - Implemented 3D raycasting pointer handlers to allow picking individual neurons across any layer.
   - Highlighting highlights immediate adjacent synapses in electric blue (`#4da6ff`).

---

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm or pnpm / yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/AmirtterK/neural-flow.git
cd neural-flow

# Install dependencies
npm install

# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm run start
```

---

## 📓 Retraining the Model

A companion Google Colab notebook is provided in the repository:
- **`train_emnist_colab.ipynb`**

You can open this notebook directly in Google Colab to train the network on the full **EMNIST Letters** dataset (145,600 samples) on a free GPU in ~15 seconds and export the updated `mlp_weights.json`.

---

## 🛠️ Tech Stack

- **Framework:** Next.js (App Router), React
- **3D Graphics:** Three.js, OrbitControls
- **Styling:** Tailwind CSS, Lucide Icons
- **Language:** TypeScript

---

## 📄 License

MIT License. Feel free to use and build upon this project!
