'use client'

import dynamic from 'next/dynamic'

// WebGL needs the browser, so the 3D scene is rendered client-side only
const NeuralVisualizerApp = dynamic(() => import('./NeuralVisualizerApp'), { ssr: false })

export default function App() {
  return (
    <main className="relative h-screen w-screen overflow-hidden bg-black">
      <NeuralVisualizerApp />
    </main>
  )
}
