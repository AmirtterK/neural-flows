'use client'

import { useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { Bloom, EffectComposer } from '@react-three/postprocessing'
import * as THREE from 'three'
import { LAYER_GAP, NODES, layerX } from '@/data/network'
import { SCENARIO_BY_ID, firstDivergence } from '@/data/scenarios'
import { labels } from '@/lib/labels'
import { LAST_STEP, anim, useStore } from '@/lib/store'
import { NodeView } from './Nodes'
import { BaseEdges, ScenarioEdges } from './Edges'

/** Advances the animation clock every frame and mirrors the integer step into the store for the UI. */
function Driver() {
  useFrame((_, dt) => {
    const st = useStore.getState()
    if (st.playing) {
      anim.target = Math.min(LAST_STEP, anim.target + dt * 0.55)
      anim.progress = anim.target
      if (anim.target >= LAST_STEP) st.pause()
    } else {
      anim.progress += (anim.target - anim.progress) * (1 - Math.exp(-6 * dt))
      if (Math.abs(anim.target - anim.progress) < 0.001) anim.progress = anim.target
    }
    const s = Math.min(LAST_STEP, Math.floor(anim.progress + 0.05))
    if (s !== st.step) st.syncStep(s)
  })
  return null
}

/** Moves the DOM labels (see lib/labels.ts) to the screen position of their 3D anchor. */
function Projector() {
  const v = useMemo(() => new THREE.Vector3(), [])
  useFrame(({ camera, size }) => {
    labels.forEach(({ el, pos, kind }) => {
      v.copy(pos).project(camera)
      const x = (v.x * 0.5 + 0.5) * size.width
      const y = (-v.y * 0.5 + 0.5) * size.height
      const scale = THREE.MathUtils.clamp(10 / camera.position.distanceTo(pos), 0.6, 1.5)
      el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%) scale(${scale})`
      let show = v.z < 1
      if (kind === 'diverge') show = show && anim.progress >= Number(el.dataset.layer) - 0.4
      el.style.visibility = show ? 'visible' : 'hidden'
    })
  })
  return null
}

/** Thin vertical line marking where the selected routes stop being identical. */
function DivergenceLine() {
  const selected = useStore((s) => s.selected)
  const layer = useMemo(
    () => firstDivergence(selected.map((id) => SCENARIO_BY_ID[id])),
    [selected]
  )
  const line = useMemo(() => ({ current: null as THREE.Mesh | null }), [])

  useFrame(() => {
    if (line.current && layer !== null) line.current.visible = anim.progress >= layer - 0.4
  })

  if (layer === null) return null
  return (
    <mesh
      ref={(m) => {
        line.current = m
      }}
      position={[layerX(layer) - LAYER_GAP / 2, 0, 0]}
      visible={false}
    >
      <planeGeometry args={[0.015, 6.6]} />
      <meshBasicMaterial color="#ffffff" transparent opacity={0.35} toneMapped={false} />
    </mesh>
  )
}

function World() {
  const selected = useStore((s) => s.selected)
  const scenarios = useMemo(() => selected.map((id) => SCENARIO_BY_ID[id]), [selected])

  return (
    <>
      <Driver />
      <Projector />
      <BaseEdges />
      {scenarios.map((s, i) => (
        <ScenarioEdges key={s.id} scenario={s} index={i} total={scenarios.length} />
      ))}
      {NODES.map((n) => (
        <NodeView key={n.id} node={n} scenarios={scenarios} />
      ))}
      <DivergenceLine />
    </>
  )
}

export default function Scene() {
  return (
    <Canvas
      camera={{ position: [0, 0.4, 13.5], fov: 45 }}
      dpr={[1, 2]}
      onPointerMissed={() => useStore.getState().selectNode(null)}
    >
      <color attach="background" args={['#070a12']} />
      <fog attach="fog" args={['#070a12', 16, 34]} />
      <ambientLight intensity={0.6} />
      <pointLight position={[0, 6, 8]} intensity={40} />
      <World />
      <OrbitControls
        enablePan={false}
        enableDamping
        minDistance={7}
        maxDistance={22}
        minPolarAngle={Math.PI * 0.25}
        maxPolarAngle={Math.PI * 0.75}
        minAzimuthAngle={-Math.PI * 0.4}
        maxAzimuthAngle={Math.PI * 0.4}
      />
      <EffectComposer>
        <Bloom mipmapBlur intensity={0.9} luminanceThreshold={0.25} luminanceSmoothing={0.4} />
      </EffectComposer>
    </Canvas>
  )
}
