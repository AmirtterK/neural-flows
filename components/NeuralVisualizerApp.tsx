'use client'

import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import weightsData from '@/data/model_weights.json'

interface LayerMeta {
  name: string
  activation: string
  weights: number[][]
  biases: number[]
}

interface SelectionDetail {
  layerIndex: number
  neuronIndex: number
  layerLabel: string
  activationName: string | null
  activationValue: number | null
  preActivation: number | null
  bias: number | null
  incoming: {
    sourceIndex: number
    weight: number
    sourceActivation: number
    contribution: number
  }[]
  outgoing: {
    targetIndex: number
    weight: number
    targetActivation: number
    contribution: number
  }[]
  previousLayerSize: number | null
  nextLayerSize: number | null
}

const LETTERS = weightsData.classes || Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function softmax(arr: number[] | Float32Array): number[] {
  const values = Array.from(arr)
  if (!values.length) return []
  const maxVal = Math.max(...values)
  const exps = values.map((v) => Math.exp(v - maxVal))
  const sum = exps.reduce((acc, v) => acc + v, 0)
  return exps.map((v) => (sum === 0 ? 0 : v / sum))
}

function maxAbsValue(values: number[] | Float32Array): number {
  let max = 0
  for (let i = 0; i < values.length; i++) {
    const mag = Math.abs(values[i])
    if (mag > max) max = mag
  }
  return max
}

function centerDrawnPixels(raw: Float32Array): Float32Array {
  let totalMass = 0
  let weightedY = 0
  let weightedX = 0

  for (let r = 0; r < 28; r++) {
    for (let c = 0; c < 28; c++) {
      const v = raw[r * 28 + c]
      if (v > 0.04) {
        totalMass += v
        weightedY += r * v
        weightedX += c * v
      }
    }
  }

  if (totalMass < 1e-4) return raw

  const cy = weightedY / totalMass
  const cx = weightedX / totalMass

  const shiftR = Math.round(13.5 - cy)
  const shiftC = Math.round(13.5 - cx)

  if (shiftR === 0 && shiftC === 0) return raw

  const out = new Float32Array(784)
  for (let r = 0; r < 28; r++) {
    const nr = r + shiftR
    if (nr >= 0 && nr < 28) {
      for (let c = 0; c < 28; c++) {
        const nc = c + shiftC
        if (nc >= 0 && nc < 28) {
          out[nr * 28 + nc] = raw[r * 28 + c]
        }
      }
    }
  }
  return out
}

export default function NeuralVisualizerApp() {
  const mountRef = useRef<HTMLDivElement>(null)

  // Settings
  const [maxConnections, setMaxConnections] = useState(24)
  const [connectionThreshold, setConnectionThreshold] = useState(0.0)
  const [connectionThickness, setConnectionThickness] = useState(0.0035)
  const [brushSize, setBrushSize] = useState(1.5)
  const [brushStrength, setBrushStrength] = useState(0.95)
  const [showFps, setShowFps] = useState(true)
  const [fps, setFps] = useState(60)

  // Modals
  const [infoOpen, setInfoOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)

  // Probabilities & Selection state
  const [probabilities, setProbabilities] = useState<number[]>(new Array(26).fill(0))
  const [selectedDetail, setSelectedDetail] = useState<SelectionDetail | null>(null)
  const [hasDrawn, setHasDrawn] = useState(false)

  // References to visualizer controllers
  const visualizerRef = useRef<any>(null)

  // Grid Refs
  const gridCellsRef = useRef<HTMLDivElement[]>([])
  const gridValuesRef = useRef<Float32Array>(new Float32Array(784))
  const isDrawingRef = useRef(false)
  const drawModeRef = useRef<'draw' | 'erase'>('draw')

  // Load and setup Three.js scene
  useEffect(() => {
    if (!mountRef.current) return

    const container = mountRef.current
    let animationFrameId: number
    let isDisposed = false

    // Dimensions
    const width = container.clientWidth || window.innerWidth
    const height = container.clientHeight || window.innerHeight

    // Scene
    const scene = new THREE.Scene()
    scene.background = null

    // Camera: (-18, 6, 18) looking at (0, 0, 0)
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 200)
    camera.position.set(-18, 6, 18)
    // Pull the camera back on narrow (portrait) viewports so the whole network fits
    if (width / height < 1.6) {
      camera.position.multiplyScalar(Math.min(2, 1.6 / (width / height)))
    }

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(width, height)
    renderer.domElement.style.position = 'absolute'
    renderer.domElement.style.inset = '0'
    renderer.domElement.style.pointerEvents = 'auto'
    container.appendChild(renderer.domElement)

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 8
    controls.maxDistance = 52
    controls.target.set(0, 0, 0)

    // Lighting (Studio setup matching neuralviz3d)
    const ambient = new THREE.AmbientLight(0xffffff, 0.75)
    scene.add(ambient)

    const hemisphere = new THREE.HemisphereLight(0x99b7ff, 0x0b1b2d, 0.45)
    hemisphere.position.set(0, 20, 0)
    scene.add(hemisphere)

    const directional = new THREE.DirectionalLight(0xffffff, 1.2)
    directional.position.set(24, 36, 20)
    scene.add(directional)

    const fillLight = new THREE.DirectionalLight(0x88c6ff, 0.55)
    fillLight.position.set(-24, 18, -18)
    scene.add(fillLight)

    const rimLight = new THREE.PointLight(0x88a4ff, 1.1, 80, 1.6)
    rimLight.position.set(0, 24, -36)
    scene.add(rimLight)

    // Selection Glow Sprite Texture
    const createGlowTexture = () => {
      const size = 256
      const canvas = document.createElement('canvas')
      canvas.width = size
      canvas.height = size
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.clearRect(0, 0, size, size)
        const gradient = ctx.createRadialGradient(
          size / 2, size / 2, size * 0.12,
          size / 2, size / 2, size * 0.5
        )
        gradient.addColorStop(0, 'rgba(77, 166, 255, 0.95)')
        gradient.addColorStop(0.35, 'rgba(39, 132, 255, 0.7)')
        gradient.addColorStop(1, 'rgba(18, 64, 158, 0)')
        ctx.fillStyle = gradient
        ctx.fillRect(0, 0, size, size)
      }
      const texture = new THREE.CanvasTexture(canvas)
      texture.needsUpdate = true
      return texture
    }

    const glowMaterial = new THREE.SpriteMaterial({
      map: createGlowTexture(),
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      opacity: 0.95,
      color: 0x4da6ff,
    })
    const selectionGlowSprite = new THREE.Sprite(glowMaterial)
    selectionGlowSprite.visible = false
    selectionGlowSprite.renderOrder = 10
    scene.add(selectionGlowSprite)

    // Letter Sprites (A to Z)
    const labelGroup = new THREE.Group()
    scene.add(labelGroup)
    const outputLabels: THREE.Sprite[] = []

    const buildLetterSprite = (text: string) => {
      const size = 128
      const canvas = document.createElement('canvas')
      canvas.width = size
      canvas.height = size
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.clearRect(0, 0, size, size)
        ctx.font = `900 ${Math.floor(size * 0.65)}px "Inter", "Segoe UI", sans-serif`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillStyle = 'rgba(235, 245, 255, 0.95)'
        ctx.strokeStyle = 'rgba(12, 25, 44, 0.85)'
        ctx.lineWidth = size * 0.08
        ctx.shadowColor = 'rgba(0, 0, 0, 0.55)'
        ctx.shadowBlur = size * 0.12
        ctx.strokeText(text, size / 2, size / 2)
        ctx.fillText(text, size / 2, size / 2)
      }
      const texture = new THREE.CanvasTexture(canvas)
      texture.needsUpdate = true
      const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthTest: false,
        depthWrite: false,
      })
      const sprite = new THREE.Sprite(material)
      sprite.center.set(0, 0.5)
      return sprite
    }

    // Architecture & Layers setup
    const arch = weightsData.architecture // [784, 64, 32, 26]
    const layers = weightsData.layers as LayerMeta[]
    const layerSpacing = 5.5
    const inputSpacing = 0.24
    const hiddenSpacing = 0.95
    const outputSpacing = 0.48 // Compact vertical spacing for 26 letters
    const inputNodeSize = 0.18
    const hiddenNodeRadius = 0.22
    const outputNodeRadius = 0.19

    const totalWidth = (arch.length - 1) * layerSpacing
    const startX = -totalWidth / 2

    // Compute layer positions
    const computePositions = (layerIndex: number, neuronCount: number, layerX: number) => {
      const positions: THREE.Vector3[] = []
      const isOutput = layerIndex === arch.length - 1

      if (layerIndex === 0) {
        const rows = 28
        const cols = 28
        const h = (rows - 1) * inputSpacing
        const w = (cols - 1) * inputSpacing
        let filled = 0
        for (let r = 0; r < rows && filled < neuronCount; r++) {
          for (let c = 0; c < cols && filled < neuronCount; c++) {
            const y = h / 2 - r * inputSpacing
            const z = -w / 2 + c * inputSpacing
            positions.push(new THREE.Vector3(layerX, y, z))
            filled++
          }
        }
      } else if (isOutput) {
        // 26 letters arranged in a clean vertical column
        const h = (neuronCount - 1) * outputSpacing
        for (let i = 0; i < neuronCount; i++) {
          const y = h / 2 - i * outputSpacing
          positions.push(new THREE.Vector3(layerX, y, 0))
        }
      } else {
        const cols = Math.max(1, Math.ceil(Math.sqrt(neuronCount)))
        const rows = Math.ceil(neuronCount / cols)
        const h = (rows - 1) * hiddenSpacing
        const w = (cols - 1) * hiddenSpacing
        for (let i = 0; i < neuronCount; i++) {
          const r = Math.floor(i / cols)
          const c = i % cols
          const y = h / 2 - r * hiddenSpacing
          const z = -w / 2 + c * hiddenSpacing
          positions.push(new THREE.Vector3(layerX, y, z))
        }
      }
      return positions
    }

    // Layer Meshes
    const tempObject = new THREE.Object3D()
    const tempColor = new THREE.Color()
    const tempQuaternion = new THREE.Quaternion()
    const upVector = new THREE.Vector3(0, 1, 0)
    const highlightColor = new THREE.Color(0x4da6ff)

    interface LayerMeshInfo {
      mesh: THREE.InstancedMesh
      positions: THREE.Vector3[]
      type: 'input' | 'hidden' | 'output'
      layerIndex: number
    }
    const layerMeshes: LayerMeshInfo[] = []

    arch.forEach((count, idx) => {
      const lx = startX + idx * layerSpacing
      const pos = computePositions(idx, count, lx)
      const isOutput = idx === arch.length - 1

      if (idx === 0) {
        // Input boxes
        const geom = new THREE.BoxGeometry(inputNodeSize, inputNodeSize, inputNodeSize)
        const mat = new THREE.MeshLambertMaterial({ emissive: 0x111620 })
        const mesh = new THREE.InstancedMesh(geom, mat, count)
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)

        const colorAttr = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3)
        colorAttr.setUsage(THREE.DynamicDrawUsage)
        mesh.instanceColor = colorAttr

        pos.forEach((p, i) => {
          tempObject.position.copy(p)
          tempObject.updateMatrix()
          mesh.setMatrixAt(i, tempObject.matrix)
          mesh.setColorAt(i, tempColor.setRGB(0.06, 0.06, 0.06))
        })
        mesh.instanceMatrix.needsUpdate = true
        mesh.instanceColor.needsUpdate = true
        scene.add(mesh)
        layerMeshes.push({ mesh, positions: pos, type: 'input', layerIndex: idx })
      } else {
        // Hidden / Output spheres
        const r = isOutput ? outputNodeRadius : hiddenNodeRadius
        const geom = new THREE.SphereGeometry(r, 16, 16)
        const mat = new THREE.MeshBasicMaterial({ toneMapped: false })
        const mesh = new THREE.InstancedMesh(geom, mat, count)
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)

        const colorAttr = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3)
        colorAttr.setUsage(THREE.DynamicDrawUsage)
        mesh.instanceColor = colorAttr

        pos.forEach((p, i) => {
          tempObject.position.copy(p)
          tempObject.updateMatrix()
          mesh.setMatrixAt(i, tempObject.matrix)
          mesh.setColorAt(i, tempColor.setRGB(0.08, 0.08, 0.08))
        })
        mesh.instanceMatrix.needsUpdate = true
        mesh.instanceColor.needsUpdate = true
        scene.add(mesh)
        const layerType = isOutput ? 'output' : 'hidden'
        layerMeshes.push({ mesh, positions: pos, type: layerType, layerIndex: idx })

        if (isOutput) {
          pos.forEach((p, i) => {
            const letter = LETTERS[i] || String(i)
            const sprite = buildLetterSprite(letter)
            sprite.position.copy(p)
            sprite.position.x += 0.55
            sprite.scale.set(0.38, 0.38, 0.38)
            labelGroup.add(sprite)
            outputLabels.push(sprite)
          })
        }
      }
    })

    // Connections setup
    interface ConnectionGroup {
      mesh: THREE.InstancedMesh
      connections: { sourceIndex: number; targetIndex: number; weight: number }[]
      sourceLayer: number
      maxAbsWeight: number
    }
    const connectionGroups: ConnectionGroup[] = []

    const findImportantConnections = (
      layer: LayerMeta,
      limit: number,
      minMag: number
    ) => {
      const selected: { sourceIndex: number; targetIndex: number; weight: number }[] = []
      let maxAbs = 0
      for (let target = 0; target < layer.weights.length; target++) {
        const row = layer.weights[target]
        const cands: { sourceIndex: number; targetIndex: number; weight: number; mag: number }[] = []
        for (let source = 0; source < row.length; source++) {
          const w = row[source]
          const m = Math.abs(w)
          cands.push({ sourceIndex: source, targetIndex: target, weight: w, mag: m })
          if (m > maxAbs) maxAbs = m
        }
        cands.sort((a, b) => b.mag - a.mag)
        const take = Math.min(limit, cands.length)
        for (let i = 0; i < take; i++) {
          if (cands[i].mag < minMag) break
          selected.push({
            sourceIndex: cands[i].sourceIndex,
            targetIndex: cands[i].targetIndex,
            weight: cands[i].weight,
          })
        }
      }
      return { selected, maxAbsWeight: maxAbs }
    }

    const buildConnections = (connRadius = connectionThickness, maxPerNeuron = maxConnections, threshold = connectionThreshold) => {
      connectionGroups.forEach((g) => {
        scene.remove(g.mesh)
        g.mesh.geometry.dispose()
        if (Array.isArray(g.mesh.material)) {
          g.mesh.material.forEach((m) => m.dispose())
        } else {
          g.mesh.material.dispose()
        }
      })
      connectionGroups.length = 0

      const baseGeom = new THREE.CylinderGeometry(connRadius, connRadius, 1, 8, 1, true)
      const baseMat = new THREE.MeshBasicMaterial({ toneMapped: false })

      layers.forEach((layer, lIdx) => {
        const { selected, maxAbsWeight } = findImportantConnections(layer, maxPerNeuron, threshold)
        if (!selected.length) return

        const mesh = new THREE.InstancedMesh(baseGeom.clone(), baseMat.clone(), selected.length)
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)

        const colorAttr = new THREE.InstancedBufferAttribute(new Float32Array(selected.length * 3), 3)
        colorAttr.setUsage(THREE.DynamicDrawUsage)
        mesh.instanceColor = colorAttr

        selected.forEach((conn, instIdx) => {
          const srcPos = layerMeshes[lIdx].positions[conn.sourceIndex]
          const tgtPos = layerMeshes[lIdx + 1].positions[conn.targetIndex]
          const dir = tgtPos.clone().sub(srcPos)
          const len = dir.length()
          const mid = srcPos.clone().addScaledVector(dir, 0.5)

          tempObject.position.copy(mid)
          const quat = new THREE.Quaternion().setFromUnitVectors(upVector, dir.clone().normalize())
          tempObject.scale.set(1, len, 1)
          tempObject.quaternion.copy(quat)
          tempObject.updateMatrix()
          mesh.setMatrixAt(instIdx, tempObject.matrix)
          mesh.setColorAt(instIdx, tempColor.setRGB(0.0, 0.0, 0.0))
        })
        mesh.instanceMatrix.needsUpdate = true
        mesh.instanceColor.needsUpdate = true
        scene.add(mesh)
        connectionGroups.push({
          mesh,
          connections: selected,
          sourceLayer: lIdx,
          maxAbsWeight,
        })
      })
    }

    buildConnections()

    // Selection Specific Connection Meshes
    interface SelectionConnGroup {
      mesh: THREE.InstancedMesh
      connections: any[]
      direction: 'incoming' | 'outgoing'
    }
    const selectionConnGroups: SelectionConnGroup[] = []

    const clearSelectionConnMeshes = () => {
      selectionConnGroups.forEach((g) => {
        scene.remove(g.mesh)
        g.mesh.geometry.dispose()
        if (Array.isArray(g.mesh.material)) {
          g.mesh.material.forEach((m) => m.dispose())
        } else {
          g.mesh.material.dispose()
        }
      })
      selectionConnGroups.length = 0
    }

    // Active Selection State
    let activeSelectedNeuron: { layerIndex: number; neuronIndex: number } | null = null
    let lastActivations: any[] = []
    let lastPreActivations: any[] = []

    const collectSelectionData = (lIdx: number, nIdx: number) => {
      const tgtLayer = layerMeshes[lIdx]
      const tgtPos = tgtLayer?.positions?.[nIdx]
      const incoming: any[] = []
      const outgoing: any[] = []
      let bias = 0
      let prevSize = null
      let nextSize = null

      if (lIdx > 0 && layers[lIdx - 1]) {
        const prevMesh = layerMeshes[lIdx - 1]
        const wLayer = layers[lIdx - 1]
        const weights = wLayer.weights?.[nIdx]
        bias = wLayer.biases?.[nIdx] ?? 0
        if (weights && tgtPos) {
          prevSize = weights.length
          for (let sIdx = 0; sIdx < weights.length; sIdx++) {
            const sPos = prevMesh?.positions?.[sIdx]
            if (!sPos) continue
            incoming.push({
              sourceLayer: lIdx - 1,
              targetLayer: lIdx,
              sourceIndex: sIdx,
              targetIndex: nIdx,
              weight: weights[sIdx],
              sourcePosition: sPos,
              targetPosition: tgtPos,
            })
          }
        }
      }

      if (lIdx < layerMeshes.length - 1 && layers[lIdx]) {
        const nextMesh = layerMeshes[lIdx + 1]
        const wLayer = layers[lIdx]
        const weights = wLayer.weights ?? []
        nextSize = weights.length
        for (let tIdx = 0; tIdx < weights.length; tIdx++) {
          const row = weights[tIdx]
          const tPos = nextMesh?.positions?.[tIdx]
          if (!row || !tgtPos || !tPos) continue
          outgoing.push({
            sourceLayer: lIdx,
            targetLayer: lIdx + 1,
            sourceIndex: nIdx,
            targetIndex: tIdx,
            weight: row[nIdx],
            sourcePosition: tgtPos,
            targetPosition: tPos,
          })
        }
      }

      return { incoming, outgoing, bias, previousLayerSize: prevSize, nextLayerSize: nextSize }
    }

    const buildSelectionMeshes = (lIdx: number, nIdx: number) => {
      clearSelectionConnMeshes()
      const data = collectSelectionData(lIdx, nIdx)
      const baseGeom = new THREE.CylinderGeometry(connectionThickness * 1.6, connectionThickness * 1.6, 1, 8, 1, true)
      const connMat = new THREE.MeshBasicMaterial({ toneMapped: false })

      const createInstMesh = (conns: any[]) => {
        if (!conns.length) return null
        const m = new THREE.InstancedMesh(baseGeom.clone(), connMat.clone(), conns.length)
        m.instanceMatrix.setUsage(THREE.DynamicDrawUsage)

        const colorAttr = new THREE.InstancedBufferAttribute(new Float32Array(conns.length * 3), 3)
        colorAttr.setUsage(THREE.DynamicDrawUsage)
        m.instanceColor = colorAttr

        conns.forEach((c, i) => {
          const dir = c.targetPosition.clone().sub(c.sourcePosition)
          const len = dir.length()
          const mid = c.sourcePosition.clone().addScaledVector(dir, 0.5)
          tempObject.position.copy(mid)
          tempQuaternion.setFromUnitVectors(upVector, dir.clone().normalize())
          tempObject.quaternion.copy(tempQuaternion)
          tempObject.scale.set(1, len, 1)
          tempObject.updateMatrix()
          m.setMatrixAt(i, tempObject.matrix)
          m.setColorAt(i, tempColor.setRGB(0.15, 0.33, 0.5))
        })
        m.instanceMatrix.needsUpdate = true
        m.instanceColor.needsUpdate = true
        return m
      }

      if (data.incoming.length) {
        const m = createInstMesh(data.incoming)
        if (m) {
          scene.add(m)
          selectionConnGroups.push({ mesh: m, connections: data.incoming, direction: 'incoming' })
        }
      }
      if (data.outgoing.length) {
        const m = createInstMesh(data.outgoing)
        if (m) {
          scene.add(m)
          selectionConnGroups.push({ mesh: m, connections: data.outgoing, direction: 'outgoing' })
        }
      }

      // Update Selection Glow
      const l = layerMeshes[lIdx]
      const pos = l?.positions?.[nIdx]
      if (pos) {
        selectionGlowSprite.position.copy(pos)
        const baseSize = l.type === 'input' ? inputNodeSize : l.type === 'output' ? outputNodeRadius : hiddenNodeRadius
        const scale = Math.max(baseSize * 3.5, 0.25)
        selectionGlowSprite.scale.set(scale, scale, 1)
        selectionGlowSprite.visible = true
      }
    }

    const updateSelectionDetailAndColors = (lIdx: number, nIdx: number) => {
      const data = collectSelectionData(lIdx, nIdx)
      const act = lastActivations[lIdx]?.[nIdx] ?? 0
      const prevActs = lIdx > 0 ? lastActivations[lIdx - 1] : null
      const nextActs = lIdx < arch.length - 1 ? lastActivations[lIdx + 1] : null

      let sumContrib = 0
      const incoming = data.incoming.map((c) => {
        const srcAct = prevActs ? prevActs[c.sourceIndex] ?? 0 : 0
        const contrib = srcAct * c.weight
        sumContrib += contrib
        return {
          sourceIndex: c.sourceIndex,
          weight: c.weight,
          sourceActivation: srcAct,
          contribution: contrib,
        }
      })

      const outgoing = data.outgoing.map((c) => {
        const tgtAct = nextActs ? nextActs[c.targetIndex] ?? 0 : 0
        return {
          targetIndex: c.targetIndex,
          weight: c.weight,
          targetActivation: tgtAct,
          contribution: act * c.weight,
        }
      })

      const preAct = lIdx > 0 ? (lastPreActivations[lIdx - 1]?.[nIdx] ?? sumContrib + data.bias) : act
      const layerNames = [
        'Input Layer (28×28)',
        'Hidden Layer 1 (64)',
        'Hidden Layer 2 (32)',
        `Output Layer: Letter '${LETTERS[nIdx] || nIdx}' (26 Classes)`,
      ]

      const detail: SelectionDetail = {
        layerIndex: lIdx,
        neuronIndex: nIdx,
        layerLabel: layerNames[lIdx] || `Layer ${lIdx}`,
        activationName: lIdx === 0 ? 'Input Pixel' : lIdx === arch.length - 1 ? `Letter '${LETTERS[nIdx]}'` : 'ReLU',
        activationValue: act,
        preActivation: preAct,
        bias: lIdx > 0 ? data.bias : null,
        incoming,
        outgoing,
        previousLayerSize: data.previousLayerSize,
        nextLayerSize: data.nextLayerSize,
      }
      setSelectedDetail(detail)

      // Color selection connections in vibrant blue (matching highlight glow #4da6ff)
      const selBlue = { r: 0.302, g: 0.651, b: 1.0 } // #4da6ff
      const inGroup = selectionConnGroups.find((g) => g.direction === 'incoming')
      if (inGroup) {
        const maxVal = detail.incoming.reduce((a, b) => Math.max(a, Math.abs(b.contribution) || Math.abs(b.weight)), 0) || 1
        detail.incoming.forEach((item, idx) => {
          const v = Math.abs(item.contribution) > 1e-4 ? Math.abs(item.contribution) : Math.abs(item.weight)
          const norm = clamp(v / maxVal, 0, 1)
          const intensity = 0.45 + 0.55 * norm
          tempColor.setRGB(selBlue.r * intensity, selBlue.g * intensity, selBlue.b * intensity)
          inGroup.mesh.setColorAt(idx, tempColor)
        })
        if (inGroup.mesh.instanceColor) inGroup.mesh.instanceColor.needsUpdate = true
      }

      const outGroup = selectionConnGroups.find((g) => g.direction === 'outgoing')
      if (outGroup) {
        const maxVal = detail.outgoing.reduce((a, b) => Math.max(a, Math.abs(b.contribution) || Math.abs(b.weight)), 0) || 1
        detail.outgoing.forEach((item, idx) => {
          const v = Math.abs(item.contribution) > 1e-4 ? Math.abs(item.contribution) : Math.abs(item.weight)
          const norm = clamp(v / maxVal, 0, 1)
          const intensity = 0.45 + 0.55 * norm
          tempColor.setRGB(selBlue.r * intensity, selBlue.g * intensity, selBlue.b * intensity)
          outGroup.mesh.setColorAt(idx, tempColor)
        })
        if (outGroup.mesh.instanceColor) outGroup.mesh.instanceColor.needsUpdate = true
      }
    }

    const selectNeuron = (lIdx: number, nIdx: number) => {
      if (activeSelectedNeuron?.layerIndex === lIdx && activeSelectedNeuron?.neuronIndex === nIdx) {
        clearSelection()
        return
      }
      activeSelectedNeuron = { layerIndex: lIdx, neuronIndex: nIdx }
      connectionGroups.forEach((g) => (g.mesh.visible = false))
      buildSelectionMeshes(lIdx, nIdx)
      updateSelectionDetailAndColors(lIdx, nIdx)
      applyColors()
    }

    const clearSelection = () => {
      activeSelectedNeuron = null
      selectionGlowSprite.visible = false
      clearSelectionConnMeshes()
      connectionGroups.forEach((g) => (g.mesh.visible = true))
      setSelectedDetail(null)
      applyColors()
    }

    // Apply colors to all nodes and connections
    const applyColors = () => {
      const hasActiveDrawing = gridValuesRef.current.some((v) => v > 0.02)

      // Update Node Colors
      layerMeshes.forEach((l, lIdx) => {
        const vals = lastActivations[lIdx]
        if (!vals) return
        const isInput = l.type === 'input'
        const isOutput = l.type === 'output'
        const scale = isInput ? 1 : maxAbsValue(vals) || 1
        const dValuesMax = isOutput && hasActiveDrawing ? Math.max(...(Array.from(vals) as number[])) : 0

        for (let i = 0; i < vals.length; i++) {
          const isSel = activeSelectedNeuron && activeSelectedNeuron.layerIndex === lIdx && activeSelectedNeuron.neuronIndex === i
          if (isSel) {
            tempColor.copy(highlightColor)
          } else if (isInput) {
            const raw = gridValuesRef.current[i] || 0
            if (raw > 0.02) {
              const v = clamp(raw, 0, 1)
              tempColor.setRGB(0.9, 0.95, 1.0)
            } else {
              tempColor.setRGB(0.06, 0.06, 0.06)
            }
          } else if (isOutput) {
            if (!hasActiveDrawing) {
              tempColor.setRGB(0.1, 0.1, 0.1)
            } else {
              const p = clamp(vals[i], 0, 1)
              const isHighest = dValuesMax > 0.04 && vals[i] >= dValuesMax
              if (isHighest) {
                tempColor.setRGB(0.0, 0.96, 0.63) // Vibrant neon green
              } else {
                tempColor.setRGB(0.1 + 0.5 * p, 0.12 + 0.6 * p, 0.15 + 0.7 * p)
              }
            }
          } else {
            // Hidden nodes
            if (!hasActiveDrawing) {
              tempColor.setRGB(0.08, 0.08, 0.08)
            } else {
              const norm = clamp(vals[i] / scale, 0, 1)
              if (norm > 0.05) {
                tempColor.setRGB(0.15 + 0.75 * norm, 0.2 + 0.75 * norm, 0.3 + 0.65 * norm)
              } else {
                tempColor.setRGB(0.08, 0.08, 0.08)
              }
            }
          }
          l.mesh.setColorAt(i, tempColor)
        }
        if (l.mesh.instanceColor) l.mesh.instanceColor.needsUpdate = true
      })

      // Update Connection Colors: ALL BLACK BY DEFAULT, ONLY HIGHLIGHT ACTIVE WHILE DRAWING
      if (!activeSelectedNeuron) {
        if (!hasActiveDrawing) {
          connectionGroups.forEach((g) => {
            for (let i = 0; i < g.connections.length; i++) {
              g.mesh.setColorAt(i, tempColor.setRGB(0.0, 0.0, 0.0))
            }
            if (g.mesh.instanceColor) g.mesh.instanceColor.needsUpdate = true
          })
        } else {
          connectionGroups.forEach((g) => {
            const srcVals = lastActivations[g.sourceLayer]
            if (!srcVals) return
            const isFromInput = g.sourceLayer === 0
            const rawPixels = gridValuesRef.current

            const contribs = new Float32Array(g.connections.length)
            let maxC = 0
            g.connections.forEach((conn, i) => {
              const srcAct = isFromInput ? (rawPixels[conn.sourceIndex] || 0) : (srcVals[conn.sourceIndex] || 0)
              if (srcAct > 0.04) {
                const c = srcAct * conn.weight
                contribs[i] = c
                const m = Math.abs(c)
                if (m > maxC) maxC = m
              } else {
                contribs[i] = 0
              }
            })
            const scale = maxC > 1e-5 ? maxC : g.maxAbsWeight || 1

            g.connections.forEach((conn, i) => {
              const srcAct = isFromInput ? (rawPixels[conn.sourceIndex] || 0) : (srcVals[conn.sourceIndex] || 0)
              if (srcAct <= 0.04) {
                tempColor.setRGB(0.0, 0.0, 0.0)
              } else {
                const norm = clamp(contribs[i] / scale, -1, 1)
                const mag = Math.abs(norm)
                if (mag < 0.06) {
                  tempColor.setRGB(0.0, 0.0, 0.0)
                } else if (norm > 0) {
                  tempColor.setRGB(0.05, 0.25 + 0.75 * mag, 0.1)
                } else {
                  tempColor.setRGB(0.25 + 0.75 * mag, 0.05, 0.08)
                }
              }
              g.mesh.setColorAt(i, tempColor)
            })
            if (g.mesh.instanceColor) g.mesh.instanceColor.needsUpdate = true
          })
        }
      }
    }

    // Forward propagation function
    const propagate = (pixels: Float32Array) => {
      const isCanvasEmpty = !pixels.some((p) => p > 0.02)
      setHasDrawn(!isCanvasEmpty)

      if (isCanvasEmpty) {
        lastActivations = [
          new Float32Array(784),
          new Float32Array(64),
          new Float32Array(32),
          new Float32Array(26),
        ]
        lastPreActivations = [
          new Float32Array(64),
          new Float32Array(32),
          new Float32Array(26),
        ]
        setProbabilities(new Array(26).fill(0))
        applyColors()
        return
      }

      // Auto-center drawn strokes by center of mass before passing to MLP
      const centered = centerDrawnPixels(pixels)

      const mean = weightsData.normalization.mean
      const std = weightsData.normalization.std
      const input = new Float32Array(centered.length)
      for (let i = 0; i < centered.length; i++) {
        input[i] = (centered[i] - mean) / std
      }

      const acts: any[] = [new Float32Array(pixels)] // Layer 0: raw pixels [0, 1]
      const preActs: any[] = []
      let curr: any = input

      layers.forEach((layer) => {
        const outSize = layer.biases.length
        const lin = new Float32Array(outSize)
        for (let neuron = 0; neuron < outSize; neuron++) {
          let sum = layer.biases[neuron]
          const w = layer.weights[neuron]
          for (let s = 0; s < w.length; s++) {
            sum += w[s] * curr[s]
          }
          lin[neuron] = sum
        }
        preActs.push(lin)
        let act: Float32Array
        if (layer.activation === 'relu') {
          act = new Float32Array(outSize)
          for (let i = 0; i < outSize; i++) act[i] = lin[i] > 0 ? lin[i] : 0
        } else {
          act = Float32Array.from(softmax(lin))
        }
        acts.push(act)
        curr = act
      })

      lastActivations = acts
      lastPreActivations = preActs

      // Update probabilities from output layer
      const probs = Array.from(acts[acts.length - 1] as Float32Array)
      setProbabilities(probs)

      applyColors()
      if (activeSelectedNeuron) {
        updateSelectionDetailAndColors(activeSelectedNeuron.layerIndex, activeSelectedNeuron.neuronIndex)
      }
    }

    // Raycasting for Neuron Selection
    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    let pointerDownPos: { x: number; y: number } | null = null

    const onPointerDown = (e: MouseEvent) => {
      if (e.button === 0) pointerDownPos = { x: e.clientX, y: e.clientY }
    }

    const onPointerUp = (e: MouseEvent) => {
      if (!pointerDownPos || e.button !== 0) return
      const dx = e.clientX - pointerDownPos.x
      const dy = e.clientY - pointerDownPos.y
      pointerDownPos = null
      if (Math.hypot(dx, dy) > 4) return // was dragging orbit controls

      const rect = renderer.domElement.getBoundingClientRect()
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(pointer, camera)

      const hits: { dist: number; lIdx: number; nIdx: number }[] = []
      layerMeshes.forEach((l) => {
        const intersects = raycaster.intersectObject(l.mesh, false)
        intersects.forEach((hit) => {
          if (Number.isInteger(hit.instanceId)) {
            hits.push({ dist: hit.distance, lIdx: l.layerIndex, nIdx: hit.instanceId! })
          }
        })
      })

      hits.sort((a, b) => a.dist - b.dist)
      if (hits.length > 0) {
        selectNeuron(hits[0].lIdx, hits[0].nIdx)
      } else {
        clearSelection()
      }
    }

    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointerup', onPointerUp)

    // Window Resize
    const onResize = () => {
      const w = container.clientWidth || window.innerWidth
      const h = container.clientHeight || window.innerHeight
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    window.addEventListener('resize', onResize)

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') clearSelection()
    }
    window.addEventListener('keydown', onKeyDown)

    // Animation Loop
    let frameCount = 0
    let lastFpsUpdate = performance.now()

    const animate = (time: number) => {
      if (isDisposed) return
      controls.update()
      renderer.render(scene, camera)

      frameCount++
      if (time - lastFpsUpdate >= 500) {
        setFps(Math.round((frameCount * 1000) / (time - lastFpsUpdate)))
        frameCount = 0
        lastFpsUpdate = time
      }

      animationFrameId = requestAnimationFrame(animate)
    }
    animationFrameId = requestAnimationFrame(animate)

    // Attach controllers to ref
    visualizerRef.current = {
      propagate,
      clearSelection,
      updateConnectionSettings: (radius: number, limit: number, thresh: number) => {
        buildConnections(radius, limit, thresh)
        applyColors()
      },
    }

    // Initialize with a blank grid
    propagate(new Float32Array(784))

    return () => {
      isDisposed = true
      cancelAnimationFrame(animationFrameId)
      window.removeEventListener('resize', onResize)
      window.removeEventListener('keydown', onKeyDown)
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      controls.dispose()
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [connectionThickness, maxConnections, connectionThreshold])

  // Drawing Pad Functions
  const updateCellColors = () => {
    const vals = gridValuesRef.current
    gridCellsRef.current.forEach((cell, idx) => {
      if (!cell) return
      const v = vals[idx]
      if (v > 0.02) {
        const alpha = Math.min(1, 0.15 + v * 0.85)
        cell.style.background = `rgba(255, 255, 255, ${alpha})`
        cell.style.boxShadow = v > 0.35 ? '0 0 6px rgba(255, 255, 255, 0.6)' : 'none'
      } else {
        cell.style.background = 'transparent'
        cell.style.boxShadow = 'none'
      }
    })
  }

  const applyBrushAt = (centerR: number, centerC: number, mode: 'draw' | 'erase') => {
    const radius = mode === 'erase' ? brushSize * 1.6 : brushSize
    const strength = mode === 'erase' ? -brushStrength : brushStrength
    const softness = 0.45
    const radCeil = Math.ceil(radius)
    const vals = gridValuesRef.current
    let modified = false

    for (let r = centerR - radCeil; r <= centerR + radCeil; r++) {
      if (r < 0 || r >= 28) continue
      for (let c = centerC - radCeil; c <= centerC + radCeil; c++) {
        if (c < 0 || c >= 28) continue
        const dist = Math.hypot(r - centerR, c - centerC)
        if (dist > radius) continue
        const falloff = 1 - dist / radius
        if (falloff <= 0) continue
        const influence = Math.pow(falloff, 1 + softness * 2)
        const delta = strength * influence
        if (Math.abs(delta) < 1e-3) continue

        const idx = r * 28 + c
        const cur = vals[idx]
        const next = clamp(cur + delta, 0, 1)
        if (next !== cur) {
          vals[idx] = next
          modified = true
        }
      }
    }

    if (modified) {
      updateCellColors()
      visualizerRef.current?.propagate(gridValuesRef.current)
    }
  }

  // Pointer events for sketchpad
  const handlePointerAction = (e: React.PointerEvent) => {
    const element = document.elementFromPoint(e.clientX, e.clientY)
    if (!element) return
    const cell = element.closest('[data-cell-idx]') as HTMLElement
    if (!cell) return
    const idx = Number(cell.dataset.cellIdx)
    if (Number.isNaN(idx)) return

    const r = Math.floor(idx / 28)
    const c = idx % 28
    applyBrushAt(r, c, drawModeRef.current)
  }

  useEffect(() => {
    const onWindowPointerUp = () => {
      isDrawingRef.current = false
    }
    window.addEventListener('pointerup', onWindowPointerUp)
    return () => window.removeEventListener('pointerup', onWindowPointerUp)
  }, [])

  const resetGrid = () => {
    gridValuesRef.current.fill(0)
    setHasDrawn(false)
    updateCellColors()
    visualizerRef.current?.propagate(gridValuesRef.current)
  }

  const loadPresetLetter = (index: number) => {
    const templates = (weightsData as any).templates as number[][]
    if (templates && templates[index]) {
      const templ = templates[index]
      for (let i = 0; i < 784; i++) {
        gridValuesRef.current[i] = templ[i] || 0
      }
      setHasDrawn(true)
      updateCellColors()
      visualizerRef.current?.propagate(gridValuesRef.current)
    }
  }

  // Find top predicted letter class
  const maxProbIndex = probabilities.indexOf(Math.max(...probabilities))
  const topLetter = LETTERS[maxProbIndex] || 'A'
  const topProb = probabilities[maxProbIndex] || 0

  // Sorted predictions for ranking list
  const rankedPredictions = probabilities
    .map((p, idx) => ({ letter: LETTERS[idx], prob: p, idx }))
    .sort((a, b) => b.prob - a.prob)

  return (
    <div className="relative flex h-dvh w-full flex-col overflow-hidden bg-black text-white select-none font-sans md:block">
      {/* 3D WebGL Canvas Container */}
      <div ref={mountRef} className="relative h-[52dvh] w-full shrink-0 z-0 cursor-grab active:cursor-grabbing md:absolute md:inset-0 md:h-auto" />

      {/* Top Instructions Banner */}
      <div className="absolute top-2 md:top-4 left-1/2 -translate-x-1/2 z-10 max-w-[calc(100%-1rem)] px-3 md:px-5 py-1.5 md:py-2 rounded-full bg-black/85 backdrop-blur-md border border-white/20 shadow-2xl text-[11px] md:text-xs text-neutral-400 flex items-center gap-4 pointer-events-none">
        <div className="md:hidden whitespace-nowrap">
          <span className="text-white font-semibold">Drag</span> rotate &bull; <span className="text-white font-semibold">Pinch</span> zoom &bull; <span className="text-white font-semibold">Tap</span> neuron
        </div>
        <div className="hidden md:block">
          <span className="text-white font-semibold">Draw a letter:</span> Click & drag on grid (A–Z)
        </div>
        <div className="hidden md:block w-px h-3 bg-white/20" />
        <div className="hidden md:block">
          <span className="text-white font-semibold">3D Controls:</span> Left drag = Rotate • Right drag = Pan • Scroll = Zoom
        </div>
      </div>

      {/* Panels: scrollable column below the 3D view on mobile, floating overlays on desktop */}
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto overscroll-contain p-3 pb-6 md:contents">
      {/* Left Overlay: 28x28 Drawing SketchPad + Alphabet Presets */}
      <div className="relative z-10 flex flex-col gap-2 md:absolute md:top-16 md:left-6">
        <div className="p-4 rounded-xl bg-black/85 backdrop-blur-xl border border-white/20 shadow-[0_4px_30px_rgba(0,0,0,0.8)] flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold tracking-wider text-neutral-400 uppercase">Draw a letter</span>
            <button
              onClick={resetGrid}
              className="text-xs px-2.5 py-1 rounded-md bg-neutral-900 hover:bg-neutral-800 text-white transition border border-white/20 active:scale-95"
              title="Clear Canvas"
            >
              ↺ Reset
            </button>
          </div>

          <div className="flex gap-3">
            {/* 28x28 DOM Grid */}
            <div
              className="grid grid-cols-28 grid-rows-28 w-[min(54vw,14rem)] aspect-square shrink-0 self-start md:w-56 md:h-56 bg-black rounded-lg border border-white/20 p-0.5 cursor-crosshair touch-none select-none"
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(28, 1fr)',
                gridTemplateRows: 'repeat(28, 1fr)',
              }}
              onContextMenu={(e) => e.preventDefault()}
              onPointerDown={(e) => {
                e.preventDefault()
                isDrawingRef.current = true
                drawModeRef.current = e.button === 2 ? 'erase' : 'draw'
                handlePointerAction(e)
              }}
              onPointerMove={(e) => {
                if (!isDrawingRef.current) return
                handlePointerAction(e)
              }}
            >
              {Array.from({ length: 784 }).map((_, i) => (
                <div
                  key={i}
                  data-cell-idx={i}
                  ref={(el) => {
                    if (el) gridCellsRef.current[i] = el
                  }}
                  className="w-full h-full rounded-[0.5px] transition-colors duration-75 pointer-events-auto"
                />
              ))}
            </div>

            {/* Alphabet preset buttons A-Z (4 columns x 7 rows, NO scrollbar) */}
            <div className="grid grid-cols-4 gap-1 min-w-0 flex-1 self-stretch md:flex-none md:w-32 md:h-56 select-none overflow-hidden content-between">
              {LETTERS.map((letter, idx) => {
                const isTop = maxProbIndex === idx && hasDrawn
                const isWide = letter === 'Y' || letter === 'Z'
                return (
                  <button
                    key={letter}
                    onClick={() => loadPresetLetter(idx)}
                    className={`h-[26px] ${isWide ? 'col-span-2' : 'col-span-1'} text-xs font-bold rounded-md flex items-center justify-center transition border ${
                      isTop
                        ? 'bg-white text-black border-white shadow-[0_0_10px_rgba(255,255,255,0.7)]'
                        : 'bg-neutral-900 hover:bg-neutral-800 border-white/10 text-neutral-300 hover:text-white'
                    }`}
                    title={`Load Letter ${letter}`}
                  >
                    {letter}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Right Overlay: Prediction Chart & Network Overview */}
      <div className="relative z-10 flex w-full flex-col gap-3 md:absolute md:top-16 md:right-6 md:w-76">
        {/* Probabilities / Predictions */}
        <div className="p-4 rounded-xl bg-black/85 backdrop-blur-xl border border-white/20 shadow-[0_4px_30px_rgba(0,0,0,0.8)] flex flex-col gap-2.5">
          <div className="flex items-center justify-between pb-1 border-b border-white/10">
            <span className="text-xs font-bold tracking-wider text-neutral-400 uppercase">Letter Predictions</span>
            <span className="text-xs font-bold text-white">
              {hasDrawn ? `Top: Letter ${topLetter} (${(topProb * 100).toFixed(1)}%)` : 'Idle (Draw a letter)'}
            </span>
          </div>

          {/* Top 8 Ranked Letter Predictions */}
          <div className="flex flex-col gap-1.5 max-h-48 overflow-y-auto pr-1">
            {rankedPredictions.slice(0, 8).map(({ letter, prob, idx }) => {
              const isTop = idx === maxProbIndex && hasDrawn && prob > 0.05
              const pct = (prob * 100).toFixed(1)
              return (
                <div key={letter} className="flex items-center gap-2 text-xs">
                  <span className={`w-4 font-bold text-center ${isTop ? 'text-white' : 'text-neutral-500'}`}>
                    {letter}
                  </span>
                  <div className="flex-1 h-3.5 bg-neutral-950 rounded-full overflow-hidden p-0.5 border border-white/10 relative">
                    <div
                      className={`h-full rounded-full transition-all duration-200 ${
                        isTop
                          ? 'bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)]'
                          : 'bg-neutral-700'
                      }`}
                      style={{ width: `${Math.max(2, prob * 100)}%` }}
                    />
                  </div>
                  <span className={`w-10 text-right text-[11px] font-mono ${isTop ? 'text-white font-bold' : 'text-neutral-500'}`}>
                    {hasDrawn ? `${pct}%` : '-'}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Network Overview */}
        <div className="p-3.5 rounded-xl bg-black/85 backdrop-blur-xl border border-white/20 shadow-[0_4px_30px_rgba(0,0,0,0.8)] flex flex-col gap-2 text-xs">
          <span className="text-[11px] font-bold tracking-wider text-neutral-400 uppercase border-b border-white/10 pb-1">
            Network Architecture (Letters A–Z)
          </span>
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div>
              <span className="text-neutral-400">Input Nodes:</span> <span className="font-semibold text-white">784</span>
            </div>
            <div>
              <span className="text-neutral-400">Hidden 1:</span> <span className="font-semibold text-white">64 (ReLU)</span>
            </div>
            <div>
              <span className="text-neutral-400">Hidden 2:</span> <span className="font-semibold text-white">32 (ReLU)</span>
            </div>
            <div>
              <span className="text-neutral-400">Output:</span> <span className="font-semibold text-white">26 (A–Z)</span>
            </div>
          </div>
          <div className="text-[11px] pt-1 border-t border-white/5 flex justify-between">
            <span className="text-neutral-400">Total Weights:</span>
            <span className="font-semibold text-white">53,090 parameters</span>
          </div>
        </div>
      </div>

      </div>

      {/* Selected Neuron Detail Inspector */}
      {selectedDetail && (
        <div className="fixed inset-x-3 bottom-3 max-h-[55dvh] md:absolute md:inset-x-auto md:bottom-6 md:left-1/2 md:-translate-x-1/2 md:max-h-none z-20 md:w-[420px] p-4 rounded-xl bg-black/95 backdrop-blur-2xl border border-white/30 shadow-[0_0_30px_rgba(255,255,255,0.15)] flex flex-col gap-3 overflow-y-auto md:overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-200">
          <div className="flex items-center justify-between pb-2 border-b border-white/10">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-white shadow-[0_0_8px_white]" />
                {selectedDetail.layerLabel}
              </h3>
              <p className="text-[11px] text-neutral-400">
                Neuron #{selectedDetail.neuronIndex} ({selectedDetail.activationName})
              </p>
            </div>
            <button
              onClick={() => visualizerRef.current?.clearSelection()}
              className="text-neutral-400 hover:text-white p-1 rounded-md hover:bg-neutral-800 transition"
            >
              ✕
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs bg-neutral-950 p-2.5 rounded-lg border border-white/15">
            <div>
              <span className="text-neutral-400">Activation:</span>
              <p className="font-mono text-sm font-semibold text-white">
                {selectedDetail.activationValue !== null ? selectedDetail.activationValue.toFixed(3) : '0.000'}
              </p>
            </div>
            <div>
              <span className="text-neutral-400">Pre-Activation:</span>
              <p className="font-mono text-sm font-semibold text-white">
                {selectedDetail.preActivation !== null ? selectedDetail.preActivation.toFixed(3) : '0.000'}
              </p>
            </div>
            {selectedDetail.bias !== null && (
              <div className="col-span-2 pt-1 border-t border-white/5 flex justify-between">
                <span className="text-neutral-400">Bias:</span>
                <span className="font-mono font-semibold text-white">{selectedDetail.bias.toFixed(4)}</span>
              </div>
            )}
          </div>

          {/* Connections Summary */}
          <div className="flex flex-col gap-2 text-[11px]">
            {selectedDetail.incoming.length > 0 && (
              <div>
                <span className="text-neutral-400 font-semibold uppercase text-[10px] tracking-wider block mb-1">
                  Incoming Synapses ({selectedDetail.incoming.length})
                </span>
                <div className="flex flex-wrap gap-1 bg-neutral-950 p-1.5 rounded border border-white/10">
                  {selectedDetail.incoming
                    .slice()
                    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                    .slice(0, 8)
                    .map((c, i) => (
                    <span key={i} className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-neutral-900 border border-white/5">
                      <span className="text-neutral-500">#{c.sourceIndex}</span>{' '}
                      <span className={c.contribution >= 0 ? 'text-white' : 'text-neutral-400'}>
                        {c.contribution >= 0 ? '+' : ''}{c.contribution.toFixed(2)}
                      </span>
                    </span>
                  ))}
                  {selectedDetail.incoming.length > 8 && (
                    <span className="font-mono text-[9px] px-1.5 py-0.5 text-neutral-500">+{selectedDetail.incoming.length - 8} more</span>
                  )}
                </div>
              </div>
            )}

            {selectedDetail.outgoing.length > 0 && (
              <div>
                <span className="text-neutral-400 font-semibold uppercase text-[10px] tracking-wider block mb-1">
                  Outgoing Synapses ({selectedDetail.outgoing.length})
                </span>
                <div className="flex flex-wrap gap-1 bg-neutral-950 p-1.5 rounded border border-white/10">
                  {selectedDetail.outgoing
                    .slice()
                    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                    .slice(0, 8)
                    .map((c, i) => (
                    <span key={i} className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-neutral-900 border border-white/5">
                      <span className="text-neutral-500">{LETTERS[c.targetIndex] || c.targetIndex}</span>{' '}
                      <span className={c.contribution >= 0 ? 'text-white' : 'text-neutral-400'}>
                        {c.contribution >= 0 ? '+' : ''}{c.contribution.toFixed(2)}
                      </span>
                    </span>
                  ))}
                  {selectedDetail.outgoing.length > 8 && (
                    <span className="font-mono text-[9px] px-1.5 py-0.5 text-neutral-500">+{selectedDetail.outgoing.length - 8} more</span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Floating Bottom Action Buttons */}
      <div className="absolute top-[calc(52dvh-3rem)] left-3 md:top-auto md:bottom-6 md:left-6 z-10 flex items-center gap-3">
        <button
          onClick={() => setSettingsOpen(true)}
          className="w-10 h-10 rounded-full bg-black/85 hover:bg-neutral-900 border border-white/20 shadow-xl flex items-center justify-center text-lg text-white transition hover:scale-105 active:scale-95"
          title="Advanced Visualization Settings"
        >
          ⚙
        </button>
        <button
          onClick={() => setInfoOpen(true)}
          className="w-10 h-10 rounded-full bg-black/85 hover:bg-neutral-900 border border-white/20 shadow-xl flex items-center justify-center text-sm font-bold text-white transition hover:scale-105 active:scale-95"
          title="Information"
        >
          i
        </button>

        {showFps && (
          <div className="px-3 py-1.5 rounded-full bg-black/80 backdrop-blur-md border border-white/15 text-[11px] font-mono text-neutral-400">
            {fps} FPS
          </div>
        )}
      </div>

      {/* Settings Modal */}
      {settingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
          <div className="w-full max-w-md max-h-[90dvh] overflow-y-auto p-6 rounded-2xl bg-neutral-950 border border-white/20 shadow-2xl flex flex-col gap-5 text-sm">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="text-base font-bold text-white">Advanced Settings</h3>
              <button onClick={() => setSettingsOpen(false)} className="text-neutral-400 hover:text-white p-1">
                ✕
              </button>
            </div>

            <div className="flex flex-col gap-4">
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-neutral-400">Max Connections Per Neuron</span>
                  <span className="font-mono text-white">{maxConnections}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="64"
                  step="1"
                  value={maxConnections}
                  onChange={(e) => {
                    const v = Number(e.target.value)
                    setMaxConnections(v)
                    visualizerRef.current?.updateConnectionSettings(connectionThickness, v, connectionThreshold)
                  }}
                  className="w-full accent-white"
                />
                <p className="text-[11px] text-neutral-500 mt-1">Controls how many of the strongest weights are rendered per neuron.</p>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-neutral-400">Connection Weight Threshold</span>
                  <span className="font-mono text-white">{connectionThreshold.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="0.5"
                  step="0.005"
                  value={connectionThreshold}
                  onChange={(e) => {
                    const v = Number(e.target.value)
                    setConnectionThreshold(v)
                    visualizerRef.current?.updateConnectionSettings(connectionThickness, maxConnections, v)
                  }}
                  className="w-full accent-white"
                />
                <p className="text-[11px] text-neutral-500 mt-1">Hides connections with absolute weight below threshold.</p>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-neutral-400">Connection Line Thickness</span>
                  <span className="font-mono text-white">{connectionThickness.toFixed(4)}</span>
                </div>
                <input
                  type="range"
                  min="0.001"
                  max="0.015"
                  step="0.0005"
                  value={connectionThickness}
                  onChange={(e) => {
                    const v = Number(e.target.value)
                    setConnectionThickness(v)
                    visualizerRef.current?.updateConnectionSettings(v, maxConnections, connectionThreshold)
                  }}
                  className="w-full accent-white"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-neutral-400">Drawing Brush Size</span>
                  <span className="font-mono text-white">{brushSize.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="3.0"
                  step="0.1"
                  value={brushSize}
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                  className="w-full accent-white"
                />
              </div>
            </div>

            <button
              onClick={() => setSettingsOpen(false)}
              className="w-full py-2.5 rounded-lg bg-white hover:bg-neutral-200 text-black font-semibold text-xs transition"
            >
              Done
            </button>
          </div>
        </div>
      )}

      {/* Info Modal */}
      {infoOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
          <div className="w-full max-w-lg p-6 rounded-2xl bg-neutral-950 border border-white/20 shadow-2xl flex flex-col gap-4 text-sm max-h-[85vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="text-base font-bold text-white">3D Alphabet Neural Network Visualizer</h3>
              <button onClick={() => setInfoOpen(false)} className="text-neutral-400 hover:text-white p-1">
                ✕
              </button>
            </div>

            <div className="flex flex-col gap-3 text-xs text-neutral-300 leading-relaxed">
              <p className="text-white">
                An interactive 3D Multi-Layer Perceptron (MLP) trained to classify handwritten and drawn English alphabet letters (A through Z).
                Draw letters on the grid or click preset alphabet buttons to watch neural pathways illuminate in real time.
              </p>

              <h4 className="font-bold text-white uppercase tracking-wider text-[11px] pt-1">3D Controls:</h4>
              <ul className="list-disc list-inside space-y-1">
                <li><strong>Rotate:</strong> Click & drag anywhere in 3D space.</li>
                <li><strong>Pan:</strong> Right-click & drag.</li>
                <li><strong>Zoom:</strong> Scroll with mouse wheel.</li>
                <li><strong>Inspect:</strong> Click any neuron to isolate its incoming/outgoing synapses with radial selection glow!</li>
              </ul>

              <h4 className="font-bold text-white uppercase tracking-wider text-[11px] pt-1">Architecture:</h4>
              <ul className="list-disc list-inside space-y-1">
                <li><strong>Input Layer:</strong> 28×28 pixel grid (784 box neurons)</li>
                <li><strong>Hidden Layer 1:</strong> 64 spherical neurons (ReLU)</li>
                <li><strong>Hidden Layer 2:</strong> 32 spherical neurons (ReLU)</li>
                <li><strong>Output Layer:</strong> 26 spherical neurons + 3D Letter sprites (A–Z Softmax)</li>
              </ul>
            </div>

            <button
              onClick={() => setInfoOpen(false)}
              className="w-full py-2.5 rounded-lg bg-white hover:bg-neutral-200 text-black font-semibold text-xs transition border border-white/10 mt-2"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
