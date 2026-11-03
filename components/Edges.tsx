'use client'

import { useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { LAYERS, nodeAt } from '@/data/network'
import { Scenario, routeEdges } from '@/data/scenarios'
import { anim } from '@/lib/store'

/** Faint full connectivity: every node to every node in the next layer. */
export function BaseEdges() {
  const geometry = useMemo(() => {
    const pts: number[] = []
    for (let l = 0; l < LAYERS.length - 1; l++) {
      for (let a = 0; a < LAYERS[l].nodes.length; a++) {
        for (let b = 0; b < LAYERS[l + 1].nodes.length; b++) {
          pts.push(...nodeAt(l, a).position, ...nodeAt(l + 1, b).position)
        }
      }
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
    return g
  }, [])

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#8aa0c8" transparent opacity={0.07} depthWrite={false} />
    </lineSegments>
  )
}

const vertexShader = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vN;
  void main() {
    vUv = uv;
    vN = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

/**
 * A glass tube that fills with "hot water" in the scenario's color as the signal arrives,
 * while white light keeps flowing through the filled part, from the source node to the target.
 *  - uv.y runs 0 -> 1 from source to target
 *  - uReveal is how far along the tube the signal has travelled
 */
const fragmentShader = /* glsl */ `
  uniform vec3 uColor;
  uniform float uReveal;
  uniform float uTime;
  uniform float uStrength;
  varying vec2 vUv;
  varying vec3 vN;

  void main() {
    float t = vUv.y;
    float facing = abs(normalize(vN).z);   // 1 at the tube's centre line, 0 at its silhouette
    float rim = 1.0 - facing;

    // Empty glass pipe
    vec3 emptyCol = uColor * (0.15 + 0.5 * rim);
    float emptyA = 0.08 + 0.25 * rim;

    // Comet-shaped white light bands flowing toward the target
    float d = fract(t * 1.6 - uTime * 0.55);
    float band = pow(d, 7.0) * (1.0 - smoothstep(0.96, 1.0, d));
    float core = pow(facing, 1.6);
    float light = band * core * (0.8 + 0.9 * uStrength);

    // Bright front while the signal is still travelling
    float head = exp(-pow((uReveal - t) * 14.0, 2.0)) * (1.0 - step(0.999, uReveal)) * core;

    // Tinted water filling the tube
    float water = 0.22 + 0.2 * uStrength;
    vec3 litCol = uColor * water * (0.5 + rim) + vec3(1.0) * (light + head * 1.6);
    float litA = clamp(0.3 + water + light + head, 0.0, 1.0);

    float filled = step(t, uReveal);
    gl_FragColor = vec4(mix(emptyCol, litCol, filled), mix(emptyA, litA, filled));
  }
`

function RouteTube({
  layer,
  from,
  to,
  strength,
  color,
  yOffset,
}: {
  layer: number
  from: number
  to: number
  strength: number
  color: string
  yOffset: number
}) {
  const { geometry, position, quaternion } = useMemo(() => {
    const pa = nodeAt(layer, from).position
    const pb = nodeAt(layer + 1, to).position
    const a = new THREE.Vector3(pa[0], pa[1] + yOffset, pa[2])
    const b = new THREE.Vector3(pb[0], pb[1] + yOffset, pb[2])
    const dir = b.clone().sub(a)
    const length = dir.length()
    const radius = 0.045 + 0.035 * strength
    return {
      geometry: new THREE.CylinderGeometry(radius, radius, length, 16, 1, true),
      // Cylinder's local +y becomes the a -> b direction, so uv.y runs from source to target.
      quaternion: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize()),
      position: a.add(b).multiplyScalar(0.5),
    }
  }, [layer, from, to, strength, yOffset])

  const uniforms = useMemo(
    () => ({
      uColor: { value: new THREE.Color(color).convertLinearToSRGB() },
      uReveal: { value: 0 },
      uTime: { value: 0 },
      uStrength: { value: strength },
    }),
    [color, strength]
  )

  useFrame(({ clock }) => {
    uniforms.uReveal.value = THREE.MathUtils.clamp(anim.progress - layer, 0, 1)
    uniforms.uTime.value = clock.elapsedTime
  })

  return (
    <mesh geometry={geometry} position={position} quaternion={quaternion}>
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  )
}

export function ScenarioEdges({
  scenario,
  index,
  total,
}: {
  scenario: Scenario
  index: number
  total: number
}) {
  const routes = useMemo(() => routeEdges(scenario), [scenario])
  // Fan scenarios out slightly so shared tubes sit side by side instead of overlapping.
  const yOffset = (index - (total - 1) / 2) * 0.09
  return (
    <>
      {routes.map((r) => (
        <RouteTube
          key={`${r.layer}-${r.from}-${r.to}`}
          layer={r.layer}
          from={r.from}
          to={r.to}
          strength={r.strength}
          color={scenario.color}
          yOffset={yOffset}
        />
      ))}
    </>
  )
}
