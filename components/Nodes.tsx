'use client'

import { useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Billboard } from '@react-three/drei'
import * as THREE from 'three'
import { ACTIVE_T, NodeInfo } from '@/data/network'
import { Scenario, activation } from '@/data/scenarios'
import { anim, useStore } from '@/lib/store'

const DIM = new THREE.Color('#2a3350')

function Ring({
  radius,
  color,
  layer,
}: {
  radius: number
  color: string
  layer: number
}) {
  const mat = useRef<THREE.MeshBasicMaterial>(null!)
  useFrame(() => {
    const r = THREE.MathUtils.clamp((anim.progress - (layer - 0.6)) / 0.6, 0, 1)
    mat.current.opacity = r * 0.9
  })
  return (
    <mesh>
      <ringGeometry args={[radius, radius + 0.035, 48]} />
      <meshBasicMaterial ref={mat} color={color} transparent opacity={0} toneMapped={false} depthWrite={false} />
    </mesh>
  )
}

export function NodeView({ node, scenarios }: { node: NodeInfo; scenarios: Scenario[] }) {
  const mesh = useRef<THREE.Mesh>(null!)
  const mat = useRef<THREE.MeshStandardMaterial>(null!)
  const cur = useRef(0)
  const [hovered, setHovered] = useState(false)
  const selectedNode = useStore((s) => s.selectedNode)
  const selectNode = useStore((s) => s.selectNode)

  const users = useMemo(
    () =>
      scenarios
        .map((s) => ({ s, a: activation(s, node.layer, node.index) }))
        .filter((u) => u.a >= ACTIVE_T),
    [scenarios, node]
  )

  const base = useMemo(() => {
    const c = new THREE.Color(0, 0, 0)
    users.forEach((u) => c.add(new THREE.Color(u.s.color)))
    if (users.length) c.multiplyScalar(1 / users.length)
    return c
  }, [users])

  const maxA = users.reduce((m, u) => Math.max(m, u.a), 0)

  useFrame((_, dt) => {
    const r = THREE.MathUtils.clamp((anim.progress - (node.layer - 0.6)) / 0.6, 0, 1)
    const target = users.length ? r * maxA : 0
    cur.current = THREE.MathUtils.damp(cur.current, target, 8, dt)
    const k = Math.min(1, cur.current * 1.4)
    mesh.current.scale.setScalar(0.14 + 0.3 * cur.current + (hovered ? 0.05 : 0))
    mat.current.emissiveIntensity = 0.15 + 2.0 * cur.current
    mat.current.color.copy(DIM).lerp(base, k)
    mat.current.emissive.copy(DIM).lerp(base, k)
  })

  return (
    <group position={node.position}>
      <mesh
        ref={mesh}
        onClick={(e) => {
          e.stopPropagation()
          selectNode(selectedNode === node.id ? null : node.id)
        }}
        onPointerOver={(e) => {
          e.stopPropagation()
          setHovered(true)
          document.body.style.cursor = 'pointer'
        }}
        onPointerOut={() => {
          setHovered(false)
          document.body.style.cursor = 'auto'
        }}
      >
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial ref={mat} color={DIM} emissive={DIM} roughness={0.4} metalness={0.1} toneMapped={false} />
      </mesh>

      {/* One ring per scenario when a node is shared, so overlap is visible. */}
      {users.length > 1 && (
        <Billboard>
          {users.map((u, i) => (
            <Ring key={u.s.id} radius={0.34 + i * 0.07} color={u.s.color} layer={node.layer} />
          ))}
        </Billboard>
      )}

      {selectedNode === node.id && (
        <Billboard>
          <mesh>
            <ringGeometry args={[0.62, 0.65, 48]} />
            <meshBasicMaterial color="#ffffff" transparent opacity={0.8} toneMapped={false} />
          </mesh>
        </Billboard>
      )}

    </group>
  )
}
