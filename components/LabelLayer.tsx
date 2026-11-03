'use client'

import { useMemo } from 'react'
import * as THREE from 'three'
import { ACTIVE_T, LAYERS, NODES, NODE_GAP, layerX, LAYER_GAP } from '@/data/network'
import { SCENARIO_BY_ID, activation, firstDivergence } from '@/data/scenarios'
import { labels, LabelEntry } from '@/lib/labels'
import { useStore } from '@/lib/store'

function Label({
  id,
  position,
  kind,
  className,
  layer,
  children,
}: {
  id: string
  position: [number, number, number]
  kind: LabelEntry['kind']
  className: string
  layer?: number
  children: React.ReactNode
}) {
  const pos = useMemo(() => new THREE.Vector3(...position), [position[0], position[1], position[2]]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div
      className={'ov ' + className}
      data-layer={layer}
      ref={(el) => {
        if (el) labels.set(id, { el, pos, kind })
        else labels.delete(id)
      }}
    >
      {children}
    </div>
  )
}

export default function LabelLayer() {
  const selected = useStore((s) => s.selected)
  const scenarios = useMemo(() => selected.map((id) => SCENARIO_BY_ID[id]), [selected])
  const div = useMemo(() => firstDivergence(scenarios), [scenarios])
  const top = ((Math.max(...LAYERS.map((l) => l.nodes.length)) - 1) / 2) * NODE_GAP + 1

  return (
    <div className="labels">
      {LAYERS.map((l, i) => (
        <Label key={l.name} id={`layer-${i}`} position={[layerX(i), top, 0]} kind="layer" className="layer-title">
          <span>{String(i).padStart(2, '0')}</span>
          {l.name}
        </Label>
      ))}

      {NODES.map((n) => {
        const on = scenarios.some((s) => activation(s, n.layer, n.index) >= ACTIVE_T)
        return (
          <Label
            key={n.id}
            id={n.id}
            position={[n.position[0], n.position[1] - 0.62, n.position[2]]}
            kind="node"
            className={'node-label' + (on ? ' on' : '')}
          >
            {n.label}
          </Label>
        )
      })}

      {div !== null && (
        <Label
          id="diverge"
          position={[layerX(div) - LAYER_GAP / 2, -3.6, 0]}
          kind="diverge"
          layer={div}
          className="diverge-label"
        >
          paths diverge → {LAYERS[div].name}
        </Label>
      )}
    </div>
  )
}
