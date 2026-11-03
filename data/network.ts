export const LAYERS = [
  { name: 'Input', nodes: ['Top region', 'Center region', 'Bottom region', 'Color histogram', 'Contrast'] },
  { name: 'Features', nodes: ['Edges', 'Curves', 'Fur texture', 'Metal sheen', 'Warm tones', 'Symmetry'] },
  { name: 'Parts', nodes: ['Pointy ears', 'Floppy ears', 'Snout', 'Tail', 'Wheels', 'Whiskers'] },
  { name: 'Concepts', nodes: ['Feline', 'Canine', 'Wild', 'Domestic', 'Vehicle'] },
  { name: 'Output', nodes: ['Cat', 'Dog', 'Fox', 'Car'] },
] as const

export const LAYER_GAP = 3.6
export const NODE_GAP = 1.05
/** A node counts as "touched" by a scenario when its activation reaches this value. */
export const ACTIVE_T = 0.3

export type NodeInfo = {
  id: string
  layer: number
  index: number
  label: string
  position: [number, number, number]
}

export const layerX = (layer: number) => (layer - (LAYERS.length - 1) / 2) * LAYER_GAP

export const NODES: NodeInfo[] = LAYERS.flatMap((layer, l) =>
  layer.nodes.map((label, i) => ({
    id: `${l}-${i}`,
    layer: l,
    index: i,
    label,
    position: [
      layerX(l),
      ((layer.nodes.length - 1) / 2 - i) * NODE_GAP,
      (i % 2 === 0 ? -0.3 : 0.3) * (l % 2 === 0 ? 1 : -1),
    ] as [number, number, number],
  }))
)

export const nodeAt = (layer: number, index: number) => NODES[
  LAYERS.slice(0, layer).reduce((n, l) => n + l.nodes.length, 0) + index
]

export const nodeById = (id: string) => NODES.find((n) => n.id === id)
