import { ACTIVE_T, LAYERS } from './network'

export type Scenario = {
  id: string
  label: string
  color: string
  /** Activations for layers 0..3 as nodeIndex -> value (0..1). Missing = not touched. */
  acts: Record<number, number>[]
  /** Output probabilities, in the order of the Output layer nodes. */
  output: number[]
}

/**
 * Hand-authored traces. Layer indexes:
 *  L1 Features: 0 Edges 1 Curves 2 Fur 3 Metal 4 Warm 5 Symmetry
 *  L2 Parts:    0 Pointy ears 1 Floppy ears 2 Snout 3 Tail 4 Wheels 5 Whiskers
 *  L3 Concepts: 0 Feline 1 Canine 2 Wild 3 Domestic 4 Vehicle
 *  L4 Output:   0 Cat 1 Dog 2 Fox 3 Car
 */
export const SCENARIOS: Scenario[] = [
  {
    id: 'cat',
    label: 'Cat photo',
    color: '#ff9f43',
    acts: [
      { 0: 0.8, 1: 0.9, 2: 0.7, 3: 0.5, 4: 0.6 },
      { 0: 0.7, 1: 0.6, 2: 0.9, 4: 0.5, 5: 0.6 },
      { 0: 0.8, 3: 0.7, 5: 0.9 },
      { 0: 0.9, 3: 0.8 },
    ],
    output: [0.86, 0.07, 0.06, 0.01],
  },
  {
    id: 'dog',
    label: 'Dog photo',
    color: '#4da3ff',
    acts: [
      { 0: 0.8, 1: 0.9, 2: 0.7, 3: 0.6, 4: 0.5 },
      { 0: 0.7, 1: 0.5, 2: 0.9, 4: 0.6, 5: 0.5 },
      { 1: 0.8, 2: 0.8, 3: 0.6 },
      { 1: 0.9, 3: 0.8 },
    ],
    output: [0.05, 0.88, 0.06, 0.01],
  },
  {
    id: 'fox',
    label: 'Fox photo',
    color: '#ff5d8f',
    acts: [
      { 0: 0.8, 1: 0.9, 2: 0.7, 3: 0.8, 4: 0.6 },
      { 0: 0.7, 1: 0.55, 2: 0.9, 4: 0.9, 5: 0.5 },
      { 0: 0.85, 2: 0.8, 3: 0.8 },
      { 1: 0.7, 2: 0.9 },
    ],
    output: [0.04, 0.12, 0.83, 0.01],
  },
  {
    id: 'car',
    label: 'Car photo',
    color: '#5ef2b0',
    acts: [
      { 0: 0.5, 1: 0.8, 2: 0.6, 3: 0.3, 4: 0.9 },
      { 0: 0.9, 1: 0.4, 3: 0.9, 5: 0.7 },
      { 4: 0.95 },
      { 4: 0.95 },
    ],
    output: [0.01, 0.01, 0.01, 0.97],
  },
]

export const SCENARIO_BY_ID: Record<string, Scenario> = Object.fromEntries(
  SCENARIOS.map((s) => [s.id, s])
)

export const activation = (s: Scenario, layer: number, index: number): number =>
  layer < LAYERS.length - 1 ? s.acts[layer][index] ?? 0 : s.output[index] ?? 0

export const activeSet = (s: Scenario, layer: number): number[] =>
  LAYERS[layer].nodes
    .map((_, i) => i)
    .filter((i) => activation(s, layer, i) >= ACTIVE_T)

export type Route = { layer: number; from: number; to: number; strength: number }

/**
 * The route a scenario takes: for every active node, keep its 2 strongest incoming
 * connections (activation_from * activation_to). That is the "decision path".
 */
export function routeEdges(s: Scenario): Route[] {
  const routes: Route[] = []
  for (let l = 0; l < LAYERS.length - 1; l++) {
    const sources = activeSet(s, l)
    for (const to of activeSet(s, l + 1)) {
      sources
        .map((from) => ({
          layer: l,
          from,
          to,
          strength: activation(s, l, from) * activation(s, l + 1, to),
        }))
        .sort((a, b) => b.strength - a.strength)
        .slice(0, 2)
        .filter((r) => r.strength >= 0.15)
        .forEach((r) => routes.push(r))
    }
  }
  return routes
}

/** First layer at which the selected scenarios stop touching the same nodes. */
export function firstDivergence(selected: Scenario[]): number | null {
  if (selected.length < 2) return null
  for (let l = 0; l < LAYERS.length; l++) {
    const sets = new Set(selected.map((s) => activeSet(s, l).join(',')))
    if (sets.size > 1) return l
  }
  return null
}
