import * as THREE from 'three'

export type LabelEntry = {
  el: HTMLElement
  pos: THREE.Vector3
  kind: 'node' | 'layer' | 'diverge'
}

/**
 * DOM labels that follow 3D positions. They live outside the Canvas and are moved every
 * frame by <Projector /> in Scene.tsx. This replaces drei's <Html>, which mounts a separate
 * React root per label and crashes under React 19 ("unmount a root while rendering").
 */
export const labels = new Map<string, LabelEntry>()
