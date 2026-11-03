import { create } from 'zustand'

/**
 * Mutable animation clock, read every frame by the 3D scene (no React re-renders).
 * progress: 0 = input only, 1..4 = signal has reached that layer.
 */
export const anim = { progress: 0, target: 0 }

export const LAST_STEP = 4

type State = {
  selected: string[]
  step: number
  playing: boolean
  selectedNode: string | null
  toggleScenario: (id: string) => void
  setStep: (n: number) => void
  play: () => void
  pause: () => void
  selectNode: (id: string | null) => void
  syncStep: (n: number) => void
}

export const useStore = create<State>((set, get) => ({
  selected: ['cat'],
  step: 0,
  playing: false,
  selectedNode: null,

  toggleScenario: (id) => {
    const { selected } = get()
    if (selected.includes(id)) {
      if (selected.length > 1) set({ selected: selected.filter((s) => s !== id) })
    } else if (selected.length < 4) {
      set({ selected: [...selected, id] })
    }
  },

  setStep: (n) => {
    const v = Math.max(0, Math.min(LAST_STEP, n))
    anim.target = v
    set({ step: v, playing: false })
  },

  play: () => {
    if (anim.target >= LAST_STEP) {
      anim.target = 0
      anim.progress = 0
    }
    set({ playing: true })
  },

  pause: () => set({ playing: false }),

  selectNode: (id) => set({ selectedNode: id }),

  syncStep: (n) => set({ step: n }),
}))
