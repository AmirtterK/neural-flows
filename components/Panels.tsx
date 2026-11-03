'use client'

import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import { Toggle } from '@/components/ui/toggle'
import { LAYERS, nodeById } from '@/data/network'
import { SCENARIOS, SCENARIO_BY_ID, activation, firstDivergence } from '@/data/scenarios'
import { LAST_STEP, useStore } from '@/lib/store'

const glass = 'bg-card/70 backdrop-blur-xl ring-1 ring-white/10'
const barColor =
  '[&_[data-slot=progress-indicator]]:bg-[var(--c)] [&_[data-slot=progress-indicator]]:duration-700 [&_[data-slot=progress-track]]:h-1.5'

export function ScenarioPanel() {
  const selected = useStore((s) => s.selected)
  const toggle = useStore((s) => s.toggleScenario)
  const div = firstDivergence(selected.map((id) => SCENARIO_BY_ID[id]))

  return (
    <Card className={`absolute left-4 top-4 z-10 w-72 ${glass}`}>
      <CardHeader>
        <CardTitle className="text-xl">Neural Flow</CardTitle>
        <CardDescription>
          Pick one input to follow its route, or select several to see where their paths share nodes and where they split.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap gap-2">
          {SCENARIOS.map((s) => {
            const on = selected.includes(s.id)
            return (
              <Toggle
                key={s.id}
                variant="outline"
                pressed={on}
                onPressedChange={() => toggle(s.id)}
                className="gap-2 rounded-full px-3"
                style={on ? { borderColor: s.color, background: `${s.color}2a` } : undefined}
              >
                <span
                  className="size-2 rounded-full"
                  style={{ background: s.color, opacity: on ? 1 : 0.4, boxShadow: on ? `0 0 8px ${s.color}` : 'none' }}
                />
                {s.label}
              </Toggle>
            )
          })}
        </div>

        {selected.length > 1 && (
          <p className="rounded-lg bg-white/5 px-3 py-2 text-sm leading-snug">
            {div === null
              ? 'These inputs take identical routes.'
              : `Shared route up to the ${LAYERS[Math.max(0, div - 1)].name} layer. They split at ${LAYERS[div].name}.`}
          </p>
        )}
        <p className="text-xs text-muted-foreground">Click a node to inspect it. Drag to orbit, scroll to zoom.</p>
      </CardContent>
    </Card>
  )
}

export function SidePanel() {
  const selected = useStore((s) => s.selected)
  const step = useStore((s) => s.step)
  const selectedNode = useStore((s) => s.selectedNode)
  const scenarios = selected.map((id) => SCENARIO_BY_ID[id])
  const node = selectedNode ? nodeById(selectedNode) : null
  const outLabels = LAYERS[LAST_STEP].nodes
  const done = step >= LAST_STEP

  return (
    <Card className={`absolute right-4 top-4 z-10 hidden max-h-[calc(100vh-9rem)] w-80 overflow-y-auto xl:flex ${glass}`}>
      <CardHeader>
        <CardTitle className="text-xs uppercase tracking-widest text-muted-foreground">Output</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {scenarios.map((s) => {
          const top = s.output.indexOf(Math.max(...s.output))
          return (
            <div
              key={s.id}
              className="rounded-lg border-l-[3px] bg-white/5 p-3"
              style={{ borderColor: s.color, ['--c' as string]: s.color }}
            >
              <div className="mb-2 flex items-center justify-between text-sm">
                {s.label}
                {done && <Badge variant="secondary">{outLabels[top]}</Badge>}
              </div>
              {outLabels.map((label, i) => (
                <div key={label} className="my-1.5 grid grid-cols-[48px_1fr_36px] items-center gap-2 text-xs">
                  <span className={i === top && done ? 'text-foreground' : 'text-muted-foreground'}>{label}</span>
                  <Progress value={done ? s.output[i] * 100 : 0} className={barColor} />
                  <em className="text-right tabular-nums not-italic text-muted-foreground">
                    {done ? `${Math.round(s.output[i] * 100)}%` : '–'}
                  </em>
                </div>
              ))}
            </div>
          )
        })}

        <Separator className="my-1" />

        <div className="text-xs uppercase tracking-widest text-muted-foreground">Inspector</div>
        {node ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between text-base">
              {node.label}
              <Badge variant="outline">{LAYERS[node.layer].name}</Badge>
            </div>
            {scenarios.map((s) => {
              const a = activation(s, node.layer, node.index)
              return (
                <div
                  key={s.id}
                  className="grid grid-cols-[48px_1fr_36px] items-center gap-2 text-xs"
                  style={{ ['--c' as string]: s.color }}
                >
                  <span className="text-muted-foreground">{s.label.split(' ')[0]}</span>
                  <Progress value={a * 100} className={barColor} />
                  <em className="text-right tabular-nums not-italic text-muted-foreground">{a ? a.toFixed(2) : 'idle'}</em>
                </div>
              )
            })}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">Select a node to see how strongly each input drives it.</p>
        )}
      </CardContent>
    </Card>
  )
}

export function Controls() {
  const step = useStore((s) => s.step)
  const playing = useStore((s) => s.playing)
  const { play, pause, setStep } = useStore.getState()

  return (
    <div
      className={`absolute bottom-6 left-1/2 z-10 flex -translate-x-1/2 items-center gap-3 rounded-2xl p-3 ${glass}`}
    >
      <Button variant="outline" size="icon" onClick={() => setStep(step - 1)} disabled={step <= 0} aria-label="Previous layer">
        <ChevronLeft />
      </Button>
      <Button className="min-w-24" onClick={() => (playing ? pause() : play())}>
        {playing ? <Pause /> : step >= LAST_STEP ? <RotateCcw /> : <Play />}
        {playing ? 'Pause' : step >= LAST_STEP ? 'Replay' : 'Play'}
      </Button>
      <Button variant="outline" size="icon" onClick={() => setStep(step + 1)} disabled={step >= LAST_STEP} aria-label="Next layer">
        <ChevronRight />
      </Button>

      <div className="ml-2 w-72 sm:w-80">
        <Slider
          min={0}
          max={LAST_STEP}
          step={1}
          value={[step]}
          onValueChange={(v) => setStep(Array.isArray(v) ? v[0] : v)}
        />
        <div className="mt-2 flex justify-between text-[11px] text-muted-foreground">
          {LAYERS.map((l, i) => (
            <button
              key={l.name}
              className={`cursor-pointer transition-colors hover:text-foreground ${i <= step ? 'text-foreground' : ''}`}
              onClick={() => setStep(i)}
            >
              {l.name}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
