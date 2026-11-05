# ProbeLab Design System

## 1. Atmosphere & Identity

ProbeLab feels like a field notebook upgraded into a calm scientific instrument: dark graphite surfaces, warm paper text, and one precise signal color. The signature is a live vertical “experiment rail” that makes the loop from observation to evidence visible at a glance.

## 2. Color

| Role | Token | Value | Usage |
|---|---|---|---|
| Surface / primary | --surface-primary | #101310 | App background |
| Surface / secondary | --surface-secondary | #171b17 | Panels |
| Surface / elevated | --surface-elevated | #202720 | Cards, inputs |
| Text / primary | --text-primary | #f0f1e8 | Headlines |
| Text / secondary | --text-secondary | #a8afa5 | Supporting copy |
| Text / tertiary | --text-tertiary | #727a70 | Metadata |
| Border | --border-default | #303830 | Dividers |
| Accent / primary | --accent-primary | #c8ff5a | Actions, measurement signal |
| Status / info | --status-info | #8fb6ff | Observation status |
| Status / warning | --status-warning | #f2c66d | Uncertainty |
| Status / error | --status-error | #ff8d79 | Failures |

## 3. Typography

Primary: ui-sans-serif, system-ui, -apple-system, sans-serif. Mono: ui-monospace, SFMono-Regular, Consolas, monospace.

| Level | Size | Weight | Usage |
|---|---:|---:|---|
| Display | 48px | 700 | Product title |
| H1 | 36px | 700 | Workspace title |
| H2 | 28px | 600 | Major sections |
| H3 | 20px | 600 | Cards |
| Body | 16px | 400 | Default copy |
| Small | 14px | 400 | Supporting copy |
| Caption | 12px | 600 | Labels, metadata |

## 4. Spacing & Layout

Base unit is 4px. Max width 1440px. Layout uses a 12-column grid on desktop and a single column below 920px. Breakpoints: 640px, 920px, 1280px.

## 5. Components

### Panel

Tonal surface with a 1px border, 16px radius, 24px padding. States: default, hover, error. Used for scientific work areas and summaries.

### Signal pill

Compact label with mono text, tinted border, and status dot. Used for model, status, and quality states.

### Primary action

Lime fill with graphite text, 12px radius, 48px minimum height. Focus uses a 2px lime outline offset.

## 6. Motion & Interaction

Micro interactions use 140ms ease-out. Panel entrance uses 260ms cubic-bezier(0.16, 1, 0.3, 1). Only opacity and transform are animated. Reduced motion disables non-essential motion.

## 7. Depth & Surface

Tonal-shift with restrained borders. Surfaces are separated by background value first; borders clarify interactive boundaries.

