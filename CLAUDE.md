# Project context for Claude Code

## What this repo is
<One paragraph: which deliverable + thrust (HTS-Quench / Mu2e-SlowSpill /
ICEBERG-DUNE-SN), and the scientific goal.>

## Stack & conventions
- Language / framework:
- Env manager (uv / conda / venv) and setup command:
- Test command:
- Lint/format command:
- Data location (and what must NOT be committed):

## Domain notes Claude should know
- Physics context, units, sign conventions, detector/controller specifics.
- Key papers live in my Obsidian vault under 03-Resources/Papers.

## How I want help
- Keep changes small and reviewable; state physics assumptions explicitly.
- Update docs/PROGRESS.md after meaningful changes (see below).
- Never commit data, checkpoints, or credentials.

## Progress logging
After a meaningful change, append a dated entry to docs/PROGRESS.md:
what changed, why, and the result/metric. I sync this into the matching
deliverable note in my vault during weekly review.
