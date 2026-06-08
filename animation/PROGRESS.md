# Animation Project Progress

## Status: In Progress (Paused — token limit)

---

## What's Done

### 1. Python PSF generation script ✅
- `animation/scripts/generate_psf_images.py` — complete and working
- Generates 12 PNGs to `animation/public/psf/`:
  - `pupil_{widefield,dslm,spim,lattice}.png`
  - `psf_{widefield,dslm,spim,lattice}.png`
  - `xz_{widefield,dslm,spim,lattice}.png`
- Run with: `python animation/scripts/generate_psf_images.py` from `C:\Code\biobeam`

### 2. Remotion project scaffolded ✅
- `animation/package.json` — Remotion 4.0, React 18, TypeScript
- `animation/tsconfig.json`
- `animation/remotion.config.ts`
- `npm install` done in the worktree

### 3. All source files written ✅
- `src/constants.ts` — bilingual labels, beam configs, colors, FPS
- `src/index.ts` — registerRoot entry point
- `src/Root.tsx` — Composition: PupilEducation, 10050f, 30fps, 1920×1080
- `src/PupilEducation.tsx` — Series of all scenes
- `src/part1/MicroscopeSchematic.tsx` — SVG lens+ray diagram with animated rays
- `src/part1/PupilLabels.tsx` — 入瞳/出瞳/孔径光阑 annotations
- `src/part1/KSpaceIntro.tsx` — Zoom aperture → k-space disk with axes
- `src/part2/PupilMask.tsx` — SVG animated pupil for all 4 beam types
- `src/part2/FourierArrow.tsx` — Animated FT arrow with formula
- `src/part2/PSFHeatmap.tsx` — Img reveal via clipPath sweep
- `src/part2/XZHeatmap.tsx` — xz profile reveal top-down
- `src/part2/BeamScene.tsx` — Reusable pipeline: pupil → FT → PSF → xz
- `src/part2/ComparisonPanel.tsx` — 4-column comparison grid

### 4. Verified ✅
- `npx tsc --noEmit` → 0 errors
- `npx remotion compositions src/index.ts` → shows `PupilEducation  30  1920x1080  10050 (335.00 sec)`

---

## What's NOT Done Yet

### Immediate next step: Fix node_modules in git

The commit at `feature/animation-remotion` accidentally included `node_modules/`.
Need to clean it up:

```bash
cd "C:/Code/biobeam/.worktrees/animation-remotion"
git rm -r --cached animation/node_modules/
git add animation/.gitignore
git commit -m "Remove node_modules from tracking, add .gitignore"
```

The `.gitignore` file was already written to `animation/.gitignore` with:
```
node_modules/
out/
.remotion/
```

### After that: Verify studio launches

```bash
cd "C:/Code/biobeam/.worktrees/animation-remotion/animation"
npm start
# opens http://localhost:3000
```

Scrub through to verify:
- Part 1 (0–80s): lens diagram, ray animation, pupil labels, k-space zoom
- Part 2 (80–290s): each BeamScene with pupil draw → FT arrow → PSF reveal → xz reveal
- Part 3 (290–335s): comparison panel columns fade in

### Optional polish (not yet implemented)
- Lattice scene: extra sub-panel showing kpoints/sigma_phi parameter effect
- Scale bars on PSF/xz images (microns)

---

## Project Location

- **Main repo**: `C:\Code\biobeam`
- **Worktree**: `C:\Code\biobeam\.worktrees\animation-remotion`
- **Animation project**: `C:\Code\biobeam\.worktrees\animation-remotion\animation`
- **Feature branch**: `feature/animation-remotion`

## Key Commands

```bash
# Generate PSF images (run from C:\Code\biobeam)
python animation/scripts/generate_psf_images.py

# Launch Remotion studio (run from animation/)
npm start

# Test render (first 90s)
npm run render:test

# TypeScript check
npx tsc --noEmit

# List compositions
npx remotion compositions src/index.ts
```
