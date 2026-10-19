# NexusPilot showcase website

This directory contains the static GitHub Pages showcase for NexusPilot. It has no external frontend dependencies and uses relative asset paths so it works at `https://jewelroam.github.io/NexusPilot/`.

## Local preview

```bash
npm run build
npm run dev
```

Open `http://localhost:4173/`.

## Published media

`media/` contains selected, compressed demonstration assets from the final project materials:

- `carla-demo.mp4`: a 16-second CARLA obstacle-avoidance clip
- `hardware-demo.mp4`: a 12-second physical-platform trial
- selected stills for APF planning, YOLO perception, V2V vehicle views, the Raspberry Pi platform and an indoor avoidance trial

The source recordings and final report are kept outside this public directory. Unrelated project records, credentials, terminal logs and IDE chat content are not published. The V2V still is cropped to the two onboard vehicle windows.

## Build

`npm run build` copies the static page and approved media into `dist/`. The GitHub Pages workflow publishes that directory from the repository's default branch.
