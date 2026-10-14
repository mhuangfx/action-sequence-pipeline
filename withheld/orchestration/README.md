# Nightly planner and packaging (withheld)

`plan_specs.js` picks cast, fight style, and shot templates for each night and writes job specs like the ones in `specs/`. `nightly_generate.sh` runs preflight checks, the planner, and `make_video.js` for each new spec under a lock. `emit_package.js` crops and packages finished videos. All three are wired into a publishing layer, so they are withheld.
