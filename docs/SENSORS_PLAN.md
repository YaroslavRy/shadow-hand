# Sensors plan

Status: active working plan

Goal: build tactile/contact sensing as a separate module, with a short
iteration cycle:

```text
expectation -> run -> compare -> keep/fix -> iterate
```

## Current working path

```text
[x] Stage 0a  separate module scaffold
[x] Stage 0b  tiny expectation tests
[x] Stage 0c  evaluation UI state/formatter scaffold
[ ] Stage 1   MuJoCo XML sensor wiring
[ ] Stage 1b  model/sensordata validation tests
[ ] Stage 2   live native HUD bars + totals
[ ] Stage 3   rolling plots
[ ] Stage 4   heatmap view
[ ] Stage 5   experiment logging
```

## Progress log

### Iteration 1 — local `/server` sensor UI boot

Date: 2026-08-19

Expectation:

- project env resolves `gradio`
- `uv run python app.py` starts locally
- `/server` path exposes:
  - tactile sensor status
  - finger signal table
  - linear signal plot
  - heatmap

Checks to run:

- `uv run python -m unittest tests.test_sensors_schema tests.test_sensors_runtime tests.test_sensors_aggregations tests.test_sensors_dashboard tests.test_sensors_mjcf`
- `uv run python -m py_compile app.py shadow_hand/main.py shadow_hand/sensors/__init__.py shadow_hand/sensors/schema.py shadow_hand/sensors/runtime.py shadow_hand/sensors/aggregations.py shadow_hand/sensors/dashboard.py shadow_hand/sensors/mjcf.py shadow_hand/sensors/plots.py`
- `uv run python app.py`

Observed so far:

- tests: pass
- compile sanity: pass
- app boot (first try): failed with `ModuleNotFoundError: No module named 'gradio'`
- action: `uv sync`
- env check: `gradio` now installed in `.venv`
- app boot (second try): pass
- runtime reached:
  - `Started server process`
  - `Application startup complete`
  - `Uvicorn running on http://0.0.0.0:7860`

Decision:

- keep sensor module/test work
- keep current `/server` sensor UI path
- next iteration: evaluate the live sensor widgets in-browser and then mirror the same evaluation UI into the native local MuJoCo path

### Iteration 2 — native local MuJoCo sensor HUD

Date: 2026-08-19

Expectation:

- `shadow_hand/main.py` reads real MuJoCo `sensordata`
- native MuJoCo HUD shows:
  - tactile sensor availability
  - total contact
  - per-finger bars/totals
- if sensors are missing, HUD says so explicitly instead of failing silently

Checks to run:

- `uv run python -m unittest tests.test_sensors_schema tests.test_sensors_runtime tests.test_sensors_aggregations tests.test_sensors_dashboard tests.test_sensors_mjcf`
- `uv run python -m py_compile shadow_hand/main.py shadow_hand/sensors/__init__.py shadow_hand/sensors/dashboard.py shadow_hand/sensors/mjcf.py shadow_hand/sensors/runtime.py`
- local native run after code patch: `uv run python -m shadow_hand.main --no-cv2`

Observed so far:

- native model XML already contains the first sensor sites and touch sensors
- native HUD hook already exists via `viewer_obj.set_texts(...)`
- native HUD integration code added to `shadow_hand/main.py`
- sensor tests: pass
- compile sanity: pass
- native boot under `uv run python -m shadow_hand.main --no-cv2`: fail
- blocker:
  - `RuntimeError: launch_passive requires that the Python script be run under mjpython on macOS`

Decision:

- add the smallest native HUD integration first
- keep plots/heatmaps in `/server` for now
- keep the native HUD code
- native runtime verification must use `mjpython` on macOS
- next iteration: verify the HUD under `mjpython`, then decide whether to add richer native sensor views

### Iteration 3 — native HUD row renderer fix

Date: 2026-08-19

Expectation:

- `uv run mjpython -m shadow_hand.main --no-cv2` no longer fails on `render_finger_rows(..., width=...)`
- native HUD uses a renderer compatible with MuJoCo overlay text, not the HTML-first server renderer
- regression is covered by a small test

Checks to run:

- `uv run python -m unittest tests.test_sensors_dashboard`
- `uv run python -m py_compile shadow_hand/main.py shadow_hand/sensors/__init__.py shadow_hand/sensors/dashboard.py`
- exact runtime path: `uv run mjpython -m shadow_hand.main --no-cv2`

Observed so far:

- exact failure:
  - `TypeError: render_finger_rows() got an unexpected keyword argument 'width'`
- cause:
  - native HUD reused the HTML-first `render_finger_rows` helper instead of a text/HUD-specific helper
- fix:
  - added `render_finger_rows_text(...)` for the native MuJoCo HUD
  - kept `render_finger_rows(...)` for the HTML/server path
- focused dashboard regression test: pass
- compile sanity: pass
- exact runtime path verified:
  - `uv run mjpython -m shadow_hand.main --no-cv2`
  - startup reached model load, slider wiring, and recorder setup without the previous crash

Decision:

- split native text row rendering from server HTML rendering
- keep the split renderer design
- exact `mjpython` path is now verified for startup on this machine

### Iteration 4 — native tracking visibility in HUD

Date: 2026-08-19

Expectation:

- native HUD tells us whether:
  - camera/tracker is running
  - a hand is currently detected
  - actuator targets were updated from tracking
- user can distinguish:
  - "tracking missing"
  - "tracking live but no motion"
  - "retargeting active"

Checks to run:

- `uv run python -m py_compile shadow_hand/main.py`
- exact startup path: `uv run mjpython -m shadow_hand.main --no-cv2`

Observed so far:

- current native path does not make tracking state visible enough
- "hand does not move" is ambiguous without a clear HUD status
- HUD diagnostics added:
  - tracker status
  - retarget status
- compile sanity: pass
- exact startup path re-verified:
  - `uv run mjpython -m shadow_hand.main --no-cv2`
  - startup reaches camera open, tracker thread start, model load, slider wiring, recorder setup

Decision:

- add explicit tracking/retarget HUD lines before deeper motion debugging
- keep these HUD diagnostics
- next comparison should be visual: check the tracker/retarget lines while moving a hand in frame

### Iteration 5 — native preview visibility + target activity

Date: 2026-08-19

Expectation:

- native run with preview window makes hand detection state obvious
- native HUD shows not just "targets updated" but also whether target magnitudes are changing
- user can distinguish:
  - no hand detected
  - hand detected but tiny/flat targets
  - hand detected with active target changes

Checks to run:

- `uv run python -m py_compile shadow_hand/tracking.py shadow_hand/main.py`
- exact startup path with preview enabled: `uv run mjpython -m shadow_hand.main`

Observed so far:

- without the preview window, tracking state is too hard to evaluate
- "hand does not move" still needs better target-activity visibility
- preview overlay text added:
  - `Hand detected (...)` or `No hand detected`
  - processed/dropped counters
- native HUD now also shows:
  - target peak
  - target mean
- compile sanity: pass
- exact preview-enabled startup path verified:
  - `uv run mjpython -m shadow_hand.main`
  - cv2 preview subprocess started
  - cv2 preview window opened
  - tracker thread started
  - model loaded

Decision:

- restore preview as the primary debug surface
- add preview overlay text and richer HUD target stats
- keep this visibility layer
- next comparison should use the live preview window + native HUD together

### Iteration 6 — add contact object for tactile testing

Date: 2026-08-19

Expectation:

- scene includes a simple collidable object in front of the hand
- closing the hand onto that object gives a clearer tactile test than free-space motion
- scene still loads in both native and Space asset paths

Checks to run:

- `uv run python -c "import mujoco; from shadow_hand.settings import SCENE_PATH; m=mujoco.MjModel.from_xml_path(str(SCENE_PATH)); print(m.ngeom)"`
- `uv run python -c "import mujoco, pathlib; p=pathlib.Path('space_assets/shadow_hand/my_scene.xml'); m=mujoco.MjModel.from_xml_path(str(p)); print(m.ngeom)"`

Observed so far:

- tracking now works and is visible
- sensor values are still hard to interpret without intentional contact
- tactile test block added to:
  - `assets/mujoco_menagerie/shadow_hand/my_scene.xml`
  - `space_assets/shadow_hand/my_scene.xml`
  - `public/mujoco/shadow_hand/my_scene.xml`
- scene load validation: pass
  - native scene loads
  - Space scene loads
  - browser scene loads

Decision:

- add a simple test block in front of the hand
- use it as the first tactile experiment before tuning sensor placement further
- next comparison should be live contact against the block

### Iteration 7 — native app visible grasp target

Date: 2026-08-20

Expectation:

- native app shows a clearly visible ball very close to the hand
- the ball is easier to understand than the previous block
- the native scene still loads cleanly after the object change

Checks to run:

- `uv run python -c "import mujoco; from shadow_hand.settings import SCENE_PATH; m=mujoco.MjModel.from_xml_path(str(SCENE_PATH)); print(m.ngeom)"`
- exact startup path: `uv run mjpython -m shadow_hand.main`

Observed so far:

- native app tracking works
- previous test object was not clearly visible/reachable enough
- current tactile setup is not usable yet for grasp testing
- close sphere replaced the old block in native/Space/browser scenes
- native scene load validation: pass
- exact native app startup path verified again after the scene change
- scene still reaches camera open, preview subprocess start, tracker thread start, and model load

Decision:

- replace the block with a close sphere
- verify native scene load and native startup again before asking for manual evaluation
- keep the sphere
- next comparison is visual in the native app: is the sphere clearly visible and close enough to contact?

### Iteration 8 — native app visual cleanup for tactile testing

Date: 2026-08-20

Expectation:

- the grasp target is clearly visible in the native app
- sensor sites are not visually confused with the grasp target
- HUD text is readable without overlapping blocks

Checks to run:

- `uv run python -m py_compile shadow_hand/main.py`
- `uv run python -c "import mujoco; from shadow_hand.settings import SCENE_PATH; mujoco.MjModel.from_xml_path(str(SCENE_PATH)); print('ok')"`
- exact startup path: `uv run mjpython -m shadow_hand.main`

Observed so far:

- screenshot shows no visible grasp target
- cyan site markers are visually distracting
- HUD is too dense for native evaluation

Decision:

- move the target farther in front of the palm and enlarge it
- hide sensor sites in normal rendering
- reduce/split HUD content for native use

Observed after user screenshot:

- cyan markers are the touch sensor sites, not the grasp target
- grasp target is still not visually usable
- native HUD still has too much text density

Next fix:

- make sensor sites invisible in normal use
- move the grasp ball farther out and make it larger/brighter
- remove the raw signal dump from native HUD and move tactile summary to top-right

### Iteration 9 — native app grasp target + HUD cleanup actually lands

Date: 2026-08-20

Expectation:

- native app shows a visible grasp ball in front of the palm
- tactile sensor sites are hidden in normal rendering
- native HUD has no overlapping text blocks

Checks to run:

- `uv run python -m py_compile shadow_hand/main.py`
- `uv run python -c "import mujoco; from shadow_hand.settings import SCENE_PATH; mujoco.MjModel.from_xml_path(str(SCENE_PATH)); print('ok')"`
- exact startup path: `uv run mjpython -m shadow_hand.main`

Observed so far:

- native HUD cleanup already landed in the live source
- tactile sites are already hidden in the live XML copies
- model-coordinate inspection shows the remaining issue is ball placement:
  - grasp site world pos ~= `[0.045, 0.000, 0.337]`
  - ball world pos ~= `[0.000, -0.095, 0.145]`
  - the ball is far below the actual hand

Decision:

- keep the existing HUD/site cleanup
- move only the tactile ball into the real grasp zone
- mirror the same ball change to Space/browser copies for consistency
- re-verify the exact native startup path before handing the command back

Observed after fix:

- sensor sites are now invisible in native/Space XML copies
- grasp ball now sits in the actual grasp zone instead of below the hand
- world-position sanity after the move:
  - grasp site world pos ~= `[0.045, 0.000, 0.337]`
  - ball world pos ~= `[0.040, 0.020, 0.332]`
- raw bottom-left signal dump removed from native HUD
- tactile summary moved to top-right
- compile sanity: pass
- native scene load sanity: pass
- Space scene load sanity: pass
- exact native startup path verified in this environment:
  - `uv run mjpython -m shadow_hand.main`
  - app reaches camera open, tracker thread start, model load, HUD boot, and stable loop logs

Pass / partial / fail:

- pass for the three code-level expectations in this iteration
- still needs user visual confirmation for exact ball placement in the native app

Next step:

- user checks the native app visually
- if the ball is still not easy to contact, tune only `tactile_test_ball` pose/size next

### Iteration 10 — native app sensor visibility + de-overlapped HUD

Date: 2026-08-20

Expectation:

- sensors stay visible in the native app, but are visually distinct from the ball
- grasp ball sits closer to the actual grasp zone
- HUD blocks do not overlap even with the viewer UI open

Checks to run:

- `.venv/bin/python -m py_compile shadow_hand/main.py`
- `.venv/bin/python -c "import mujoco; from shadow_hand.settings import SCENE_PATH; m=mujoco.MjModel.from_xml_path(str(SCENE_PATH)); d=mujoco.MjData(m); mujoco.mj_forward(m,d); print('grasp_site', d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'grasp_site')]); print('ball_body', d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'tactile_test_ball')])"`
- exact startup path: `UV_CACHE_DIR=.uv-cache uv run --no-sync mjpython -m shadow_hand.main`

Observed so far:

- ball is present, but visually a bit off from the intended grasp zone
- hiding sensors hurt interpretability
- top overlays still collide under the current MuJoCo viewer font/UI layout

Decision:

- restore sites with a subtler green tint instead of cyan
- place the ball using the measured grasp-site world position as the guide
- move tactile HUD to bottom-right and reduce overlay font scale

Observed after fix:

- sensors are visible again, now as small green markers instead of cyan
- measured native positions:
  - `grasp_site = [0.045, 0.000, 0.33701]`
  - `tactile_test_ball = [0.045, 0.000, 0.34000]`
- native overlay font reduced from `150` to `100`
- tactile HUD moved from top-right to bottom-right
- compile sanity: pass
- native scene load sanity: pass
- verified startup wrapper path again:
  - `UV_CACHE_DIR=.uv-cache uv run --no-sync mjpython -m shadow_hand.main`

Pass / partial / fail:

- pass for sensor visibility restoration
- pass for ball alignment by measured scene coordinates
- partial for HUD readability until user visually confirms the new layout with their viewer settings

Next step:

- user checks whether the bottom-right tactile block is now readable
- if text still feels dense, split `Calibration` and `Tracking` into separate corners next

### Iteration 11 — native app overlay removal + clearer contact readout

Date: 2026-08-20

Expectation:

- no top overlay text remains in the native app
- tactile text no longer collides with MuJoCo's built-in sensor plot
- native tactile readout makes it obvious when *any* sensor is active

Checks to run:

- `.venv/bin/python -m py_compile shadow_hand/main.py`
- `UV_CACHE_DIR=.uv-cache uv run --no-sync python -c "import mujoco; from shadow_hand.settings import SCENE_PATH; mujoco.MjModel.from_xml_path(str(SCENE_PATH)); print('ok')"`
- exact startup path: `UV_CACHE_DIR=.uv-cache uv run --no-sync mjpython -m shadow_hand.main`

Observed so far:

- user screenshot still shows top overlay text
- bottom-right tactile text now competes with MuJoCo's own `Sensor data` figure
- MuJoCo's built-in sensor figure is a single-channel plot, so squeezing can stay flat when the selected channel is not the one in contact

Decision:

- remove the top tracking/calibration overlay from the native app
- move tactile overlay to bottom-left
- show `active sensors` and `peak sensor` in the tactile readout so contact is visible even when the built-in single-channel plot is not the right sensor

Observed after fix:

- top overlay text removed from the native app
- tactile block moved to bottom-left, away from MuJoCo's built-in `Sensor data` figure
- tactile block now reports:
  - `total contact`
  - `active sensors`
  - `peak sensor`
- compile sanity: pass
- native scene load sanity: pass
- verified startup wrapper path again:
  - `UV_CACHE_DIR=.uv-cache uv run --no-sync mjpython -m shadow_hand.main`

Pass / partial / fail:

- pass for removing the top overlay collision
- pass for clarifying why a squeeze can be invisible in the single-channel MuJoCo sensor plot
- partial until user visually confirms the bottom-left placement feels clean enough

Interpretation:

- the built-in MuJoCo `Sensor data` figure is plotting one selected sensor channel, not aggregated contact
- a squeeze can stay flat there when:
  - the selected channel is not the contacted site
  - contact happens on a different fingertip/pad/palm site
  - the hand flexes without actual site contact

### Iteration 12 — separate native diagnostics window + sustained contact

Date: 2026-08-20

Expectation:

- native app no longer depends on MuJoCo overlay text for tactile diagnostics
- a separate native diagnostics window shows:
  - total contact
  - sustained/static contact estimate
  - active sensors / peak sensor
  - heatmap
  - rolling total-contact trace
  - all 20 driven hand actuators
- tactile target gains a visible thread/cord cue in the scene

Checks to run:

- `uv run python -m unittest tests.test_sensors_dashboard`
- `.venv/bin/python -m py_compile shadow_hand/main.py shadow_hand/sensors/plots.py`
- `UV_CACHE_DIR=.uv-cache uv run --no-sync python -c "import mujoco; from shadow_hand.settings import SCENE_PATH; m=mujoco.MjModel.from_xml_path(str(SCENE_PATH)); print(m.ngeom)"`
- exact startup path: `UV_CACHE_DIR=.uv-cache uv run --no-sync mjpython -m shadow_hand.main`

Observed so far:

- bottom-left MuJoCo overlay still collides visually with built-in viewer content
- built-in `Sensor data` plot is single-channel, not aggregated tactile contact
- current native path does not expose all driven actuators in one readable panel

Decision:

- move native diagnostics into a dedicated OpenCV window
- compute a sustained-contact summary from rolling total-contact history
- show all 20 driven hand actuators in that panel
- add a thin thread visual above the ball

Observed after fix:

- `tests.test_sensors_dashboard`: pass
- compile sanity for `shadow_hand/main.py` and `shadow_hand/sensors/plots.py`: pass
- native scene load sanity: pass (`ngeom = 66`)
- verified startup wrapper path again:
  - `UV_CACHE_DIR=.uv-cache uv run --no-sync mjpython -m shadow_hand.main`
- native path now has a dedicated diagnostics window instead of relying on MuJoCo overlay text
- diagnostics panel includes:
  - total contact
  - static contact (rolling mean)
  - active sensors
  - peak sensor
  - rolling total-contact trace
  - tactile heatmap
  - all 20 driven hand actuators
- tactile target now has a visible thread/anchor cue in the scene

Pass / partial / fail:

- pass for replacing overlay diagnostics with a separate native window
- pass for making sustained/static contact visible
- pass for exposing all 20 driven hand actuators in the native diagnostics UI
- partial for “full independent joint reconstruction”: current UI covers all driven actuators, but the perception-to-joint mapping is still reduced rather than a full joint-level optimizer

Interpretation:

- the native UI now covers every driven robot actuator (`20/20`)
- this is different from full independent reconstruction from MediaPipe landmarks
- to move beyond the reduced signal map, the next step is a joint-space optimization layer:
  - user landmarks
  - robot forward kinematics
  - minimize pose mismatch under joint limits/couplings

### Iteration 13 — dynamic threaded ball + scene-matched diagnostics

Date: 2026-08-21

Expectation:

- tactile target moves dynamically rather than acting like a fixed prop
- static pressure is visible even when raw touch values are spiky
- native diagnostics panel better matches the scene's dark minimal style

Checks to run:

- `UV_CACHE_DIR=.uv-cache uv run --no-sync python -m unittest tests.test_sensors_dashboard`
- `.venv/bin/python -m py_compile shadow_hand/main.py shadow_hand/sensors/plots.py`
- `UV_CACHE_DIR=.uv-cache uv run --no-sync python -c "import mujoco; from shadow_hand.settings import SCENE_PATH; m=mujoco.MjModel.from_xml_path(str(SCENE_PATH)); print(m.ngeom, m.njnt)"`
- exact startup path: `UV_CACHE_DIR=.uv-cache uv run --no-sync mjpython -m shadow_hand.main`

Observed so far:

- thread was only visual, not dynamic
- rolling mean is too weak to communicate sustained pressure clearly
- diagnostics panel still reads as debug UI rather than scene UI

Decision:

- suspend the ball from a hinge-based thread body
- restore gravity so the threaded target can behave physically
- add EMA + peak-hold pressure summaries
- simplify and restyle the diagnostics panel with the same dark/cool palette as the scene

Observed after fix:

- `tests.test_sensors_dashboard`: pass
- compile sanity for `shadow_hand/main.py` and `shadow_hand/sensors/plots.py`: pass
- native scene load sanity: pass (`ngeom = 66`, `njnt = 31`)
- verified startup wrapper path again:
  - `UV_CACHE_DIR=.uv-cache uv run --no-sync mjpython -m shadow_hand.main`
- tactile target is now suspended from a hinge-based thread body rather than being a fixed prop
- gravity restored so the threaded target can hang and swing
- diagnostics panel now shows:
  - total contact
  - static contact (EMA)
  - hold peak
  - active sensors
  - peak sensor
  - rolling total-contact trace
  - tactile map
  - all 20 driven actuators
- diagnostics panel restyled toward the same dark/cool visual language as the scene

Pass / partial / fail:

- pass for dynamic threaded tactile target
- pass for clearer static/sustained pressure visibility
- pass for moving the diagnostics toward a cleaner scene-matched visual style
- partial for full independent reconstruction: UI coverage is complete for driven actuators, but perception is still reduced-map control rather than a joint-space optimizer

Next step:

- visually evaluate the new diagnostics window and threaded ball behavior
- if the style still feels too heavy, simplify the panel typography/layout further
- then start the joint-space optimization layer for fuller reconstruction

## Module split

```text
shadow_hand/sensors/
  schema.py        # canonical names, finger groups, region groups
  runtime.py       # data.sensordata -> structured snapshot
  aggregations.py  # finger totals, region totals, normalization
  dashboard.py     # evaluation UI state + HTML rendering
  mjcf.py          # XML sensor declarations / validation
  plots.py         # later: rolling linear plots
  heatmap.py       # later: hand-region heatmaps
  logging.py       # later: csv/json/npz experiment logging
  experiments.py   # later: repeatable touch / grasp protocols
```

## Current iteration

- [x] Stage 0 pure-Python schema/runtime/aggregation tests
- [x] Separate module boundary for sensor logic
- [ ] Stage 1 MuJoCo sensor declarations
- [ ] Stage 1 model validation tests
- [ ] Evaluation UI in app
- [ ] Native local viewer overlay
- [ ] Stage 2 contact behavior experiments
- [ ] Stage 3 logging and comparison tools

## This slice

```text
expectation tests
  -> MuJoCo sensor names/count validation
  -> simple evaluation UI (bars + total + symbolic heatmap)
  -> inspect current hand motion against live values
```

## Small expectation tests first

### Stage 0 — pure-Python tests

- sensor names are unique and stable
- runtime rejects wrong-length vectors
- runtime maps flat vectors to named values
- finger totals are deterministic
- region totals are heatmap-ready
- normalization is bounded to `[0, 1]`

### Stage 1 — MuJoCo wiring tests

- model loads with sensor declarations
- expected sensor count matches schema
- sensor names in MuJoCo match schema
- `data.sensordata.shape[0]` matches expected length
- dashboard history window stays bounded
- dashboard region ordering stays stable

### Stage 2 — contact behavior tests

- no contact => near-zero baseline
- single fingertip press => target sensor rises
- release => sensor decays back
- palm contact affects palm channels more than fingertip channels

### Stage 3 — experiment / visualization tests

- rolling plot buffer keeps fixed window
- heatmap output preserves region ordering
- logged run contains metadata + aligned sensor samples

## Evaluation UI, first slice

Keep the first UI simple and local:

- per-finger bar rows
- palm total
- total contact scalar
- symbolic heatmap rows by region

First rendering target:

- native MuJoCo HUD in `shadow_hand/main.py`

Why:

- zero extra infra
- visible during teleop
- easiest debugging loop while sensor wiring is still changing

## First sensor layout

Start simple and visible:

- thumb_tip
- index_tip
- middle_tip
- ring_tip
- pinky_tip
- thumb_pad
- index_pad
- middle_pad
- ring_pad
- pinky_pad
- palm_radial
- palm_center
- palm_ulnar

This is enough to get:

- line plots per finger
- global contact trend
- first hand heatmap

## Recommendation

Do sensor development in native local MuJoCo first.
Server/browser are useful later for display, but not required for the core
sensing loop.

## UI for evaluation

Keep the first UI intentionally small:

```text
[ per-finger bars ]
[ total contact ]
[ symbolic hand heatmap ]
[ short rolling signal trace ]
```

Meaning:

- bars answer: which finger is active?
- total answers: is there any contact at all?
- heatmap answers: where is the contact concentrated?
- rolling trace answers: when did contact rise or decay?

## Chosen path

Current implementation path for the PoC:

```text
right_hand.xml touch sites
    ->
MuJoCo touch sensors
    ->
shadow_hand.sensors.runtime
    ->
dashboard state
    ->
evaluation UI
       - per-finger bars
       - rolling total-contact plot
       - symbolic hand heatmap
```

Why this path:

- it keeps the first loop honest: real MuJoCo `sensordata`, not proxy values
- it gives immediate visual feedback during debugging
- it stays modular, so later native and browser views can reuse the same
  sensor pipeline

## First implementation slice

Stage A:

- add 13 touch sites and 13 touch sensors to `right_hand.xml`
- keep names identical to the schema
- validate model wiring in tests

Stage B:

- add sensor discovery / reading helpers
- fail clearly if sensor names or dimensions drift

Stage C:

- surface a tiny evaluation panel in the UI
- bars, rolling plot, heatmap
- if sensors are missing, say so explicitly
