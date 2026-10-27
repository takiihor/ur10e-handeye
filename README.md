# ur10e-handeye

Eye-in-hand camera calibration for a **UR10e** with a wrist-mounted **Intel RealSense D435i**,
using the A4 ChArUco calibration board from [charuco-generator](../charuco-generator) and the
detection and pose code from [charuco-realsense](../charuco-realsense). The robot is driven over
RTDE with `ur_rtde`.

The result is the camera pose in the robot **flange** frame (`camera_in_flange`), plus the board
pose in the robot base frame. With it, anything the camera locates can be expressed in robot
coordinates:

```
p_base = T_base_flange @ camera_in_flange @ p_camera
```

> **Status.** Verified end to end in simulation (simulated UR10e kinematics plus a rendered camera
> image through the real detector) and against a fake `ur_rtde`. It has **not yet been run on the
> physical robot or URSim**. Do the first run with `--dry-run`, then at reduced speed with a hand
> on the emergency stop.

## Setup

```sh
./start.sh --help        # creates .venv, installs pinned packages and the sibling repos
```

`../charuco-generator` and `../charuco-realsense` must sit next to this repository. They are
installed editable with `--no-deps`, as charuco-realsense's README describes, because their OpenCV
pins conflict.

Print and mount the board:

```sh
../charuco-generator/generate_calibration.sh      # 5 x 7 squares of 30 mm, DICT_6X6_250, IDs 0-16
```

Print `output/calibration/calibration_charuco.pdf` at **Actual size / 100%** and glue it to a
flat, rigid plate. Measure both scale bars: a print that is 1% too large makes every calibrated
translation 1% wrong, with no visible symptom in any metric below. Fix the plate in the cell so
that it cannot move during calibration.

On the robot: power on, release the brakes and enable **Remote Control**. Connecting RTDE control
uploads ur_rtde's control script, which stops any running program. Only one application may own
RTDE control: disconnect Robot Bench or other RTDE clients first.

## Workflow

```sh
# 1. Read-only: robot state, and whether the camera sees the board well enough
./start.sh check --robot-ip 192.168.1.10

# 2. Jog the robot so the camera sees the whole board from 0.3-0.6 m, roughly centred.
#    Plan and check every view without moving:
./start.sh collect --robot-ip 192.168.1.10 --dry-run

# 3. Collect (asks for confirmation before the probe moves and before the views)
./start.sh collect --robot-ip 192.168.1.10 --joint-speed 0.1 --output sessions/cal-01

# 4. Solve offline (no robot or camera needed); writes sessions/cal-01/calibration.json
./start.sh solve --session sessions/cal-01

# 5. Verify on a separately collected session (or a replay of the same views)
./start.sh collect --robot-ip 192.168.1.10 --mode replay --replay sessions/cal-01 --output sessions/check-01
./start.sh verify --calibration sessions/cal-01/calibration.json --session sessions/check-01

# 6. Use it: a board's pose in the robot base frame (read-only)
./start.sh locate --robot-ip 192.168.1.10 --calibration sessions/cal-01/calibration.json \
    --manifest ../charuco-generator/output/board_manifest.json --board front \
    --active-tcp TX TY TZ RX RY RZ
```

### What `collect` does (auto mode)

1. Captures the start view. Each capture averages 5 frames with the robot at rest and rejects the
   view if the corners jitter (mount still vibrating), if quality gates fail, or if the joints
   moved during the capture.
2. **Probe**: three joint-space moves to flange endpoints translated 30 mm along tool X, Y and Z, returning to the start
   after each. The board's apparent motion in the camera measures the camera's orientation in the
   flange, so no hand-entered mounting guess is needed.
3. Plans 14 views that rotate the camera about the board centre: a ring of 25° tilts in eight
   directions, an inner ring with larger rolls, two pure rolls, and near and far distances. Each
   view is IK-solved from the start joints (one arm configuration throughout). Each is checked
   against the controller's safety limits, the per-move joint-step limit (75°) and the travel
   limit (450 mm). Views are then ordered nearest-first, routing through the start pose when a
   direct move would be too large. The complete plan is shown and must be confirmed.
4. After 6 good samples, the remaining views are re-planned around an interim calibration.
   The updated plan requires a separate operator confirmation before motion resumes.
5. Returns to the start pose.

Every move is a `moveJ` at 0.35 rad/s by default (`--joint-speed`, at most 1.05 rad/s). After each
move it waits for the controller to report steady, plus `--settle-s`. Motion is asynchronous
with bounded feedback polling, allowing Ctrl+C to issue `stopJ` during a move. **Any robot error or Ctrl+C
stops the arm where it is; the software never makes a recovery motion on its own.**

Other modes:

- `--mode manual [--freedrive]`: you position the robot (pendant jog, or hand-guiding with
  `--freedrive`). Press Enter to capture and `q` to finish. Freedrive is released for each
  capture, so the arm is held still.
- `--mode replay --replay SESSION`: repeat a previous session's views, for periodic checks.
  Every saved joint target and return pose is checked before moving; `--dry-run` checks only.
  `--dry-run --mode manual` is refused before any connection, including freedrive.

Everything is recorded in the **flange** frame through the controller's calibrated kinematics
(`getForwardKinematics(q, [0]*6)`), never the active TCP. Editing the TCP in PolyScope therefore
cannot silently invalidate the calibration.

## The solver

1. OpenCV's five closed-form hand-eye methods (Tsai, Park, Horaud, Andreff, Daniilidis) and Shah's
   joint robot-world/hand-eye method. Each is reported, and the best one seeds step 2.
2. Robust Levenberg-Marquardt on the **corner reprojection error** over the camera-in-flange and
   board-in-base transforms together. This is the maximum-likelihood problem for pixel noise, and
   it works from raw corners rather than per-image PnP poses. A Huber loss limits the influence of
   any single bad corner.

## Reading the result

`solve` prints, and `calibration.json` records:

| Metric | Meaning | Typical good value |
| --- | --- | --- |
| Reprojection RMSE | Corner fit across all views | < 0.5 px |
| Board-in-base consistency | Spread of the board pose implied by each view | < 0.5 mm |
| Leave-one-out | Refit without each view, then predict it: out-of-sample error near the calibration working distance | < 1 mm |
| 1-sigma camera translation | Uncertainty of the camera mount: pixel noise (from the fit) combined with assumed robot pose noise (Monte Carlo, `--robot-sigma-mm 0.05 --robot-sigma-deg 0.01`) | < 0.5 mm |
| Rotation diversity | Largest rotation between views and the spread of rotation axes; near 0 means the camera translation is unobservable | > 30°, > 0.15 |

The **verdict** is `good`, `acceptable` or `poor`, with explicit warnings. `solve` exits 1 for
`poor`, and consumers should refuse such a calibration. Leave-one-out is the best check of
predictions near the calibration views. The 1-sigma figure is the honest number for the mount
itself: in simulation, leave-one-out reads about 0.08 mm while the true mount error is 0.1-0.3 mm.

Use `--exclude N ...` to drop a view the warnings point to, and compare. Sessions are never
overwritten, so different choices can be re-solved on identical data.

### Result file (schema 1)

The contract shared with the pick-and-place software. Units are metres and UR rotation vectors;
`*_mm` / `*_deg` fields are for reading.

- `camera_in_flange`: `matrix` (4x4), `pose_ur`, `xyz_mm`, `rpy_deg`, `quaternion_xyzw`. Entered as
  a PolyScope TCP, `pose_ur` puts the TCP at the camera optical centre (X right, Y down, Z forward).
- `camera_in_tcp`: the same camera relative to the TCP active during calibration, plus that
  `tcp_offset`. For read-only consumers that only have `getActualTCPPose()`: valid only while that
  TCP is active.
- `board_in_base`: the calibration board in the base frame (valid while the board stays put).
- `intrinsics`, `camera`, `robot`, `board`, `verdict`, `warnings`, `metrics`, `methods`, `conventions`.

## Simulation

```sh
./start.sh simulate                 # full pipeline against ground truth; exit 0 if within 1 mm / 0.2 deg
./start.sh simulate --seed 4 --robot-noise-mm 0.05
```

The simulator uses nominal UR10e DH kinematics with damped least-squares IK and the same planning
checks as the real robot. It renders the generated board through a distorted pinhole model
(per-pixel remap) and runs it through the real charuco-realsense detector. Across seeds the
recovered camera mount is within 0.09-0.32 mm and 0.05° of the truth.

## Detection note (affects charuco-realsense)

Measured against exact renders, OpenCV 4.11's `CharucoDetector.detectBoard` returns ChArUco corners
offset by **about +0.5 px in x and y**; its ArUco marker corners are unbiased. One extra
`cornerSubPix` pass removes this systematic offset, and this package applies it
(`vision.refine_corners`). Uncorrected, it shifts every board pose by about 0.2 mm at 0.4 m. A test
pins the behaviour, so an OpenCV upgrade that fixes it will be noticed. charuco-realsense's live
poses now use the same image-based correction by default. The calibration observer opts out of
the detector's pass and calls the shared refinement helper itself, so each frame is refined once.

## Limits

- Calibration accuracy cannot exceed the robot's own pose accuracy at the views used, or the
  print's dimensional accuracy. Validate the whole chain (robot, camera, calibration, application)
  against the placement tolerance with independent measurements before production use.
- RealSense factory intrinsics are used unless `--intrinsics FILE` is given; the hand-eye fit does
  not re-estimate intrinsics.
- Inverse Brown-Conrady and fisheye intrinsics are rejected by charuco-realsense rather than
  approximated.

## Tests

```sh
.venv/bin/python -m pytest -q
```

They cover transform conventions (including the `cv2.Rodrigues` small-angle trap), UR10e
kinematics, real-robot safety refusals against a fake `ur_rtde` (protective stop, unsafe pose, no
IK, IK/FK mismatch, configuration flips, rejected moves, read-only mode), simulated detection
accuracy and the OpenCV corner offset, vibration rejection, the solver (exact recovery for every
method, robust loss, degenerate rotations, uncertainty), collection safety (dry run, operator
refusal, faults that stop without recovery motion, manual freedrive, replay), the session and
result formats, and the CLI end to end.


Read-only `locate` requires `--active-tcp` followed by the six values from
`camera_in_tcp.tcp_offset` (replace TX/TY/TZ/RX/RY/RZ above). It cross-checks that
declaration against nominal UR10e kinematics and rejects motion during capture.
`--use-fk` instead uses calibrated flange FK and opens RTDE control. Poor and
simulated calibrations are refused; a recorded camera serial selects that device.
`verify` checks both position (`--max-mm`, default 2) and rotation (`--max-deg`,
default 1). Session solving refuses a board manifest changed since collection.

Plans print every flange pose and joint target, and store them in session metadata.
Travel limits and controller preflight checks apply to endpoints. Joint-space moves,
including the probes, can follow curved Cartesian paths; validate clearance for the
whole arm, camera, cables and board. `verify --refit-board` requires at least six
views with sufficient rotations about two axes, since a single view can absorb a
wrong camera calibration into the refitted board pose.
