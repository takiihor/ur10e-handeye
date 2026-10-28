# UR10e eye-in-hand calibration

Calibrate a wrist-mounted RealSense D435i against a fixed ChArUco board. The result
is `calibration.json`, containing the camera pose in the robot flange frame.

## Start

Keep `charuco-generator` and `charuco-realsense` next to this repository. The launcher
creates `.venv` and installs dependencies automatically. Python 3.11–3.13 and Tk
(`python3-tk` on Ubuntu/Debian) are required.

The launcher prefers `~/.local/opt/ur-handeye-python/bin/python3.12` when creating
an environment, if installed. This build uses system Tk for smooth desktop fonts.
Set `PYTHON_BIN` to select another Python when creating a new environment.

```sh
./start_calibration_eye_in_robot.sh
```

## Calibration procedure

1. **Prepare.** Mount the camera rigidly on the wrist. Print the board at 100% scale,
   measure its squares, and fix it to a flat plate that stays stationary. Check arm,
   camera and cable clearance throughout the planned motion area.
2. **Set up.** Enter the robot IP and choose a new session folder. For the board
   tested on this PC, select
   `../charuco-generator/output/two_stage/calibration_stage2/board_manifest.json`
   and board name `calibration_stage2`. Its squares must measure **15 mm**.
   Start with **0.1 rad/s** and 14 views. Complete the preparation checks.
3. **Connect the robot.** Power on, release brakes and enable Remote Control.
   Disconnect other RTDE control applications. Pendant-jog until the camera sees
   the whole board, roughly centred at 0.3–0.6 m.
4. **Check and plan.** Choose **Check view (read-only)**, then **Inspect plan
   (no motion)**. Adjust the initial pose until the check passes. Review every
   target and clearance; endpoint checks do not detect collisions along the paths.
   Planning opens RTDE control and can stop an existing robot program.
5. **Collect.** Choose **Start automatic collection**. Review and approve the
   three 30 mm probe moves, then the calibration views. The software moves and
   captures automatically; an updated plan requires another approval. Supervise
   motion with the emergency stop accessible. A successful run returns to the start.
6. **Solve.** Choose **Solve session**. Review the verdict and warnings. Prefer a
   **GOOD** result; investigate warnings and recollect poor data. The result is
   saved as `SESSION/calibration.json`.
7. **Verify.** Keep the board fixed and choose a new verification folder. Collect
   fresh views using replay, or varied manual views for a stronger check. Choose
   **Verify calibration**. Default limits are **2 mm and 1°**. Use a separate
   session from the calibration data.
8. **Use.** Supply the verified `calibration.json` to the consuming application:
   `p_base = T_base_flange @ camera_in_flange @ p_camera`.

**Stop operation** requests a software stop; it is not an emergency stop. A fault
leaves the robot in place without automatic recovery motion. Physical UR10e/URSim
operation has not yet been validated; the first hardware run must be slow and attended.

## Test without the robot

Close other camera previews before running a check:

```sh
./start_calibration_eye_in_robot.sh check \
  --manifest ../charuco-generator/output/two_stage/calibration_stage2/board_manifest.json \
  --board calibration_stage2
```

For a complete simulated calibration, use **Try a simulation** in the UI or:

```sh
./start_calibration_eye_in_robot.sh simulate
```

Simulated results must not be used on hardware. For all command-line options:

```sh
./start_calibration_eye_in_robot.sh --help
```

## Tests and CI

```sh
.venv/bin/python -m pytest -q
```

Tests use simulation and fake hardware. GitHub Actions runs them on this PC's
self-hosted Linux runner on pushes and manual dispatch. Review code before running
it on the self-hosted machine; pull requests do not automatically trigger CI.
