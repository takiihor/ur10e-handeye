# UR10e camera calibration

Calibrate a RealSense RGB camera using a ChArUco board. Choose the matching setup
in the UI; each uses its own capture procedure.

| Setup | Camera | Calibration board | Result |
| --- | --- | --- | --- |
| **Eye in hand** | On robot flange | Fixed on platform | `camera_in_flange` |
| **Eye to hand** | Fixed on platform | On robot flange | `camera_in_base` |

## Start

Place `charuco-generator` and `charuco-realsense` next to this repository.
Install Python 3.11–3.13 and Tk (`python3-tk` on Ubuntu/Debian), then run:

```sh
./start_calibration_eye_in_robot.sh
```

The launcher creates `.venv` and installs dependencies. Set `PYTHON_BIN` to choose
Python when creating a new environment.

For either setup, print the board at 100%, measure its square size and use a flat
backing. Enter the robot IP, board manifest, board name and a new session folder.
Select RGB intrinsics for this camera/resolution if available; otherwise capture
uses the factory model. Check clearance and keep the emergency stop accessible.

## Eye to hand — fixed camera, board on flange

1. Secure the camera to the platform and the board to the flange.
2. Keep the robot in **Local Control**, powered on with brakes released. Enable
   RTDE receive (port **30004**) and the read-only Primary interface (**30011**)
   in the robot/network configuration. The software reads calibrated poses and
   the active TCP offset; it does not upload a robot program or take motion control.
3. Jog using the pendant until the whole board is visible, roughly centred and
   0.3–0.6 m away. Complete the preparation checks and choose **Check view**.
   This checks both the camera and the calibrated pose source.
4. Continue to Capture and choose **Start manual collection**. Jog the board,
   release the jog/Freedrive control, wait for settling, then choose **Capture view**.
   Collect 10–15 views with different positions, distances and tilts about at least
   two axes, spanning at least 30°. Keep the complete board visible. Choose **Finish collection**.
5. In Results, choose **Solve session**. Prefer **GOOD**; investigate warnings and
   recollect poor data. The software estimates both `camera_in_base` and
   `board_in_flange`; no measured board mounting offset is needed.
6. Keep both mounts unchanged. Choose a new verification folder and **Collect
   different views (manual)**, still in Local Control. Collect fresh varied views,
   finish, then choose **Verify calibration**. Default limits are **2 mm and 1°**.

**Stop operation** ends eye-to-hand capture; it does not stop pendant-controlled
robot motion. Automatic planning is not available for this setup. The eye-to-hand
UI uses manual verification; CLI `--mode replay` is a separate operation that
requires Remote Control, endpoint review and explicit motion approval.

## Eye in hand — camera on flange, fixed board

1. Secure the camera to the flange and keep the board fixed on the platform.
2. Set the first visible board pose using the pendant, then select **Remote Control**
   and disconnect other RTDE control clients. Complete the preparation checks.
3. Choose **Check view**, then **Inspect plan (no motion)**. Review targets and clearance.
4. Choose **Start automatic collection**, starting at 0.1 rad/s. Approve the three
   30 mm probe moves and subsequent calibration plans. Success returns to the start.
5. Choose **Solve session** and review the verdict and warnings. Prefer **GOOD**.
6. Keep the board fixed and choose a new verification folder. Collect fresh views
   using replay, then choose **Verify calibration**. Default limits are **2 mm and 1°**.

The eye-in-hand motion pipeline still uses RTDE control. Planning validates endpoints,
not collision clearance along paths. Its software stop is not an emergency stop.

## Use the result

Results are saved to `SESSION/calibration.json`; repeated UI solves use new filenames.
Matrices use metres and camera optical coordinates. For homogeneous points:

```text
Eye in hand: p_base = T_base_flange @ camera_in_flange.matrix @ p_camera
Eye to hand: p_base = camera_in_base.matrix @ p_camera
```

Applications must handle the explicit `type` field. Sessions record their setup;
older sessions without a setup remain eye-in-hand sessions. Mixed setups are refused
for verification and replay. Physical UR10e/RealSense/URSim validation is still pending.

## Command line and offline tools

```sh
./start_calibration_eye_in_robot.sh collect \
  --calibration-type eye_to_hand --mode manual --robot-ip ROBOT_IP \
  --manifest PATH/board_manifest.json --board BOARD_NAME --output sessions/fixed-camera
./start_calibration_eye_in_robot.sh solve --session sessions/fixed-camera
./start_calibration_eye_in_robot.sh --help
```

Solving selects the setup from the session. Use a separate fresh session with
`verify --calibration FILE --session FRESH_SESSION`. `locate` accepts both result types.
For an eye-to-hand camera check, include `check --calibration-type eye_to_hand`.

For inaccurate RGB intrinsics, use **Estimate RGB intrinsics from session**, solve
again and independently verify. The CLI equivalents are `intrinsics --session SESSION
--output NEW_FILE` and `solve --session SESSION --intrinsics NEW_FILE`.
**Calibrate RGB and solve** combines intrinsic estimation and the hand-eye solve
into one offline operation (`calibrate --session SESSION --intrinsics-output NEW_RGB_FILE
--output NEW_CALIBRATION_FILE`). Estimated intrinsics remain unvalidated until fresh
views pass verification. The UI remembers a selected real camera model across restarts;
serial and resolution checks still apply. `ui --calibration FILE` opens a saved result.

Use `reprocess --session SESSION --output NEW_SESSION` to apply updated corner filtering
to all saved raw frames without changing the original recording. It preserves robot
poses, setup type and motion rejections. Reprocessed data is not independent verification.
An intrinsic fit's training error does not establish physical accuracy.

**Try eye-in-hand simulation** runs without hardware. The rendered simulator supports
eye in hand only; simulated results must not be used on hardware.
Run tests with `.venv/bin/python -m pytest -q`.
