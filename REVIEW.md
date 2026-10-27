# Software review — 27 October 2026

Reviewed the calibration, board generator, RealSense detector/service, and pick/place
repositories together. The review covered CLI option combinations, robot motion and
force-probe paths, UI connection ownership, calibration/observation persistence,
transform conventions, capture quality, generator publication, and package setup.
No robot, camera, gripper or URSim connection was opened.

## Findings corrected

| Priority | Finding | Correction |
| --- | --- | --- |
| High | `collect --dry-run` was ignored by replay/manual modes; manual could enable freedrive. | Replay has a checked dry-run path; manual dry-run and incompatible flags fail before opening hardware. |
| High | Replay checked joint differences but did not preflight every saved endpoint. | All targets and the return pose now pass controller joint/TCP safety and travel checks before motion; each move rechecks limits. |
| High | Blocking standalone `moveJ`/`moveL` calls delayed Python interruption handling. | Calibration and pick/place use monitored asynchronous moves with deadlines, feedback checks and explicit stop handling. The UI shares the pick/place implementation. |
| High | An interim calibration changed the motion plan without another confirmation; only aggregate plan information was displayed. | Updated plans require confirmation. Every target is printed and the initial full plan is recorded. |
| High | Read-only `locate` silently assumed the calibration-time TCP was still active. | It requires an explicit matching TCP declaration, checks nominal UR10e feedback, rejects movement, and refuses poor/simulated calibrations. Recorded camera serials select the calibrated device. |
| High | Simulated calibration and demo waypoints could reach hardware paths, including preview motion. | Live workflows reject simulated calibration and demo observe poses before motion; teaching checks the declared TCP and camera identity. |
| High | Placement descent lacked the pick descent's gripper safety guard. Late contact could also bypass the descent budget. | Both searches use the guard. Contact beyond the hard descent budget cannot authorize gripping or releasing. |
| Medium | Changed board manifests could silently change an offline solve's geometry. | Recorded manifest hashes must match. A missing in-memory board cannot silently fall back to different geometry. |
| Medium | Corrupt transform translations/bottom rows and redundant fields were insufficiently checked. | Calibration and saved observation readers check finite rigid transforms, redundant TCP relations, observation composition, flags and timestamps. Solver inputs reject invalid IDs/pixels/noise assumptions. |
| Medium | Sparse/repeated frames, profile changes, stale RTDE feedback and TCP changes could evade observation checks. | Frame counts/numbers and profiles are checked. Vision checks robot feedback throughout capture and refuses movement or TCP changes. |
| Medium | Two processes could consume the same preview concurrently. | An exclusive preview lock is acquired before RTDE control; persisted one-attempt consumption remains in force. Configuration hashes are checked around loading. |
| Medium | Verification could pass a rotation error or absorb camera error into a board refit with only one view. | Verification gates position and rotation. Board refitting needs at least six views with sufficient rotation diversity. |
| Medium | SIGTERM bypassed CLI interruption cleanup, and interrupted camera/serial setup could leak a resource. | Robot CLIs route SIGTERM through interruption cleanup. Camera and serial setup clean up on interruption. |
| Medium | The robust solver's acceptance objective did not match its Huber weights. | It now evaluates the corresponding Huber objective, reports convergence conservatively and refuses insufficient rotation range. |
| Medium | Generator `--clean` deleted old prints before replacement generation succeeded. | Deletion waits until staging passes validation. Invalid grid/extreme numeric inputs and non-object saved manifests produce explicit refusals. |
| Medium | Manifest-reading runtime dependencies were missing from consumer package declarations. | Pillow/PDF support and matching packaging/charset pins are declared and installed. Stale generator metadata in the RealSense environment was refreshed. |

## Validation

- Board generator: 180 tests passed; configured Ruff checks passed.
- RealSense: 93 tests passed, including fake camera startup interruption and service/process tests.
- Hand-eye calibration: 85 tests passed; configured Ruff checks passed.
- Pick/place: 178 tests run, 177 passed and one intentional non-refine skip.
- Static correctness checks, Python compilation, shell syntax checks, CLI help and the offline pick/place preview passed.
- A fresh calibration simulation using the generated A4 manifest returned GOOD: camera translation error 0.316 mm and rotation error 0.0505 degrees. The reported per-axis camera translation 1-sigma was approximately [0.305, 0.441, 0.088] mm.

## Remaining validation and limits

This is a software review and simulated/fake-hardware validation, not proof that
there are no remaining defects. Real UR10e/RealSense/URSim operation remains
untested. First use is still `collect --dry-run`, followed by a slow attended run
with a hand on the e-stop.

Controller checks and software travel limits validate endpoints. Joint-space
moves, including the probe moves, can follow curved Cartesian paths; clearance
for the complete robot, camera, cables, tool and board needs cell validation.
Printed scale/flatness, camera intrinsics, mounting rigidity, robot pose accuracy
and planar pose ambiguity still need independent physical measurements.

The consumer environments intentionally use `opencv-python` while the generator's
standalone package metadata requires `opencv-contrib-python-headless`. The sibling
is installed with `--no-deps` to avoid installing two distributions providing
`cv2`; `pip check` therefore reports that one expected metadata conflict. Its
required OpenCV APIs and pinned 4.11 runtime were exercised by the test suites.

No commits were pushed and no hardware motion was requested during this review.
