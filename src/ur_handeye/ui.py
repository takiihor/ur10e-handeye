"""Guided local desktop UI; no robot or camera is opened until an action is chosen."""

import base64
import io
import json
from datetime import datetime
from pathlib import Path

from .cli import DEFAULT_MANIFEST
from .dataset import load_result
from .ui_jobs import JobRunner


def session_name(prefix="cal"):
    return str(Path("sessions") / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S-%f}")


def build_command(action, settings, *, calibration=None, verification=None):
    """Build an argument list, never a shell command. Refuse bad paths before connecting."""
    session = Path(settings["session"]).expanduser().resolve()
    if action == "solve":
        return ["solve", "--session", str(session)]
    if action == "verify":
        if not calibration or not verification:
            raise ValueError("Load a calibration and choose a verification session first")
        if session == Path(verification).expanduser().resolve():
            raise ValueError("Choose fresh verification observations, separate from the calibration session")
        return ["verify", "--calibration", str(calibration), "--session", str(verification)]
    if action == "simulate":
        output = session
        command = ["simulate", "--output", str(output)]
    else:
        if not settings["robot_ip"].strip():
            raise ValueError("Enter the robot IP address in Setup")
        manifest = Path(settings["manifest"]).expanduser().resolve()
        if not manifest.is_file():
            raise ValueError("Choose the generated calibration board_manifest.json in Setup")
        common = [
            "--robot-ip",
            settings["robot_ip"].strip(),
            "--manifest",
            str(manifest),
            "--board",
            settings["board"].strip(),
        ]
        if settings["serial"].strip():
            common += ["--serial", settings["serial"].strip()]
        if action == "check":
            return ["check", *common]
        if action not in ("plan", "collect", "replay", "manual"):
            raise ValueError("Unknown UI action")
        from .poses import PlanOptions
        from .robot import MotionLimits

        speed, views = float(settings["speed"]), int(settings["views"])
        MotionLimits(joint_speed_rad_s=speed)
        PlanOptions(max_views=views)
        if views > 14:
            raise ValueError("Choose 6-14 calibration views")
        output = session
        if action == "plan":
            output = session.parent / (session.name + f"-plan-{datetime.now():%H%M%S-%f}")
        elif action in ("replay", "manual"):
            if not verification:
                raise ValueError("Choose a new verification session folder")
            output = Path(verification).expanduser().resolve()
        command = ["collect", *common, "--output", str(output), "--joint-speed", str(speed), "--views", str(views)]
        if action == "plan":
            command += ["--dry-run"]
        elif action == "replay":
            command += ["--mode", "replay", "--replay", str(session)]
        elif action == "manual":
            command += ["--mode", "manual"]
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"{output} already contains data. Choose a new session folder; existing data is preserved.")
    return command


class CalibrationUI:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.root = tk, ttk, root
        self.runner = JobRunner()
        self.action = None
        self.closing = False
        self.checked = self.planned = None
        self.result = self.result_path = None
        self.preview_photo = None
        self.accepted = self.rejected = 0
        self.widgets = []
        self.settings = {
            key: tk.StringVar(value=value)
            for key, value in {
                "robot_ip": "192.168.1.10",
                "manifest": DEFAULT_MANIFEST,
                "board": "calibration",
                "serial": "",
                "session": session_name(),
                "speed": "0.1",
                "views": "14",
            }.items()
        }
        self.verification = tk.StringVar(value=session_name("check"))
        self.status = tk.StringVar(value="Ready · choose Setup to begin")
        self.sample_status = tk.StringVar(value="No captures yet")
        self.result_status = tk.StringVar(value="No calibration loaded")
        self.result_location = tk.StringVar()
        self.warnings = tk.StringVar(value="Solve a recorded session or run a simulation to see results.")
        self.pose = tk.StringVar(value="-")
        self.verify_status = tk.StringVar(value="Collect fresh views with the board in the same fixed position.")
        self._layout()
        root.report_callback_exception = self.callback_error
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(80, self.poll)

    def _button(self, parent, text, command, **kwargs):
        button = self.ttk.Button(parent, text=text, command=command, **kwargs)
        self.widgets.append(button)
        return button

    def _text(self, parent, text, **kwargs):
        return self.ttk.Label(parent, text=text, wraplength=570, justify="left", **kwargs)

    def _entry(self, parent, row, label, variable, browse=None):
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
        entry = self.ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        self.widgets.append(entry)
        if browse:
            self._button(parent, "Browse...", browse).grid(row=row, column=2, padx=(6, 0), pady=5)

    def _layout(self):
        import tkinter.font as font

        tk, ttk, root = self.tk, self.ttk, self.root
        root.title("UR10e · Camera calibration")
        root.geometry("1240x920")
        root.minsize(1060, 820)
        root.configure(background="#f3f5f8")
        style = ttk.Style(root)
        style.theme_use("clam")
        font.nametofont("TkDefaultFont").configure(family="Helvetica", size=10)
        font.nametofont("TkTextFont").configure(family="Helvetica", size=10)
        style.configure("TFrame", background="#f3f5f8")
        style.configure("TLabel", background="#f3f5f8", foreground="#203047", font=("Helvetica", 10))
        style.configure("Title.TLabel", font=("Helvetica", 20, "bold"))
        style.configure("Heading.TLabel", font=("Helvetica", 13, "bold"))
        style.configure("TButton", padding=(12, 8), font=("Helvetica", 10))
        style.configure("TNotebook", background="#f3f5f8", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 9), background="#e2e8f0", font=("Helvetica", 10))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#0f766e")])
        style.configure("TCheckbutton", background="#f3f5f8", font=("Helvetica", 10))
        style.configure("TLabelframe", background="#f3f5f8")
        style.configure("TLabelframe.Label", background="#f3f5f8", font=("Helvetica", 10))
        style.configure("Primary.TButton", background="#0f766e", foreground="white")
        style.map(
            "Primary.TButton",
            background=[("disabled", "#e2e8f0"), ("active", "#115e59")],
            foreground=[("disabled", "#6b7280")],
        )
        style.configure("Stop.TButton", foreground="#a12424", font=("Helvetica", 11, "bold"))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        header = ttk.Frame(root, padding=(22, 18))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Camera calibration", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="UR10e + RealSense  /  Guided hand-eye calibration").grid(row=1, column=0, sticky="w")
        self.stop_button = ttk.Button(
            header, text="Stop operation", style="Stop.TButton", command=self.stop, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, rowspan=2)
        body = ttk.Frame(root, padding=(22, 0, 22, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        self.tabs = ttk.Notebook(body)
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        self.pages = []
        for title in ("1 Setup", "2 Check & plan", "3 Capture", "4 Results", "5 Verify"):
            page = ttk.Frame(self.tabs, padding=16)
            self.tabs.add(page, text=title)
            self.pages.append(page)
        self._setup_page()
        self._plan_page()
        self._capture_page()
        self._result_page()
        self._verify_page()
        preview = ttk.Frame(body, padding=(8, 12))
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        ttk.Label(preview, text="Latest board image", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self._text(preview, "Updated after a check or capture. Green circles mark detected corners.").grid(
            row=1, column=0, sticky="w", pady=(6, 14)
        )
        self.image_label = tk.Label(
            preview,
            text="Position the camera so the whole\nboard is visible, then Check view.",
            background="#e1e7ef",
            foreground="#53647b",
            width=46,
            height=16,
        )
        self.image_label.grid(row=2, column=0, sticky="ew")
        ttk.Label(preview, textvariable=self.sample_status, wraplength=390).grid(row=3, column=0, sticky="w", pady=12)
        self._text(
            preview,
            "Keep the board fixed and supervise motion with the physical emergency stop accessible. "
            "Stop operation requests software interruption and cleanup.",
            foreground="#8b4c12",
        ).grid(row=4, column=0, sticky="w", pady=8)
        self.prompt_frame = ttk.LabelFrame(root, text="Operator action required", padding=12)
        self.prompt_frame.grid(row=2, column=0, sticky="ew", padx=22, pady=8)
        self.prompt_frame.columnconfigure(0, weight=1)
        self.prompt_text = tk.Text(self.prompt_frame, height=6, wrap="word", font=("Courier", 9))
        self.prompt_text.grid(row=0, column=0, sticky="ew")
        prompt_scroll = ttk.Scrollbar(self.prompt_frame, command=self.prompt_text.yview)
        prompt_scroll.grid(row=0, column=1, sticky="ns")
        self.prompt_text.configure(yscrollcommand=prompt_scroll.set)
        self.prompt_buttons = ttk.Frame(self.prompt_frame)
        self.prompt_buttons.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.prompt_frame.grid_remove()
        log_frame = ttk.LabelFrame(root, text="Activity", padding=8)
        log_frame.grid(row=3, column=0, sticky="ew", padx=22, pady=(4, 8))
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame,
            height=5,
            state="disabled",
            wrap="word",
            font=("Courier", 9),
            background="#19273b",
            foreground="#e6edf6",
        )
        self.log.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)
        ttk.Label(root, textvariable=self.status, padding=(22, 8)).grid(row=4, column=0, sticky="ew")

    def _setup_page(self):
        ttk, page = self.ttk, self.pages[0]
        self._text(page, "Start with one clear view", style="Heading.TLabel").pack(anchor="w")
        self._text(
            page, "Set the first pose using the pendant. Automatic collection handles the remaining views."
        ).pack(anchor="w", pady=(6, 12))
        form = ttk.Frame(page)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        for row, (key, label, browse) in enumerate(
            [
                ("robot_ip", "Robot IP", None),
                ("manifest", "Board manifest", self.browse_manifest),
                ("board", "Board name", None),
                ("serial", "Camera serial (optional)", None),
                ("session", "Session folder", self.browse_session),
                ("speed", "Joint speed (rad/s)", None),
                ("views", "Views (6-14)", None),
            ]
        ):
            self._entry(form, row, label, self.settings[key], browse)
        self.ready = []
        for text in (
            "Camera rigidly mounted; board flat, fixed and printed at 100% scale",
            "Robot in Remote Control; other RTDE control clients disconnected",
            "Initial pose set: whole board visible, roughly centred, 0.3-0.6 m away",
            "Entire motion area checked; emergency stop accessible",
        ):
            value = self.tk.BooleanVar(value=False)
            box = ttk.Checkbutton(page, text=text, variable=value)
            box.pack(anchor="w", pady=4)
            self.widgets.append(box)
            self.ready.append(value)
        buttons = ttk.Frame(page)
        buttons.pack(fill="x", pady=(14, 0))
        self._button(buttons, "Continue to check >", lambda: self.tabs.select(1)).pack(side="left")
        self._button(buttons, "Try a simulation", self.simulate).pack(side="left", padx=8)
        self._text(page, "Simulation uses no hardware. Live robot operation still needs physical validation.").pack(
            anchor="w", pady=10
        )

    def _plan_page(self):
        page = self.pages[1]
        self._text(page, "Check the view, then inspect the plan", style="Heading.TLabel").pack(anchor="w")
        self._text(
            page, "1. Check reads robot status and captures a board image. Adjust the first pose until the view passes."
        ).pack(anchor="w", pady=12)
        self._button(page, "Check view (read-only)", lambda: self.start("check"), style="Primary.TButton").pack(
            anchor="w"
        )
        self._text(
            page, "2. Inspect a preliminary plan without moving. Review each flange pose and joint target in Activity."
        ).pack(anchor="w", pady=12)
        self._button(page, "Inspect plan (no motion)", lambda: self.start("plan")).pack(anchor="w")
        self._text(
            page,
            "Planning opens RTDE control and may stop a running robot program. The dry run assumes camera "
            "orientation; the live plan is recalculated after probing. Endpoint checks do not check collision "
            "clearance along joint-motion paths.",
            foreground="#8b4c12",
        ).pack(anchor="w", pady=16)
        self._button(page, "Continue to capture >", lambda: self.tabs.select(2)).pack(anchor="w", pady=10)

    def _capture_page(self):
        page = self.pages[2]
        self._text(page, "Automatic capture from your first pose", style="Heading.TLabel").pack(anchor="w")
        self._text(page, "After you start, review and approve each motion plan below. The robot will:").pack(
            anchor="w", pady=12
        )
        self._text(
            page,
            "1. Capture the starting view.\n\n2. Probe 30 mm along three flange axes, returning after each."
            "\n\n3. Visit up to 14 views with different tilts, rolls and distances."
            "\n\n4. Wait for settling, then average five frames at each pose."
            "\n\n5. Ask before an updated plan, then return to the starting pose on success.",
        ).pack(anchor="w")
        self._button(page, "Start automatic collection", lambda: self.start("collect"), style="Primary.TButton").pack(
            anchor="w", pady=20
        )
        self._text(
            page,
            "On a fault or interruption, motion stops in place; there is no automatic recovery move. "
            "After collection, choose Results to solve the saved session.",
        ).pack(anchor="w")

    def _result_page(self):
        ttk, page = self.ttk, self.pages[3]
        page.columnconfigure(0, weight=1)
        ttk.Label(page, textvariable=self.result_status, style="Heading.TLabel", wraplength=570).grid(
            row=0, column=0, sticky="w"
        )
        buttons = ttk.Frame(page)
        buttons.grid(row=1, column=0, sticky="w", pady=12)
        self._button(buttons, "Solve session", lambda: self.start("solve"), style="Primary.TButton").pack(side="left")
        self._button(buttons, "Open calibration...", self.open_result).pack(side="left", padx=6)
        self._button(buttons, "Open session...", self.browse_session).pack(side="left")
        ttk.Label(page, textvariable=self.result_location, wraplength=570).grid(row=2, column=0, sticky="w", pady=4)
        metric_frame = ttk.Frame(page)
        metric_frame.grid(row=3, column=0, sticky="ew", pady=6)
        metric_frame.columnconfigure(0, weight=1)
        self.metrics = ttk.Treeview(metric_frame, columns=("value",), show="tree headings", height=9)
        self.metrics.heading("#0", text="Quality metric")
        self.metrics.heading("value", text="Result")
        self.metrics.column("#0", width=300)
        self.metrics.column("value", width=210)
        self.metrics.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(metric_frame, command=self.metrics.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.metrics.configure(yscrollcommand=scrollbar.set)
        ttk.Label(page, textvariable=self.warnings, wraplength=570, foreground="#8b4c12").grid(
            row=4, column=0, sticky="w", pady=6
        )
        ttk.Label(page, text="Camera in flange · UR pose [m, rad]", style="Heading.TLabel").grid(
            row=5, column=0, sticky="w", pady=(12, 4)
        )
        ttk.Label(page, textvariable=self.pose, wraplength=570, font=("Courier", 10)).grid(row=6, column=0, sticky="w")
        exports = ttk.Frame(page)
        exports.grid(row=7, column=0, sticky="w", pady=12)
        self._button(exports, "Copy UR pose", self.copy_pose).pack(side="left")
        self._button(exports, "Save calibration copy...", self.save_result).pack(side="left", padx=6)
        self._button(exports, "Continue to verify >", lambda: self.tabs.select(4)).pack(side="left")

    def _verify_page(self):
        page = self.pages[4]
        self._text(page, "Verify using fresh observations", style="Heading.TLabel").pack(anchor="w")
        self._text(
            page,
            "Keep the board fixed. Replaying captures new images at previous poses. For a stronger check, "
            "manually collect 10-15 different views with rotations about two axes. Jog using the pendant, "
            "then choose Capture view while stationary; choose Finish when done.",
        ).pack(anchor="w", pady=12)
        form = self.ttk.Frame(page)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self._entry(form, 0, "Verification folder", self.verification, self.browse_verification)
        self._button(page, "Collect fresh views (replay)", lambda: self.start("replay")).pack(anchor="w", pady=(14, 8))
        self._button(page, "Collect different views (manual)", lambda: self.start("manual")).pack(anchor="w", pady=4)
        self._button(page, "Verify calibration", lambda: self.start("verify"), style="Primary.TButton").pack(
            anchor="w", pady=12
        )
        self._text(
            page,
            "Default pass limits: maximum board-position error <= 2 mm and rotation error <= 1°. "
            "The CLI offers board refitting if the board was moved.",
        ).pack(anchor="w", pady=6)
        self.ttk.Label(page, textvariable=self.verify_status, wraplength=570, style="Heading.TLabel").pack(
            anchor="w", pady=12
        )

    def values(self):
        return {key: value.get() for key, value in self.settings.items()}

    def hardware_key(self):
        return tuple((k, v) for k, v in self.values().items() if k != "session")

    def browse_manifest(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            parent=self.root, title="Select board manifest", filetypes=[("JSON", "*.json")]
        )
        if path:
            self.settings["manifest"].set(path)

    def browse_session(self):
        from tkinter import filedialog

        path = filedialog.askdirectory(
            parent=self.root, title="Choose a new or recorded session folder", mustexist=False
        )
        if path:
            self.settings["session"].set(path)
            self.result_location.set(f"Selected session: {path}")

    def browse_verification(self):
        from tkinter import filedialog

        path = filedialog.askdirectory(parent=self.root, title="Choose verification session folder", mustexist=False)
        if path:
            self.verification.set(path)

    def error(self, message):
        from tkinter import messagebox

        messagebox.showerror("Calibration", str(message), parent=self.root)

    def simulate(self):
        if self.runner.active:
            return
        self.settings["session"].set(session_name("sim"))
        self.tabs.select(3)
        self.start("simulate")

    def start(self, action):
        try:
            if self.runner.active:
                raise ValueError("Wait for the current operation and hardware cleanup to finish")
            if action in ("plan", "collect", "replay", "manual") and not all(v.get() for v in self.ready):
                raise ValueError("Complete the four preparation checks in Setup before opening robot control")
            if action in ("plan", "collect") and self.checked != self.hardware_key():
                raise ValueError("Run Check view successfully with the current settings first")
            if action == "collect" and self.planned != self.hardware_key():
                raise ValueError("Inspect the preliminary plan with the current settings first")
            if action in ("replay", "manual", "verify"):
                if not self.result:
                    raise ValueError("Solve or open a calibration first")
                if self.result["verdict"] == "poor":
                    raise ValueError("This calibration is poor. Review warnings and collect a better session first")
                if action != "verify" and self.is_simulated(self.result):
                    raise ValueError("A simulated calibration cannot be used for live verification collection")
                source = self.result.get("session")
                if action == "verify" and source and Path(source).resolve() == Path(self.verification.get()).resolve():
                    raise ValueError("Verify with fresh observations, separate from the calibration session")
            argv = build_command(
                action, self.values(), calibration=self.result_path, verification=self.verification.get()
            )
            self.job_key = self.hardware_key()
            self.runner.start(argv)
            self.action = action
            if action in ("check", "collect"):
                self.planned = None
            if action in ("collect", "solve", "simulate"):
                self.result = self.result_path = None
                self.result_status.set("Collecting..." if action == "collect" else "Solving...")
                self.result_location.set(f"Session: {self.settings['session'].get()}")
                self.metrics.delete(*self.metrics.get_children())
                self.pose.set("-")
                self.warnings.set("Results will appear when solving completes.")
            self.prompt_frame.grid_remove()
            if action in ("check", "collect", "simulate", "manual", "replay"):
                self.accepted = self.rejected = 0
                self.preview_photo = None
                self.image_label.configure(image="", text="Waiting for the first image...", width=46, height=16)
                self.sample_status.set("Waiting for the first capture...")
            if action == "verify":
                self.verify_status.set("Verification running...")
            self.status.set(f"Running {action}...")
            self.append_log(f"Starting {action} · {' '.join(argv)}")
            self.set_busy(True)
        except (ValueError, OSError, RuntimeError) as exc:
            self.error(exc)

    def set_busy(self, busy):
        for widget in self.widgets:
            widget.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy and not self.runner.stopping else "disabled")

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 800:
            self.log.delete("1.0", "100.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def show_request(self, event):
        if self.runner.stopping:
            return
        self.status.set(
            "Waiting for your approval" if event["kind"] == "motion" else "Position the robot, then capture"
        )
        self.prompt_text.configure(state="normal")
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", event["summary"])
        self.prompt_text.configure(state="disabled")
        self.prompt_text.yview_moveto(0)
        for widget in self.prompt_buttons.winfo_children():
            widget.destroy()
        options = (
            (("Approve these moves", True), ("Cancel plan", False))
            if event["kind"] == "motion"
            else (("Capture view", ""), ("Finish collection", "q"))
        )
        for label, value in options:
            self.ttk.Button(
                self.prompt_buttons, text=label, command=lambda v=value, i=event["id"]: self.answer(i, v)
            ).pack(side="left", padx=(0, 8))
        self.prompt_frame.grid()

    def answer(self, request_id, value):
        try:
            self.runner.answer(request_id, value)
            self.prompt_frame.grid_remove()
            self.status.set(f"Running {self.action}...")
        except (ValueError, OSError) as exc:
            self.error(exc)

    def show_observation(self, event):
        from PIL import Image, ImageTk

        if event.get("image"):
            picture = Image.open(io.BytesIO(base64.b64decode(event["image"])))
            picture.thumbnail((430, 260))
            self.preview_photo = ImageTk.PhotoImage(picture)
            self.image_label.configure(image=self.preview_photo, text="", width=430, height=260)
        if event["status"] == "ok":
            self.accepted += 1
        else:
            self.rejected += 1
        rmse = event.get("rmse_px")
        detail = f" · RMSE {rmse:.3f} px" if rmse is not None else ""
        reasons = ", ".join(event.get("reasons", []))
        self.sample_status.set(
            f"{event['label']} · {event['status'].upper()}\n{event['corners']} corners{detail}\n"
            f"{self.accepted} accepted · {self.rejected} rejected" + (f"\n{reasons}" if reasons else "")
        )

    @staticmethod
    def is_simulated(result):
        return bool(
            result.get("simulation_truth")
            or result.get("camera", {}).get("simulated")
            or result.get("robot", {}).get("host") == "simulated"
        )

    def show_result(self, result, path):
        from .transforms import describe

        self.result, self.result_path = result, path
        simulated = self.is_simulated(result)
        self.result_status.set(("SIMULATION · " if simulated else "") + result["verdict"].upper())
        self.result_location.set(str(path))
        m = result.get("metrics", {})
        c, loo = m.get("consistency", {}), m.get("leave_one_out", {})
        rows = [
            ("Usable samples", m.get("sample_count"), ""),
            ("Corner reprojection RMSE", m.get("reprojection_rmse_px"), "px"),
            ("Board consistency / RMS", c.get("position_rms_mm"), "mm"),
            ("Board consistency / rotation RMS", c.get("rotation_rms_deg"), "°"),
            ("Leave-one-out / RMS", loo.get("position_rms_mm"), "mm"),
            ("Rotation range", m.get("rotation_diversity", {}).get("max_relative_rotation_deg"), "°"),
            ("Rotation axis spread", m.get("rotation_diversity", {}).get("axis_spread"), ""),
        ]
        self.metrics.delete(*self.metrics.get_children())
        for label, value, unit in rows:
            display = "Unavailable" if value is None else f"{value:.3f} {unit}".strip()
            self.metrics.insert("", "end", text=label, values=(display,))
        sigma = m.get("camera_in_flange_uncertainty", {}).get("combined_translation_mm")
        if sigma:
            self.metrics.insert(
                "",
                "end",
                text="Camera translation 1-sigma / X Y Z",
                values=(" / ".join(f"{v:.3f}" for v in sigma) + " mm",),
            )
        camera = describe(result["camera_in_flange"]["matrix"])
        xyz = camera["xyz_mm"]
        if xyz:
            self.metrics.insert(
                "", "end", text="Camera in flange / X Y Z", values=(" / ".join(f"{v:.2f}" for v in xyz) + " mm",)
            )
        board = describe(result["board_in_base"]["matrix"])
        self.metrics.insert(
            "", "end", text="Board in base / X Y Z", values=(" / ".join(f"{v:.2f}" for v in board["xyz_mm"]) + " mm",)
        )
        self.metrics.insert(
            "",
            "end",
            text="Board in base / roll pitch yaw",
            values=(" / ".join(f"{v:.2f}" for v in board["rpy_deg"]) + " deg",),
        )
        warnings = list(result.get("warnings", []))
        if simulated:
            warnings.insert(0, "Simulated result - do not use on live hardware.")
        if result["verdict"] == "poor":
            warnings.insert(0, "Calibration failed quality limits. Collect a better session before use.")
        self.warnings.set("\n".join(warnings) or "Quality checks passed. Verify with fresh observations before use.")
        self.pose.set(json.dumps([round(v, 8) for v in camera["pose_ur"]]))
        self.verify_status.set("Collect fresh views with the board in the same fixed position.")
        self.tabs.select(3)

    def open_result(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(parent=self.root, title="Open calibration", filetypes=[("JSON", "*.json")])
        if path:
            try:
                result = load_result(path)
                if result.get("session"):
                    self.settings["session"].set(result["session"])
                self.show_result(result, path)
            except (ValueError, OSError) as exc:
                self.error(exc)

    def copy_pose(self):
        if self.result:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.pose.get())
            self.status.set("Camera-in-flange UR pose copied")

    def save_result(self):
        from tkinter import filedialog

        if self.result:
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save calibration copy",
                defaultextension=".json",
                initialfile="calibration.json",
                filetypes=[("JSON", "*.json")],
            )
            if path:
                try:
                    Path(path).write_text(json.dumps(self.result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    self.status.set(f"Saved calibration copy to {path}")
                except OSError as exc:
                    self.error(exc)

    def stop(self):
        try:
            self.runner.stop()
        except ProcessLookupError:
            pass
        self.prompt_frame.grid_remove()
        self.stop_button.configure(state="disabled")
        self.status.set("Stop requested · waiting for interruption and hardware cleanup; check the robot state")

    def close(self):
        self.closing = True
        if self.runner.active:
            self.stop()
        else:
            self.root.destroy()

    def callback_error(self, error_type, error, traceback):
        if issubclass(error_type, KeyboardInterrupt):
            self.close()
        else:
            self.stop()
            self.append_log(f"UI action failed; stop requested: {error}")

    def _poll_events(self):
        for event in self.runner.drain():
            kind = event["type"]
            if kind == "log":
                self.append_log(event["message"])
            elif kind == "request":
                self.show_request(event)
            elif kind == "observation":
                self.show_observation(event)
            elif kind == "result":
                self.show_result(event["result"], event["path"])
            elif kind == "verification":
                report = event["report"]
                c = report["consistency"]
                self.verify_status.set(
                    f"{'PASS' if report['passed'] else 'FAIL'} · maximum position "
                    f"{c['position_max_mm']:.3f} mm / rotation {c['rotation_max_deg']:.3f}°"
                )
            elif kind == "done":
                code = event["code"]
                self.set_busy(False)
                self.prompt_frame.grid_remove()
                if code == 0 and self.action in ("check", "plan"):
                    if self.action == "check":
                        self.checked = self.job_key
                    else:
                        self.planned = self.job_key
                elif self.action == "check":
                    self.checked = self.planned = None
                elif self.action == "plan":
                    self.planned = None
                if self.action in ("collect", "solve", "simulate") and not self.result:
                    self.result_status.set("No new calibration · review Activity")
                self.status.set(
                    f"{self.action.capitalize()} {'completed' if code == 0 else 'stopped or failed'} "
                    f"· exit {code} · see Activity for details"
                )
                if code == 0 and self.action == "collect":
                    self.tabs.select(3)
                self.append_log(self.status.get())

    def poll(self):
        try:
            self._poll_events()
        except Exception as exc:
            # A display/decoding error must not leave an active worker running unattended,
            # or prevent later exit events from being processed.
            self.stop()
            self.result = self.result_path = None
            self.result_status.set("Display update failed · review Activity")
            self.append_log(f"UI update failed; stop requested: {exc}")
        if self.closing and not self.runner.active:
            self.root.destroy()
            return
        self.root.after(80, self.poll)


def launch():
    try:
        import fcntl
        import tkinter as tk

        from PIL import ImageTk  # noqa: F401 -- check optional system Tk support before opening the window
    except ImportError as exc:
        raise ValueError("Desktop UI needs Tk. On Ubuntu/Debian install python3-tk, then rerun "
                         "./start_calibration_eye_in_robot.sh") from exc
    directory = Path.home() / ".cache" / "ur10e-handeye"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "ui.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("The calibration UI is already open for this user") from exc
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            raise ValueError("Desktop UI needs a graphical display. Run ./start_calibration_eye_in_robot.sh "
                             "in your desktop session") from exc
        try:
            app = CalibrationUI(root)
            # SIGTERM is routed through KeyboardInterrupt by the CLI. Keep the event loop alive
            # until the worker has acknowledged interruption and exited its hardware cleanup path.
            while True:
                try:
                    root.mainloop()
                    break
                except KeyboardInterrupt:
                    app.close()
                    if not app.runner.active:
                        break
        except Exception:
            if "app" in locals() and app.runner.active:
                app.close()
                root.mainloop()
            raise
    return 0
