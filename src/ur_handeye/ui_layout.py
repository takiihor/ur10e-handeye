"""Scrollable desktop panels and wrapping button rows; no hardware ownership."""


class ScrollPage:
    def __init__(self, parent, tk, ttk, *, padding=16):
        self.frame = ttk.Frame(parent)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.frame, width=1, height=1, highlightthickness=0, background="#f3f5f8")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(self.frame, command=self.canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.content = ttk.Frame(self.canvas, padding=padding)
        self.window = self.canvas.create_window(0, 0, window=self.content, anchor="nw")
        self.canvas.bind("<Configure>", self._resize)
        self.content.bind("<Configure>", self._region)
        root = parent.winfo_toplevel()
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            root.bind(event, self._wheel, add="+")
        root.bind("<FocusIn>", self._focus, add="+")

    def _resize(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)
        self._region()

    def _region(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox(self.window))
        if self.content.winfo_reqheight() <= self.canvas.winfo_height():
            self.canvas.yview_moveto(0)

    def _contains(self, widget):
        while widget is not None:
            if widget in (self.content, self.canvas):
                return True
            widget = getattr(widget, "master", None)
        return False

    def _wheel(self, event):
        if (not self._contains(event.widget) or event.widget.winfo_class() in ("Text", "Treeview", "TScrollbar")
                or self.content.winfo_reqheight() <= self.canvas.winfo_height()):
            return None
        if getattr(event, "num", None) in (4, 5):
            units = -3 if event.num == 4 else 3
        else:
            delta = getattr(event, "delta", 0)
            if not delta:
                return None
            units = (-1 if delta > 0 else 1) * max(1, abs(delta) // 120) * 3
        self.canvas.yview_scroll(units, "units")
        return "break"

    def _focus(self, event):
        if self._contains(event.widget):
            self.ensure_visible(event.widget)

    def ensure_visible(self, widget):
        """Keep keyboard navigation usable when a field or action lives below the viewport."""
        if not self._contains(widget):
            return
        top = self.canvas.canvasy(0)
        height = self.canvas.winfo_height()
        y = widget.winfo_rooty() - self.content.winfo_rooty()
        bottom = y + widget.winfo_height()
        if y < top:
            target = y - 8
        elif bottom > top + height:
            target = bottom - height + 8
        else:
            return
        self.canvas.yview_moveto(max(0, target) / max(1, self.content.winfo_height()))


def flow_buttons(frame):
    """Wrap existing button groups to the allocated width, including growing status lights."""
    if hasattr(frame, "flow_schedule"):
        for child in frame.winfo_children():
            child.bind("<Configure>", frame.flow_schedule, add="+")
        frame.flow_schedule()
        return
    pending = None

    def layout():
        nonlocal pending
        pending = None
        width = max(1, frame.winfo_width())
        x = y = line_height = 0
        for child in frame.winfo_children():
            child_width, child_height = child.winfo_reqwidth(), child.winfo_reqheight()
            if x and x + child_width > width:
                x, y, line_height = 0, y + line_height + 6, 0
            if child.winfo_manager() == "pack":
                child.pack_forget()
            child.place(x=x, y=y, width=child_width, height=child_height)
            x += child_width + 6
            line_height = max(line_height, child_height)
        frame.configure(height=max(1, y + line_height))

    def schedule(event=None):
        nonlocal pending
        if pending is None:
            pending = frame.after_idle(layout)

    frame.bind("<Configure>", schedule, add="+")
    frame.flow_schedule = schedule
    for child in frame.winfo_children():
        child.bind("<Configure>", schedule, add="+")
    schedule()
