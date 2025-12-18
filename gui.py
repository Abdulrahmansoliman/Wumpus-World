from __future__ import annotations

import random
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Dict, Optional, Tuple

from wumpus.agent_prolog import PrologAgent
from wumpus.bridge import PrologBridge
from wumpus.world import Action, Direction, Percept, World

Position = Tuple[int, int]


class WumpusApp:
    """Tkinter GUI for the Wumpus World simulator."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Wumpus World")
        self.root.configure(bg="#0f172a")

        self.grid_size_var = tk.IntVar(value=4)
        self.pit_prob_var = tk.DoubleVar(value=0.2)
        self.seed_var = tk.StringVar(value="")  # blank => random seed
        self.mode_var = tk.StringVar(value="agent")
        self.reveal_var = tk.BooleanVar(value=True)

        self.agent_step_delay_ms = 350  # fixed delay; removed slider

        self.canvas_size = 520
        self.world: Optional[World] = None
        self.agent: Optional[PrologAgent] = None
        self.current_percept: Optional[Percept] = None
        self.visited_percepts: Dict[Position, Percept] = {}
        self.step_count = 0
        self.running = False
        self.after_handle: Optional[str] = None

        self._build_layout()

    def _build_layout(self) -> None:
        header = tk.Label(
            self.root,
            text="Wumpus World",
            font=("Segoe UI", 18, "bold"),
            fg="#e2e8f0",
            bg="#0f172a",
        )
        header.pack(pady=(12, 6))

        controls = tk.Frame(self.root, bg="#0f172a")
        controls.pack(fill="x", padx=12)

        tk.Label(controls, text="Mode", fg="#e2e8f0", bg="#0f172a").grid(row=0, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(controls, self.mode_var, "agent", "human").grid(row=0, column=1, sticky="we", pady=(6, 0))

        tk.Checkbutton(
            controls,
            text="Reveal world",
            variable=self.reveal_var,
            fg="#e2e8f0",
            selectcolor="#1f2937",
            bg="#0f172a",
            activebackground="#0f172a",
        ).grid(row=0, column=2, sticky="w", pady=(6, 0))

        start_btn = tk.Button(
            controls,
            text="Start Game",
            command=self.prompt_and_start_game,
            bg="#22c55e",
            fg="#0b1720",
            activebackground="#16a34a",
            relief="flat",
            padx=12,
            pady=6,
        )
        start_btn.grid(row=1, column=0, columnspan=3, sticky="we", pady=(6, 0))

        for i in range(3):
            controls.columnconfigure(i, weight=1)

        body = tk.Frame(self.root, bg="#0f172a")
        body.pack(fill="both", expand=True, padx=12, pady=12)

        self.canvas = tk.Canvas(
            body,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#0b1220",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew")

        info_frame = tk.Frame(body, bg="#0f172a")
        info_frame.grid(row=0, column=1, sticky="new", padx=(12, 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=0)
        body.rowconfigure(1, weight=1)

        self.status_label = tk.Label(
            info_frame,
            text="Status: idle",
            fg="#e2e8f0",
            bg="#0f172a",
            anchor="w",
            justify="left",
            font=("Segoe UI", 10),
        )
        self.status_label.pack(fill="x", pady=(0, 6))

        self.percept_label = tk.Label(
            info_frame,
            text="Percepts: -",
            fg="#e2e8f0",
            bg="#0f172a",
            anchor="w",
            justify="left",
            font=("Segoe UI", 10),
        )
        self.percept_label.pack(fill="x", pady=(0, 6))

        legend = tk.LabelFrame(info_frame, text="Legend", bg="#0f172a", fg="#e2e8f0")
        legend.pack(fill="x", pady=(6, 6))

        def legend_row(color: str, text: str):
            row = tk.Frame(legend, bg="#0f172a")
            row.pack(fill="x", pady=1)
            swatch = tk.Canvas(row, width=14, height=14, bg="#0f172a", highlightthickness=0)
            swatch.pack(side="left", padx=(4, 6))
            swatch.create_rectangle(2, 2, 12, 12, fill=color, outline=color)
            tk.Label(row, text=text, bg="#0f172a", fg="#e2e8f0").pack(side="left")

        legend_row("#ef4444", "Pit (red)")
        legend_row("#a855f7", "Wumpus (purple)")
        legend_row("#eab308", "Gold (yellow)")
        legend_row("#38bdf8", "Agent (blue)")
        legend_row("#22c55e", "Start / Safe highlight (green)")
        legend_row("#e2e8f0", "B/S/G letters = Breeze/Stench/Glitter on visited cells")

        self.log_text = tk.Text(
            info_frame,
            height=16,
            width=34,
            bg="#111827",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            highlightthickness=0,
            relief="flat",
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, pady=(6, 0))
        self.log_text.insert("end", "Logs will appear here.\n")
        self.log_text.config(state="disabled")

        actions_frame = tk.Frame(body, bg="#0f172a")
        actions_frame.grid(row=1, column=1, sticky="sew", padx=(12, 0), pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)
        actions_frame.columnconfigure(1, weight=1)
        actions_frame.columnconfigure(2, weight=1)

        self.action_buttons = {}
        btn_specs = [
            ("Forward", Action.FORWARD, 0, 1),
            ("Turn Left", Action.TURN_LEFT, 1, 0),
            ("Turn Right", Action.TURN_RIGHT, 1, 2),
            ("Grab", Action.GRAB, 2, 0),
            ("Shoot", Action.SHOOT, 2, 1),
            ("Climb", Action.CLIMB, 2, 2),
        ]
        for label, action, r, c in btn_specs:
            btn = tk.Button(
                actions_frame,
                text=label,
                command=lambda a=action: self.on_human_action(a),
                bg="#1d4ed8",
                fg="#e2e8f0",
                activebackground="#1e3a8a",
                relief="flat",
                padx=10,
                pady=6,
                state="disabled",
            )
            btn.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            self.action_buttons[action] = btn

    def _add_labeled_entry(self, parent: tk.Frame, label: str, var, row: int, col: int, width: int) -> None:
        box = tk.Frame(parent, bg="#0f172a")
        box.grid(row=row, column=col, sticky="we", padx=(0, 8))
        tk.Label(box, text=label, fg="#e2e8f0", bg="#0f172a").grid(row=0, column=0, sticky="w")
        entry = tk.Entry(
            box,
            textvariable=var,
            width=width,
            relief="flat",
            bg="#111827",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
        )
        entry.grid(row=1, column=0, sticky="we")
        box.columnconfigure(0, weight=1)

    def prompt_and_start_game(self) -> None:
        """Ask for grid size + seed before starting."""
        dlg = tk.Toplevel(self.root)
        dlg.title("New Game Setup")
        dlg.configure(bg="#0f172a")
        dlg.transient(self.root)
        dlg.grab_set()

        size_var = tk.IntVar(value=4)
        seed_var = tk.StringVar(value="")
        pit_var = tk.DoubleVar(value=float(self.pit_prob_var.get()))

        tk.Label(dlg, text="Grid size (max 4x4):", bg="#0f172a", fg="#e2e8f0").pack(anchor="w", padx=12, pady=(10, 2))
        size_menu = tk.OptionMenu(dlg, size_var, 3, 4)
        size_menu.pack(fill="x", padx=12)

        tk.Label(dlg, text="Seed (blank = random):", bg="#0f172a", fg="#e2e8f0").pack(anchor="w", padx=12, pady=(10, 2))
        tk.Entry(dlg, textvariable=seed_var, bg="#111827", fg="#e5e7eb", insertbackground="#e5e7eb").pack(fill="x", padx=12)

        tk.Label(dlg, text="Pit probability (0.0–0.6):", bg="#0f172a", fg="#e2e8f0").pack(anchor="w", padx=12, pady=(10, 2))
        tk.Entry(dlg, textvariable=pit_var, bg="#111827", fg="#e5e7eb", insertbackground="#e5e7eb").pack(fill="x", padx=12)

        btns = tk.Frame(dlg, bg="#0f172a")
        btns.pack(fill="x", padx=12, pady=12)

        def on_cancel():
            dlg.destroy()

        def on_start():
            # apply choices
            self.grid_size_var.set(int(size_var.get()))
            self.pit_prob_var.set(float(pit_var.get()))
            self.seed_var.set(seed_var.get().strip())
            dlg.destroy()
            self.start_game()

        tk.Button(btns, text="Cancel", command=on_cancel, bg="#334155", fg="#e2e8f0", relief="flat").pack(side="right", padx=6)
        tk.Button(btns, text="Start", command=on_start, bg="#22c55e", fg="#0b1720", relief="flat").pack(side="right")

    def _resolve_seed(self) -> int:
        s = self.seed_var.get().strip()
        if s:
            return int(s)
        seed = random.randint(0, 10**9)
        self.seed_var.set(str(seed))
        return seed

    def start_game(self) -> None:
        try:
            size = int(self.grid_size_var.get())
            pit_prob = float(self.pit_prob_var.get())
            seed = self._resolve_seed()
        except ValueError:
            messagebox.showerror("Invalid input", "Grid size, pit probability, and seed must be numeric.")
            return

        if size not in (3, 4):
            messagebox.showerror("Invalid grid size", "Grid size must be 3 or 4 (max 4x4 due to Prolog complexity).")
            return
        if pit_prob < 0 or pit_prob > 0.6:
            messagebox.showerror("Invalid pit probability", "Pit probability must be between 0.0 and 0.6.")
            return

        self._cancel_after()
        self.world = World(size=size, pit_probability=pit_prob, seed=seed)
        kb_path = Path(__file__).parent / "wumpus" / "prolog" / "knowledge_base.pl"
        self.agent = None
        if self.mode_var.get() == "agent":
            self.agent = PrologAgent(grid_size=size, bridge=PrologBridge(kb_path))
            self.agent.reset()
        self.current_percept = self.world.initial_percept()
        self.visited_percepts = {self.world.agent_pos: self.current_percept}
        self.step_count = 0
        self.running = True
        self._set_buttons_enabled(self.mode_var.get() == "human")
        self._log("New game started.")
        self._update_status()
        self._draw_world()
        if self.mode_var.get() == "agent" and self.agent:
            self.after_handle = self.root.after(self.agent_step_delay_ms, self._agent_step)

    def _cancel_after(self) -> None:
        if self.after_handle is not None:
            try:
                self.root.after_cancel(self.after_handle)
            except Exception:
                pass
            self.after_handle = None

    def _agent_step(self) -> None:
        if not self.running or self.world is None or self.agent is None or self.current_percept is None:
            return
        action = self.agent.act(self.current_percept)
        result = self.world.step(action)
        self.current_percept = result.percept
        self.visited_percepts[self.world.agent_pos] = self.current_percept
        self.step_count += 1
        self._log(f"Agent: {action.value}")
        self._update_status(result)
        self._draw_world()
        if result.terminated:
            self.running = False
            messagebox.showinfo("Game over", f"Outcome: {result.outcome}, gold: {result.has_gold}, score: {result.score}")
            return
        self.after_handle = self.root.after(self.agent_step_delay_ms, self._agent_step)

    def on_human_action(self, action: Action) -> None:
        if not self.running or self.world is None or self.current_percept is None:
            return
        result = self.world.step(action)
        self.current_percept = result.percept
        self.visited_percepts[self.world.agent_pos] = self.current_percept
        self.step_count += 1
        self._log(f"You: {action.value}")
        self._update_status(result)
        self._draw_world()
        if result.terminated:
            self.running = False
            self._set_buttons_enabled(False)
            messagebox.showinfo("Game over", f"Outcome: {result.outcome}, gold: {result.has_gold}, score: {result.score}")

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for btn in self.action_buttons.values():
            btn.config(state=state)

    def _update_status(self, result: Optional[object] = None) -> None:
        outcome = getattr(result, "outcome", None)
        has_gold = getattr(result, "has_gold", False)
        status = f"Status: step {self.step_count}"
        if self.world:
            status += f" | score={self.world.score}"
        if outcome:
            status += f" | outcome={outcome}"
        if has_gold:
            status += " | carrying gold"
        self.status_label.config(text=status)

        if self.current_percept:
            p = self.current_percept
            percept_str = (
                f"Percepts: Breeze={p.breeze}, Stench={p.stench}, Glitter={p.glitter}, "
                f"Bump={p.bump}, Scream={p.scream}"
            )
            self.percept_label.config(text=percept_str)

    def _draw_world(self) -> None:
        if self.world is None:
            return
        size = self.world.size
        cell = self.canvas_size / size
        self.canvas.delete("all")

        visited = set(self.visited_percepts.keys())

        # Grid background
        for x in range(size):
            for y in range(size):
                pos = (x + 1, y + 1)
                x0 = x * cell
                y0 = (size - y - 1) * cell
                x1 = x0 + cell
                y1 = y0 + cell
                base_color = "#334155" if pos in visited else "#1f2937"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=base_color, outline="#0f172a")

        # Hazards and gold (optionally revealed)
        reveal = self.reveal_var.get()
        if reveal:
            for pit in self.world.pits:
                self._draw_circle(pit, cell, fill="#ef4444", outline="#b91c1c")
            if self.world.wumpus_alive:
                self._draw_circle(self.world.wumpus_pos, cell, fill="#a855f7", outline="#7e22ce")
            if not self.world.wumpus_alive:
                self._draw_cross(self.world.wumpus_pos, cell, color="#a855f7")
            if not self.world.has_gold and self.world.gold_pos:
                self._draw_diamond(self.world.gold_pos, cell, fill="#eab308", outline="#f59e0b")

        # Start cell highlight
        self._draw_start(cell)

        # Agent
        self._draw_agent(cell)

        # Percept markers on visited squares
        for (vx, vy), p in self.visited_percepts.items():
            tags = []
            if p.breeze:
                tags.append("B")
            if p.stench:
                tags.append("S")
            if p.glitter:
                tags.append("G")
            if tags:
                cx = (vx - 0.5) * cell
                cy = (size - vy + 0.5) * cell
                self.canvas.create_text(
                    cx,
                    cy,
                    text=" ".join(tags),
                    fill="#e2e8f0",
                    font=("Segoe UI", 12, "bold"),
                )

        # Grid lines
        for i in range(size + 1):
            pos = i * cell
            self.canvas.create_line(pos, 0, pos, self.canvas_size, fill="#0f172a")
            self.canvas.create_line(0, pos, self.canvas_size, pos, fill="#0f172a")

    def _draw_start(self, cell: float) -> None:
        if self.world is None:
            return
        sx, sy = self.world.start
        self._draw_rect((sx, sy), cell, outline="#22c55e")

    def _draw_agent(self, cell: float) -> None:
        if self.world is None:
            return
        ax, ay = self.world.agent_pos
        cx = (ax - 0.5) * cell
        cy = (self.world.size - ay + 0.5) * cell
        direction = self.world.agent_dir
        size = cell * 0.35
        if direction == Direction.NORTH:
            points = [cx, cy - size, cx - size, cy + size, cx + size, cy + size]
        elif direction == Direction.SOUTH:
            points = [cx, cy + size, cx - size, cy - size, cx + size, cy - size]
        elif direction == Direction.EAST:
            points = [cx + size, cy, cx - size, cy - size, cx - size, cy + size]
        else:
            points = [cx - size, cy, cx + size, cy - size, cx + size, cy + size]
        self.canvas.create_polygon(points, fill="#38bdf8", outline="#0ea5e9", width=2)

    def _draw_rect(self, pos, cell: float, outline: str) -> None:
        x, y = pos
        x0 = (x - 1) * cell + 4
        y0 = (self.world.size - y) * cell + 4
        x1 = x0 + cell - 8
        y1 = y0 + cell - 8
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=outline, width=2)

    def _draw_circle(self, pos, cell: float, fill: str, outline: str) -> None:
        x, y = pos
        r = cell * 0.2
        cx = (x - 0.5) * cell
        cy = (self.world.size - y + 0.5) * cell
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline=outline, width=2)

    def _draw_cross(self, pos, cell: float, color: str) -> None:
        x, y = pos
        cx = (x - 0.5) * cell
        cy = (self.world.size - y + 0.5) * cell
        r = cell * 0.25
        self.canvas.create_line(cx - r, cy - r, cx + r, cy + r, fill=color, width=3)
        self.canvas.create_line(cx - r, cy + r, cx + r, cy - r, fill=color, width=3)

    def _draw_diamond(self, pos, cell: float, fill: str, outline: str) -> None:
        x, y = pos
        cx = (x - 0.5) * cell
        cy = (self.world.size - y + 0.5) * cell
        r = cell * 0.22
        points = [cx, cy - r, cx + r, cy, cx, cy + r, cx - r, cy]
        self.canvas.create_polygon(points, fill=fill, outline=outline, width=2)

    def _log(self, message: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = WumpusApp()
    app.run()


if __name__ == "__main__":
    main()
