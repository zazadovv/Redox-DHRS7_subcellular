#!/usr/bin/env python3
"""
MSA_GUI.py
==========

A window for running the DHRS7 alignment without using a terminal.

The defaults are the published DHRS7 settings: the gene, the species compared,
the alignment options and the output location. Press Run and the build proceeds
exactly as `python dhrs7_alignment.py` would, writing the same output directory,
the same figure and the same interactive alignment browser.

Clearing the preset checkbox unlocks the gene field, and any Ensembl gene symbol
can be run through the same pipeline. The pinned ortholog records and the panel
annotations belong to DHRS7, so another gene resolves each species to its closest
ortholog and the frames and asterisks stop carrying meaning; the window says so
when the preset is off. This package remains a DHRS7 analysis -- the rest is for
looking around.

The build runs as a separate process and its output is streamed into the log
below, so the window stays responsive and can be closed without leaving a
half-finished run behind.

Only the standard library is used here (tkinter), so the window adds nothing to
the requirements of the analysis itself.

    python MSA_GUI.py
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

SCRIPT_DIR = Path(__file__).resolve().parent
DRIVER = SCRIPT_DIR / "dhrs7_alignment.py"
GENE = "DHRS7"

sys.path.insert(0, str(SCRIPT_DIR))
import dhrs7_alignment as driver  # noqa: E402

DEFAULT_SPECIES = ", ".join(driver.figure.DHRS7_SELECTED_ORDER)
REFERENCE_SPECIES = driver.REFERENCE_SPECIES

CANONICAL_RECORDS = [
    ("mouse", "Mus musculus", "ENSMUSP00000021512"),
    ("rat", "Rattus norvegicus", "ENSRNOP00000007645"),
    ("cattle", "Bos taurus", "ENSBTAP00000086519 / UniProt Q24K14"),
    ("zebrafish", "Danio rerio", "ENSDARP00000004163"),
    ("human (reference)", "Homo sapiens", "ENSP00000216500"),
]


class AlignmentWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.process: subprocess.Popen | None = None
        self.messages: "queue.Queue[str]" = queue.Queue()

        root.title(f"{GENE} cross-species alignment")
        root.geometry("880x720")
        root.minsize(760, 600)

        outer = ttk.Frame(root, padding=12)
        outer.pack(fill="both", expand=True)

        self.heading = ttk.Label(outer, text=f"{GENE} cross-species alignment",
                                 font=("Segoe UI", 15, "bold"))
        self.heading.pack(anchor="w")
        ttk.Label(outer, wraplength=830, foreground="#444",
                  text="Retrieves every ortholog from Ensembl, aligns them with MUSCLE, "
                       "attaches each species' AlphaFold model and draws the annotated panel. "
                       "Default settings are shown below; press Run to build.").pack(
            anchor="w", pady=(2, 10))

        self._build_settings(outer)
        self._build_records(outer)
        self._build_buttons(outer)
        self._build_log(outer)

        self.on_preset_toggle()
        self.gene_var.trace_add("write", lambda *_: self.sync_outdir())
        self.detect_muscle()
        self.poll_messages()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------------------------------------------------------- layout
    def _build_settings(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Settings", padding=10)
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)
        row = 0

        self.preset_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, variable=self.preset_var, command=self.on_preset_toggle,
                        text=f"Use the {GENE} preset"
                        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        ttk.Label(box, text="Gene").grid(row=row, column=0, sticky="w", pady=3)
        self.gene_var = tk.StringVar(value=GENE)
        self.gene_entry = ttk.Entry(box, textvariable=self.gene_var)
        self.gene_entry.grid(row=row, column=1, sticky="ew", padx=(8, 6))
        self.gene_note = ttk.Label(box, text="preset", foreground="#666")
        self.gene_note.grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(box, text="Species compared").grid(row=row, column=0, sticky="w", pady=3)
        self.species_var = tk.StringVar(value=DEFAULT_SPECIES)
        ttk.Entry(box, textvariable=self.species_var).grid(row=row, column=1, sticky="ew", padx=(8, 6))
        ttk.Label(box, text=f"{REFERENCE_SPECIES} is added as the reference row",
                  foreground="#666").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(box, text="Output folder").grid(row=row, column=0, sticky="w", pady=3)
        self.outdir_var = tk.StringVar(value=str(SCRIPT_DIR / f"{GENE}_Output"))
        ttk.Entry(box, textvariable=self.outdir_var).grid(row=row, column=1, sticky="ew", padx=(8, 6))
        ttk.Button(box, text="Browse...", command=self.pick_outdir).grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(box, text="MUSCLE").grid(row=row, column=0, sticky="w", pady=3)
        self.muscle_var = tk.StringVar()
        ttk.Entry(box, textvariable=self.muscle_var).grid(row=row, column=1, sticky="ew", padx=(8, 6))
        ttk.Button(box, text="Browse...", command=self.pick_muscle).grid(row=row, column=2, sticky="w")
        row += 1

        strip = ttk.Frame(box)
        strip.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(strip, text="Residues per block").pack(side="left")
        self.rpl_var = tk.StringVar(value="70")
        ttk.Spinbox(strip, from_=20, to=200, increment=10, width=6,
                    textvariable=self.rpl_var).pack(side="left", padx=(8, 20))

        ttk.Label(strip, text="Secondary structure").pack(side="left")
        self.ss_var = tk.StringVar(value="torsion")
        ttk.Radiobutton(strip, text="Torsion (phi/psi)", value="torsion",
                        variable=self.ss_var).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(strip, text="Torsion + H-bonding", value="hbond",
                        variable=self.ss_var).pack(side="left")
        row += 1

        self.figure_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, variable=self.figure_only_var,
                        text="Redraw the figure only, reusing an existing build (no network)"
                        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _build_records(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Records used for the panel", padding=8)
        box.pack(fill="x", pady=(10, 0))
        for label, latin, record in CANONICAL_RECORDS:
            line = ttk.Frame(box)
            line.pack(fill="x")
            ttk.Label(line, text=f"{label:<18}", width=18).pack(side="left")
            ttk.Label(line, text=latin, width=20, foreground="#333").pack(side="left")
            ttk.Label(line, text=record, foreground="#666").pack(side="left")
        self.records_note = ttk.Label(
            box, foreground="#666", wraplength=820, justify="left",
            text="Mouse, rat and zebrafish each return more than one ortholog from Ensembl; "
                 "the canonical translation is pinned so the comparison never picks up a "
                 "paralogous record.")
        self.records_note.pack(anchor="w", pady=(6, 0))

    # ---------------------------------------------------------------- preset
    def on_preset_toggle(self) -> None:
        """The preset fixes the gene and the species to the published set. Clearing
        it opens both up; the pinned records and the panel annotations belong to
        DHRS7, so anything else is exploratory."""
        preset = self.preset_var.get()
        self.gene_entry.configure(state="readonly" if preset else "normal")
        if preset:
            self.gene_var.set(GENE)
            self.species_var.set(DEFAULT_SPECIES)
            self.gene_note.configure(text="preset")
            self.records_note.configure(
                foreground="#666",
                text="Mouse, rat and zebrafish each return more than one ortholog from "
                     "Ensembl; the canonical translation is pinned so the comparison never "
                     "picks up a paralogous record.")
        else:
            self.gene_note.configure(text="any Ensembl gene symbol")
            self.records_note.configure(
                foreground="#8a5a00",
                text="Preset off. The records above are pinned for DHRS7 and do not apply to "
                     "another gene, which resolves each species to its closest ortholog "
                     "instead. The frames and asterisks drawn on the panel mark a "
                     "DHRS7-specific comparison and are not meaningful for other genes.")
        self.sync_outdir()

    def current_gene(self) -> str:
        return (self.gene_var.get() or GENE).strip().upper() or GENE

    def sync_outdir(self) -> None:
        """Keep the default output folder, the title and the heading in step with
        the gene, so the window never claims to be building something else."""
        gene = self.current_gene()
        current = Path(self.outdir_var.get())
        if current.parent == SCRIPT_DIR and current.name.endswith("_Output"):
            self.outdir_var.set(str(SCRIPT_DIR / f"{gene}_Output"))
        title = f"{gene} cross-species alignment"
        self.root.title(title)
        if hasattr(self, "heading"):
            self.heading.configure(text=title)

    def _build_buttons(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(12, 6))
        self.check_button = ttk.Button(bar, text="Check setup", command=self.on_check)
        self.check_button.pack(side="left")
        self.run_button = ttk.Button(bar, text="Run", command=self.on_run)
        self.run_button.pack(side="left", padx=6)
        self.stop_button = ttk.Button(bar, text="Stop", command=self.on_stop, state="disabled")
        self.stop_button.pack(side="left")
        self.results_button = ttk.Button(bar, text="Open results folder", command=self.open_results)
        self.results_button.pack(side="left", padx=(20, 0))
        self.browser_button = ttk.Button(bar, text="Open alignment browser", command=self.open_browser)
        self.browser_button.pack(side="left", padx=6)
        self.figure_button = ttk.Button(bar, text="Open figure", command=self.open_figure)
        self.figure_button.pack(side="left")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(parent, textvariable=self.status_var, foreground="#333").pack(anchor="w")
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill="x", pady=(4, 0))

    def _build_log(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Log", padding=6)
        box.pack(fill="both", expand=True, pady=(10, 0))
        self.log = tk.Text(box, height=14, wrap="none", background="#111827",
                           foreground="#e5e7eb", insertbackground="#e5e7eb",
                           font=("Consolas", 9))
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set, state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # ------------------------------------------------------------- behaviour
    def detect_muscle(self) -> None:
        found = driver.find_muscle()
        if found:
            self.muscle_var.set(str(found))
            self.write(f"MUSCLE found: {found}")
        else:
            self.muscle_var.set("")
            self.write("MUSCLE was not found. Use Browse to point at muscle.exe, "
                       "or install it (see the README).")

    def pick_outdir(self) -> None:
        chosen = filedialog.askdirectory(title="Output folder",
                                         initialdir=self.outdir_var.get() or str(SCRIPT_DIR))
        if chosen:
            self.outdir_var.set(chosen)

    def pick_muscle(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Locate the MUSCLE executable",
            filetypes=[("MUSCLE", "muscle.exe muscle*"), ("All files", "*.*")])
        if chosen:
            self.muscle_var.set(chosen)

    def build_command(self, check_only: bool) -> list[str]:
        command = [sys.executable, str(DRIVER)]
        if check_only:
            return command + ["--check"]  # gene is irrelevant to the setup check
        gene = (self.gene_var.get() or GENE).strip().upper() or GENE
        species = ",".join(part.strip() for part in self.species_var.get().split(",") if part.strip())
        command += ["--gene", gene,
                    "--species", species,
                    "--outdir", self.outdir_var.get(),
                    "--residues-per-line", self.rpl_var.get().strip() or "70",
                    "--ss", self.ss_var.get()]
        if self.figure_only_var.get():
            command.append("--figure-only")
        if self.muscle_var.get().strip():
            command += ["--muscle", self.muscle_var.get().strip()]
        return command

    def on_check(self) -> None:
        self.start(self.build_command(check_only=True), "Checking the setup ...")

    def on_run(self) -> None:
        if not self.species_var.get().strip():
            messagebox.showwarning("Nothing to compare",
                                   "Name at least one species besides the human reference.")
            return
        if not self.figure_only_var.get():
            proceed = messagebox.askokcancel(
                "Run the full build",
                f"This retrieves every {self.current_gene()} ortholog from Ensembl, UniProt "
                "and AlphaFold and takes about fifteen minutes.\n\nAnything already in the "
                "output folder is replaced.\n\nContinue?")
            if not proceed:
                return
        self.start(self.build_command(check_only=False), "Building ...")

    def start(self, command: list[str], status: str) -> None:
        if self.process is not None:
            messagebox.showinfo("Already running", "Wait for the current step to finish.")
            return
        self.status_var.set(status)
        self.progress.start(12)
        self.run_button.configure(state="disabled")
        self.check_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.write("")
        self.write("> " + " ".join(command))
        threading.Thread(target=self.worker, args=(command,), daemon=True).start()

    def worker(self, command: list[str]) -> None:
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            self.process = subprocess.Popen(
                command, cwd=str(SCRIPT_DIR), env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.messages.put(line.rstrip("\n"))
            code = self.process.wait()
        except Exception as exc:  # noqa: BLE001
            self.messages.put(f"failed to start: {exc}")
            code = 1
        finally:
            self.process = None
        self.messages.put(f"__done__{code}")

    def poll_messages(self) -> None:
        try:
            while True:
                item = self.messages.get_nowait()
                if item.startswith("__done__"):
                    self.finish(int(item[len("__done__"):]))
                else:
                    self.write(item)
        except queue.Empty:
            pass
        self.root.after(120, self.poll_messages)

    def finish(self, code: int) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.check_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if code == 0:
            self.status_var.set("Finished.")
        else:
            self.status_var.set(f"Stopped with exit code {code}. See the log above.")

    def on_stop(self) -> None:
        if self.process is not None:
            self.write("Stopping ...")
            try:
                self.process.terminate()
            except Exception:  # noqa: BLE001
                pass

    def write(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------------------------------------------------------------- opening
    def open_path(self, path: Path, what: str) -> None:
        if not path.exists():
            messagebox.showinfo("Not there yet", f"{what} does not exist yet:\n{path}")
            return
        webbrowser.open(path.as_uri())

    def open_results(self) -> None:
        path = Path(self.outdir_var.get())
        if not path.exists():
            messagebox.showinfo("Not there yet", f"No output folder yet:\n{path}")
            return
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])

    def open_browser(self) -> None:
        self.open_path(Path(self.outdir_var.get()) / "alignment_browser.html",
                       "The interactive alignment browser")

    def open_figure(self) -> None:
        gene = (self.gene_var.get() or GENE).strip().lower() or GENE.lower()
        self.open_path(Path(self.outdir_var.get()) / "plots" / f"{gene}_species_snapshot.svg",
                       "The figure")

    def on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askokcancel("Quit", "A build is still running. Stop it and quit?"):
                return
            self.on_stop()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if sys.platform.startswith("win") else "clam")
    except tk.TclError:
        pass
    AlignmentWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
