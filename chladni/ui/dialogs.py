import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class PropertiesDialog(tk.Toplevel):
    def __init__(self, parent, simulator):
        super().__init__(parent)
        self.transient(parent)
        self.title("Properties")
        self.parent = parent # Keep a reference to the parent (main app window)
        self.simulator = simulator

        self.result = None # To store dialog result (if any, e.g. new values)

        body = ttk.Frame(self, padding="10 10 10 10")
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        self.buttonbox()

        self.grab_set() # Make dialog modal

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.geometry(f"+{parent.winfo_rootx()+50}+{parent.winfo_rooty()+50}")
        self.initial_focus.focus_set()
        self.wait_window(self)

    def body(self, master):
        ttk.Label(master, text="Image Width:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.width_var = tk.IntVar(value=self.simulator.width)
        self.width_entry = ttk.Entry(master, textvariable=self.width_var, width=10)
        self.width_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(master, text="Image Height:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.height_var = tk.IntVar(value=self.simulator.height)
        self.height_entry = ttk.Entry(master, textvariable=self.height_var, width=10)
        self.height_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(master, text="Wave Capacity:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.capacity_var = tk.IntVar(value=self.simulator.capacity)
        self.capacity_entry = ttk.Entry(master, textvariable=self.capacity_var, width=10)
        self.capacity_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)

        # Add a normalize checkbox
        ttk.Label(master, text="Normalize:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.normalize_var = tk.BooleanVar(value=self.simulator.normalize)
        self.normalize_check = ttk.Checkbutton(master, variable=self.normalize_var, text="")
        self.normalize_check.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)


        return self.width_entry # Initial focus

    def buttonbox(self):
        box = ttk.Frame(self)

        ok_button = ttk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        ok_button.pack(side=tk.LEFT, padx=5, pady=5)
        cancel_button = ttk.Button(box, text="Cancel", width=10, command=self.cancel)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        box.pack()

    def ok(self, event=None):
        if not self.validate():
            self.initial_focus.focus_set() # Put focus back
            return
        self.withdraw()
        self.update_idletasks()
        self.apply()
        self.cancel()

    def cancel(self, event=None):
        self.parent.focus_set()
        self.destroy()

    def validate(self):
        try:
            w = self.width_var.get()
            h = self.height_var.get()
            c = self.capacity_var.get()
            if not (10 <= w <= 16384 and 10 <= h <= 16384): # Consistent with engine limits
                messagebox.showerror("Validation Error", "Width and Height must be between 10 and 16384.")
                return False
            if not (1 <= c <= 1000): # Reasonable capacity limit
                messagebox.showerror("Validation Error", "Capacity must be between 1 and 1000.")
                return False
        except tk.TclError:
            messagebox.showerror("Validation Error", "Invalid input. Please enter numbers.")
            return False
        return True

    def apply(self):
        self.result = {
            "width": self.width_var.get(),
            "height": self.height_var.get(),
            "capacity": self.capacity_var.get(),
            "normalize": self.normalize_var.get()
        }

class AboutDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.transient(parent)
        self.title("About Chladni Plate Simulator")

        body = ttk.Frame(self, padding="10 10 10 10")
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        self.buttonbox()
        self.grab_set()
        if not self.initial_focus: self.initial_focus = self
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.geometry(f"+{parent.winfo_rootx()+70}+{parent.winfo_rooty()+70}")
        self.initial_focus.focus_set()
        self.wait_window(self)

    def body(self, master):
        ttk.Label(master, text="Chladni Plate Simulator", font=("Helvetica", 16, "bold")).pack(pady=10)
        ttk.Label(master, text="Version 1.0 (Python Tkinter Edition)").pack(pady=2)
        ttk.Label(master, text="Based on the original by Vasily Makarov.").pack(pady=2)
        ttk.Label(master, text="Python conversion by Jules (AI Agent).").pack(pady=2)
        return None # No specific focus needed here, or focus on OK button

    def buttonbox(self):
        box = ttk.Frame(self)
        ok_button = ttk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        ok_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.ok) # Escape also closes About
        box.pack()

    def ok(self, event=None):
        self.destroy()

    def cancel(self, event=None): # In case WM_DELETE_WINDOW is used
        self.destroy()
