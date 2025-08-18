# pump_gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
from pump_class import WX10Pump

DEFAULT_BAUD = 9600

class PumpGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WX10 Pump Controller")
        self.serial_port = None
        self.pump = None

        # --- Connection frame ---
        f_conn = ttk.LabelFrame(root, text="Connection")
        f_conn.pack(fill="x", padx=10, pady=8)

        ttk.Label(f_conn, text="Port:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.port_cb = ttk.Combobox(f_conn, width=20, values=self._list_ports())
        self.port_cb.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        if self.port_cb["values"]:
            self.port_cb.current(0)
        else:
            self.port_cb.set("/dev/ttyUSB0")  # fallback

        ttk.Label(f_conn, text="Baud:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.baud_cb = ttk.Combobox(f_conn, width=8, values=[9600, 19200, 38400, 57600, 115200])
        self.baud_cb.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.baud_cb.set(DEFAULT_BAUD)

        self.btn_refresh = ttk.Button(f_conn, text="Refresh Ports", command=self._refresh_ports)
        self.btn_refresh.grid(row=0, column=4, padx=5, pady=5)

        self.btn_connect = ttk.Button(f_conn, text="Connect", command=self.connect)
        self.btn_connect.grid(row=0, column=5, padx=5, pady=5)
        self.btn_disconnect = ttk.Button(f_conn, text="Disconnect", command=self.disconnect, state="disabled")
        self.btn_disconnect.grid(row=0, column=6, padx=5, pady=5)

        # --- Control frame ---
        f_ctrl = ttk.LabelFrame(root, text="Pump Control")
        f_ctrl.pack(fill="x", padx=10, pady=8)

        ttk.Label(f_ctrl, text="Address:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.addr_var = tk.IntVar(value=2)
        ttk.Spinbox(f_ctrl, from_=1, to=255, textvariable=self.addr_var, width=5).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(f_ctrl, text="RPM:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.rpm_var = tk.DoubleVar(value=50.0)
        ttk.Entry(f_ctrl, textvariable=self.rpm_var, width=10).grid(row=0, column=3, padx=5, pady=5, sticky="w")

        self.cw_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_ctrl, text="CW", variable=self.cw_var).grid(row=0, column=4, padx=5, pady=5)

        self.full_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f_ctrl, text="Full Speed", variable=self.full_var).grid(row=0, column=5, padx=5, pady=5)

        self.btn_start = ttk.Button(f_ctrl, text="Start", command=self.start_pump, state="disabled")
        self.btn_start.grid(row=1, column=0, padx=5, pady=8, sticky="ew", columnspan=2)

        self.btn_stop = ttk.Button(f_ctrl, text="Stop", command=self.stop_pump, state="disabled")
        self.btn_stop.grid(row=1, column=2, padx=5, pady=8, sticky="ew", columnspan=2)

        self.btn_state = ttk.Button(f_ctrl, text="Get State", command=self.get_state, state="disabled")
        self.btn_state.grid(row=1, column=4, padx=5, pady=8, sticky="ew", columnspan=2)

        # --- Status ---
        f_status = ttk.LabelFrame(root, text="Status")
        f_status.pack(fill="x", padx=10, pady=8)
        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(f_status, textvariable=self.status_var).pack(anchor="w", padx=8, pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------- Actions -------------
    def connect(self):
        port = self.port_cb.get().strip()
        try:
            baud = int(self.baud_cb.get())
        except ValueError:
            baud = DEFAULT_BAUD

        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=8,
                parity=serial.PARITY_EVEN,
                stopbits=1,
                timeout=1
            )
            self.serial_port.reset_output_buffer()
            # Initial pump instance with current address
            self.pump = WX10Pump(port=self.serial_port, address=self.addr_var.get())
            self._set_connected(True)
            self.status_var.set(f"Connected to {port} @ {baud} baud")
        except Exception as e:
            self.serial_port = None
            self.pump = None
            messagebox.showerror("Connect Error", str(e))
            self.status_var.set("Connection failed")

    def disconnect(self):
        try:
            if self.pump:
                self.pump.close()
        except Exception:
            pass
        self.serial_port = None
        self.pump = None
        self._set_connected(False)
        self.status_var.set("Disconnected")

    def start_pump(self):
        if not self._ensure_connected():
            return
        try:
            addr = int(self.addr_var.get())
            rpm = float(self.rpm_var.get())
            cw = bool(self.cw_var.get())
            full = bool(self.full_var.get())

            self.pump.set_address(addr)
            self.pump.set_speed(rpm, run=True, full_speed=full, cw=cw)
            self.status_var.set(f"Pump {addr}: RUN @ {rpm:.1f} rpm ({'CW' if cw else 'CCW'}"
                                f"{', FULL' if full else ''})")
        except Exception as e:
            messagebox.showerror("Start Error", str(e))

    def stop_pump(self):
        if not self._ensure_connected():
            return
        try:
            addr = int(self.addr_var.get())
            self.pump.set_address(addr)
            self.pump.stop()
            self.status_var.set(f"Pump {addr}: STOP")
        except Exception as e:
            messagebox.showerror("Stop Error", str(e))

    def get_state(self):
        if not self._ensure_connected():
            return
        try:
            addr = int(self.addr_var.get())
            self.pump.set_address(addr)
            state = self.pump.get_state()
            txt = (f"Pump {addr} → "
                   f"{'RUN' if state['running'] else 'STOP'}, "
                   f"{state['speed_rpm']:.1f} rpm, "
                   f"{state['direction']}, "
                   f"{'FULL' if state['full_speed'] else 'NORMAL'}")
            self.status_var.set(txt)
        except Exception as e:
            messagebox.showerror("State Error", str(e))

    # ------------- Helpers -------------
    def _set_connected(self, connected: bool):
        self.btn_connect.config(state="disabled" if connected else "normal")
        self.btn_disconnect.config(state="normal" if connected else "disabled")
        for btn in (self.btn_start, self.btn_stop, self.btn_state):
            btn.config(state="normal" if connected else "disabled")
        self.port_cb.config(state="disabled" if connected else "normal")
        self.baud_cb.config(state="disabled" if connected else "normal")

    def _ensure_connected(self):
        if self.pump is None or self.serial_port is None or not self.serial_port.is_open:
            messagebox.showwarning("Not Connected", "Connect to a serial port first.")
            return False
        return True

    def _list_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        return ports

    def _refresh_ports(self):
        self.port_cb["values"] = self._list_ports()
        if self.port_cb["values"]:
            self.port_cb.current(0)

    def on_close(self):
        try:
            if self.pump:
                self.pump.close()
        except Exception:
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    PumpGUI(root)
    root.mainloop()
