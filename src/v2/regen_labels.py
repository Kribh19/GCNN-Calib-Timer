"""
regen_labels.py -- regenerate <design>.at.sdf with the SAME flow as the DAC'22 benchmark
(OpenSTA, ff library as min, ss library as max, routed SPEF, the design's SDC), so custom
designs get labels in exactly the benchmark format:
    4-channel delays (min::max for rise and fall), and per-pin AT / SLEW / RAT for every
    pin and port.

    python regen_labels.py uni_counter4b cla8b ...        # overwrite (old file kept as .at.sdf.orig)
    python regen_labels.py --check xtea                   # write to a temp file and compare with
                                                          # the benchmark's own file, change nothing

Needs OpenSTA (`sta`) inside WSL.  The benchmark was produced with OpenSTA 2.3.0; a newer
version gives slightly different numbers (about 0.5 % on xtea).
"""
import os
import shutil
import subprocess
import sys

import numpy as np

DATA_DIR = r"C:\Krish\GCNN\data"

TCL = r"""
read_liberty -min ../techlib/sky130_fd_sc_hd__ff_n40C_1v95.lib
read_liberty -max ../techlib/sky130_fd_sc_hd__ss_100C_1v60.lib
read_verilog %(d)s.synthesis_preroute.v
link_design %(d)s
read_spef %(spef)s
read_sdc %(sdc)s
write_sdf -digits 9 %(out)s
proc gp {obj name} {
  if {[catch {set v [get_property $obj $name]}]} { return "" }
  if {$v == "INF" || $v == "-INF" || $v == "inf" || $v == "-inf"} { return "" }
  return $v
}
proc rat {at slack sign} {
  if {$at == "" || $slack == ""} { return "" }
  return [expr {$at + $sign * $slack}]
}
set f [open %(out)s a]
puts $f "(ARRIVALTIMES"
set objs {}
foreach port [get_ports *] { lappend objs [list [get_full_name $port] [sta::get_port_pin $port]] }
foreach pin [get_pins -hierarchical *] { lappend objs [list [get_full_name $pin] $pin] }
foreach pair $objs {
  set n [lindex $pair 0]
  set o [lindex $pair 1]
  set a_er [gp $o arrival_min_rise]; set a_lr [gp $o arrival_max_rise]
  set a_ef [gp $o arrival_min_fall]; set a_lf [gp $o arrival_max_fall]
  puts $f " (AT $n ($a_er\:\:$a_lr) ($a_ef\:\:$a_lf))"
  puts $f " (SLEW $n ([gp $o slew_min_rise]\:\:[gp $o slew_max_rise]) ([gp $o slew_min_fall]\:\:[gp $o slew_max_fall]))"
  # setup slack = required - arrival ; hold slack = arrival - required
  set r_er [rat $a_er [gp $o slack_min_rise] -1]; set r_lr [rat $a_lr [gp $o slack_max_rise] 1]
  set r_ef [rat $a_ef [gp $o slack_min_fall] -1]; set r_lf [rat $a_lf [gp $o slack_max_fall] 1]
  puts $f " (RAT $n ($r_er\:\:$r_lr) ($r_ef\:\:$r_lf))"
}
puts $f ")"
close $f
puts "DUMPED [llength $objs] objects"
flush stdout
exit
"""


def find(d, suffix):
    for f in sorted(os.listdir(d)):
        if f.endswith(suffix):
            return f
    raise FileNotFoundError(f"no *{suffix} in {d}")


def regen(design, out_name):
    d = os.path.join(DATA_DIR, design)
    tcl = TCL % {"d": design, "spef": find(d, ".spef"), "sdc": find(d, ".sdc"), "out": out_name}
    with open(os.path.join(d, "_regen.tcl"), "w", newline="\n") as f:
        f.write(tcl)
    wsl_dir = "/mnt/" + d[0].lower() + d[2:].replace("\\", "/")
    r = subprocess.run(["wsl", "bash", "-lc", f"cd '{wsl_dir}' && sta -no_init _regen.tcl 2>&1 | grep -v 'Warning 198' | tail -5"],
                       capture_output=True, text=True)
    os.remove(os.path.join(d, "_regen.tcl"))
    ok = "DUMPED" in r.stdout
    print(f"[{design}] {'ok' if ok else 'FAILED'}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()}")
    return ok


def compare(design, new_name):
    from build_v2 import parse_at_sdf
    d = os.path.join(DATA_DIR, design)
    _, _, at0, sl0, rat0, _ = parse_at_sdf(os.path.join(d, f"{design}.at.sdf"))
    iop, inter, at1, sl1, rat1, _ = parse_at_sdf(os.path.join(d, new_name))
    for nm, a, b in (("AT", at0, at1), ("SLEW", sl0, sl1), ("RAT", rat0, rat1)):
        keys = [k for k in a if k in b]
        x = np.array([a[k] for k in keys], np.float64); y = np.array([b[k] for k in keys], np.float64)
        m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e20) & (np.abs(y) < 1e20)
        r2 = 1 - ((x[m] - y[m]) ** 2).sum() / ((x[m] - x[m].mean()) ** 2).sum()
        print(f"   {nm:<5} pins in both {len(keys)} of {len(a)} | values compared {int(m.sum())} | "
              f"R2 regenerated vs benchmark {r2:.6f} | mean abs diff {np.abs(x[m] - y[m]).mean():.4f} ns")
    print(f"   regenerated file: {len(iop)} cell arcs, {len(inter)} net edges")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--check":
        for dn in args[1:]:
            if regen(dn, "_check.at.sdf"):
                compare(dn, "_check.at.sdf")
            p = os.path.join(DATA_DIR, dn, "_check.at.sdf")
            if os.path.exists(p):
                os.remove(p)
    else:
        for dn in args:
            p = os.path.join(DATA_DIR, dn, f"{dn}.at.sdf")
            if os.path.exists(p) and not os.path.exists(p + ".orig"):
                shutil.copy2(p, p + ".orig")
            regen(dn, f"{dn}.at.sdf")
