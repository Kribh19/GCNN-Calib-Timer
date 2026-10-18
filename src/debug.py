# paste this as debug.py in src folder and run: python debug.py
import re, os, sys
sys.path.insert(0, r'C:\Krish\GCNN\src')
from paper_preprocess import parse_verilog, parse_liberty, find_files

vlog = r'C:\Krish\GCNN\data\xtea\xtea.synthesis_preroute.v'
lib  = r'C:\Krish\GCNN\data\techlib\sky130_fd_sc_hd__ff_n40C_1v95.lib'

instances, nets, ports = parse_verilog(vlog)
print(f'Ports: {len(ports)}')
print(f'Instances: {len(instances)}')
print(f'Nets: {len(nets)}')

# show first 3 instances
for i, (k,v) in enumerate(list(instances.items())[:3]):
    print(f'  inst: {k} -> {v}')

# show first 3 nets with drivers
driven = {k:v for k,v in nets.items() if v['driver']}
print(f'Nets with driver: {len(driven)}')
for i, (k,v) in enumerate(list(driven.items())[:3]):
    print(f'  net: {k} driver={v["driver"]} sinks={len(v["sinks"])}')

cell_db = parse_liberty(lib)
print(f'\nLiberty cells: {len(cell_db)}')

# check if xtea instances match liberty
matched = sum(1 for ct in instances.values() if ct in cell_db)
print(f'Instances matching Liberty: {matched}/{len(instances)}')

# check pin names for one cell
sample_ct = list(instances.values())[0]
print(f'\nSample cell type: {sample_ct}')
db = cell_db.get(sample_ct, {})
print(f'  Liberty pin_cap keys: {list(db.get("pin_cap",{}).keys())}')
print(f'  Liberty lut_delay keys: {list(db.get("lut_delay",{}).keys())[:3]}')