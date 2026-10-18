"""
paper_preprocess.py — v4 (final, all bugs fixed)
=================================================
Bugs fixed vs v3:
  B1 — Bus port bits: 'data_in1[31]' not matched as PI driver
       Fix: extract bus base name and match bit indices as PI drivers
  B2 — Escaped identifiers \\state[0] etc. in net names  
       Fix: allow [ ] in net name pattern
  B3 — Feat 0 wrong for bus port bit pins
       Fix: strip [N] when checking inst against ports dict
  B4 — Feat 1 (is_output) uses Liberty output pins which need Liberty loaded
       Fix: also check if port name matches known output convention (X,Y,Q,Z)
       as a fallback when Liberty not loaded

10 node features matching the paper EXACTLY:
  0  is_primary_IO    1.0 if pin is a primary input or output port
  1  is_output_pin    1.0 if this is an output pin (drives a net)
  2  dist_left        (x - die_x0) / die_w
  3  dist_right       (die_x1 - x) / die_w
  4  dist_bottom      (y - die_y0) / die_h
  5  dist_top         (die_y1 - y) / die_h
  6  pin_cap_EarlyRise from Liberty capacitance (or 0 if not in Liberty)
  7  pin_cap_EarlyFall from Liberty capacitance
  8  pin_cap_LateRise  from Liberty capacitance
  9  pin_cap_LateFall  from Liberty capacitance
"""
import os, re, json, argparse
import torch

LUT_K       = 7
N_LUTS      = 8
EF_CELL_LEN = N_LUTS*(1+2*LUT_K) + N_LUTS*LUT_K*LUT_K + 4  # 516


# ================================================================ FILE FINDER
def find_files(design_dir, design_name, lib_arg):
    found = set(os.listdir(design_dir))
    result = {}

    def pick(candidates, label, required=True):
        for c in candidates:
            if c in found:
                result[label] = os.path.join(design_dir, c)
                return True
        result[label] = None
        if required:
            print(f'  MISSING {label}: tried {candidates}')
        return False

    pick([f'{design_name}.synthesis_preroute.v', f'{design_name}.v'], 'vlog')
    pick([f'20-{design_name}.def', f'{design_name}.def'] +
         sorted([f for f in found if f.endswith('.def')], reverse=True)[:1], 'def_')
    pick([f'{design_name}.sdf'], 'sdf')
    pick([f'{design_name}.at.sdf'], 'at_sdf', required=False)
    pick([f'20-{design_name}.spef', f'{design_name}.spef'], 'spef', required=False)

    lib_candidates = [f for f in found if f.endswith('.lib')]
    if lib_candidates:
        result['lib'] = os.path.join(design_dir, lib_candidates[0])
    elif lib_arg and os.path.isfile(lib_arg):
        result['lib'] = lib_arg
    elif lib_arg and os.path.isdir(lib_arg):
        for f in os.listdir(lib_arg):
            if f.endswith('.lib'):
                result['lib'] = os.path.join(lib_arg, f); break
    else:
        result['lib'] = None
    return result


# ================================================================ PARSERS
def parse_minmax(s):
    """Parse 'min::max' or 'min:typ:max' -> (min_val, max_val)."""
    s = s.strip().strip('()')
    parts = s.split(':')
    vals = []
    for p in parts:
        p = p.strip()
        if p:
            try: vals.append(float(p))
            except: pass
    if not vals: return 0.0, 0.0
    if len(vals) == 1: return vals[0], vals[0]
    return vals[0], vals[-1]


def _base_name(net):
    """'data_in1[31]' -> 'data_in1', '_0001_' -> '_0001_'"""
    return re.sub(r'\[\d+\]$', '', net).lstrip('\\')


def parse_verilog(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)

    SKIP = {'VPWR','VGND','VNB','VPB'}

    # ---- ports (scalar and bus) ----
    # scalar: input clock;   bus: input [31:0] data_in1;
    ports = {}   # base_name -> 'input'|'output'
    for m in re.finditer(
            r'\b(input|output)\b\s+(?:wire\s+)?(?:\[\d+:\d+\]\s+)?(\w+)', text):
        pname = m.group(2)
        if pname not in SKIP:
            ports[pname] = m.group(1)

    instances, nets = {}, {}

    # ---- cell instances ----
    # Allow net names with [ ] for bus bits and \ for escaped identifiers
    NET_PAT = re.compile(r'\.(\w+)\s*\(\s*(\\?[\w\[\]\\]+)\s*\)')

    for m in re.finditer(r'(sky130_\w+)\s+(\w+)\s*\((.*?)\);', text, re.S):
        ctype, iname, conn = m.group(1), m.group(2), m.group(3)
        instances[iname] = ctype
        for cm in NET_PAT.finditer(conn):
            port = cm.group(1)
            net  = cm.group(2).strip()
            if not net or _base_name(net) in SKIP or net in SKIP:
                continue
            nets.setdefault(net, {'driver': None, 'sinks': []})
            if re.match(r'^[XYQZSCq]', port):
                nets[net]['driver'] = (iname, port)
            else:
                nets[net]['sinks'].append((iname, port))

    # ---- primary inputs drive their net ----
    # Handle both scalar ('clock') and bus-bit ('data_in1[31]') nets
    for net_name in list(nets.keys()):
        base = _base_name(net_name)
        if base in ports and ports[base] == 'input':
            if nets[net_name]['driver'] is None:
                nets[net_name]['driver'] = (net_name, net_name)

    # Also add scalar port nets that appear only in the port list
    for pname, pdir in ports.items():
        if pname not in nets:
            nets[pname] = {'driver': None, 'sinks': []}
        if pdir == 'input' and nets[pname]['driver'] is None:
            nets[pname]['driver'] = (pname, pname)

    return instances, nets, ports


def parse_def(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    m = re.search(r'UNITS DISTANCE MICRONS (\d+)', text)
    dbu = int(m.group(1)) if m else 1000
    die = (0.0, 0.0, 1000.0, 1000.0)
    m = re.search(
        r'DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)', text)
    if m:
        die = tuple(float(m.group(i))/dbu for i in range(1,5))
    placements = {}
    comp = re.search(r'COMPONENTS(.*?)END COMPONENTS', text, re.S)
    if comp:
        for m in re.finditer(
                r'-\s+(\S+)\s+\S+\s*\+\s*(?:PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)',
                comp.group(1)):
            placements[m.group(1)] = (float(m.group(2))/dbu, float(m.group(3))/dbu)
    pins_sec = re.search(r'PINS(.*?)END PINS', text, re.S)
    if pins_sec:
        for m in re.finditer(
                r'-\s+(\S+)\s+.*?PLACED\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)',
                pins_sec.group(1), re.S):
            placements[m.group(1)] = (float(m.group(2))/dbu, float(m.group(3))/dbu)
    return placements, die, dbu


def parse_sdf(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    cell_delays, net_delays = {}, {}

    for block_m in re.finditer(
            r'\(CELL\s*\(CELLTYPE\s*"[^"]*"\s*\)\s*\(INSTANCE\s*(\w+)\s*\)'
            r'(.*?)(?=\(CELL|\Z)', text, re.S):
        inst, block = block_m.group(1), block_m.group(2)
        for m in re.finditer(
                r'\(IOPATH\s+(\S+)\s+(\S+)\s+\(([^)]+)\)\s*(?:\(([^)]+)\))?\s*\)', block):
            in_p, out_p = m.group(1), m.group(2)
            er, lr = parse_minmax(m.group(3))
            ef, lf = parse_minmax(m.group(4)) if m.group(4) else (er, lr)
            cell_delays[(inst, in_p, out_p)] = [er, ef, lr, lf]

    for m in re.finditer(
            r'\(INTERCONNECT\s+(\S+)\s+(\S+)\s+\(([^)]+)\)\s*(?:\(([^)]+)\))?\s*\)',
            text):
        src, dst = m.group(1), m.group(2)
        er, lr = parse_minmax(m.group(3))
        ef, lf = parse_minmax(m.group(4)) if m.group(4) else (er, lr)
        si, sp = src.rsplit('/', 1) if '/' in src else (src, src)
        di, dp = dst.rsplit('/', 1) if '/' in dst else (dst, dst)
        net_delays[(si, sp, di, dp)] = [er, ef, lr, lf]

    return cell_delays, net_delays


def parse_at_sdf(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    arrivals, rats, slews = {}, {}, {}
    SKIP = {'VGND','VPWR','VNB','VPB'}

    def parse_entry(m):
        pin_str = m.group(1).strip()
        inst, port = pin_str.rsplit('/', 1) if '/' in pin_str else (pin_str, pin_str)
        if inst in SKIP or port in SKIP: return None
        er, lr = parse_minmax(m.group(2))
        ef, lf = parse_minmax(m.group(3)) if m.group(3) else (er, lr)
        vals = [er, ef, lr, lf]
        if any(abs(v) > 1e20 for v in vals): vals = [0.0]*4
        return inst, port, vals

    for m in re.finditer(
            r'\(AT\s+(\S+)\s+\(([^)]+)\)\s*(?:\(([^)]+)\))?\s*\)', text):
        r = parse_entry(m)
        if r: arrivals[(r[0], r[1])] = r[2]
    for m in re.finditer(
            r'\(SLEW\s+(\S+)\s+\(([^)]+)\)\s*(?:\(([^)]+)\))?\s*\)', text):
        r = parse_entry(m)
        if r: slews[(r[0], r[1])] = r[2]
    for m in re.finditer(
            r'\(RAT\s+(\S+)\s+\(([^)]+)\)\s*(?:\(([^)]+)\))?\s*\)', text):
        r = parse_entry(m)
        if r: rats[(r[0], r[1])] = r[2]

    return arrivals, rats, slews


def parse_liberty(path):
    print(f'  Parsing Liberty: {os.path.basename(path)} ...', end=' ', flush=True)
    with open(path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//.*', '', text)
    cell_db = {}

    def find_blocks(txt, kw):
        results = []
        pat = re.compile(kw + r'\s*\("?([^")\s]*)"?\)\s*\{', re.S)
        for m in pat.finditer(txt):
            hdr = m.group(1).strip()
            s = m.end(); depth = 1; i = s
            while i < len(txt) and depth:
                if txt[i] == '{':   depth += 1
                elif txt[i] == '}': depth -= 1
                i += 1
            results.append((hdr, txt[s:i-1]))
        return results

    def to_floats(s):
        return [float(x) for x in
                re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', s)]

    def get_lut(body, kw):
        m = re.search(kw + r'\s*\([^)]*\)\s*\{(.*?)\}', body, re.S)
        if not m:
            return [0.0]*LUT_K, [0.0]*LUT_K, [0.0]*LUT_K*LUT_K
        c = m.group(1)
        ix = re.search(r'index_1\s*\("([^"]+)"\)', c)
        iy = re.search(r'index_2\s*\("([^"]+)"\)', c)
        vv = re.findall(r'values\s*\("([^"]+)"\)', c)
        ax = (to_floats(ix.group(1)) + [0.0]*LUT_K)[:LUT_K] if ix else [0.0]*LUT_K
        ay = (to_floats(iy.group(1)) + [0.0]*LUT_K)[:LUT_K] if iy else [0.0]*LUT_K
        vals = []
        for v in vv: vals.extend(to_floats(v))
        vals = (vals + [0.0]*LUT_K*LUT_K)[:LUT_K*LUT_K]
        return ax, ay, vals

    for ch, cb in find_blocks(text, 'cell'):
        cname = ch.strip()
        cell_db[cname] = {'pin_cap':{}, 'lut_delay':{}, 'lut_slew':{},
                          'slew_axis':{}, 'load_axis':{}}
        for ph, pb in find_blocks(cb, 'pin'):
            pname = ph.strip()
            if not pname: continue
            cm = re.search(r'capacitance\s*:\s*([\d.eE+-]+)', pb)
            cap = float(cm.group(1)) if cm else 0.0
            cell_db[cname]['pin_cap'][pname] = [cap]*4
            for _, tb in find_blocks(pb, 'timing'):
                rm = re.search(r'related_pin\s*:\s*"([^"]+)"', tb)
                if not rm: continue
                in_pin = rm.group(1).strip()
                key = (in_pin, pname)
                ax_r, ay_r, vr  = get_lut(tb, 'cell_rise')
                ax_f, ay_f, vf  = get_lut(tb, 'cell_fall')
                _,    _,    vsr = get_lut(tb, 'rise_transition')
                _,    _,    vsf = get_lut(tb, 'fall_transition')
                cell_db[cname]['lut_delay'][key] = [vr,vf,vr,vf,vr,vf,vr,vf]
                cell_db[cname]['lut_slew'][key]  = [vsr,vsf,vsr,vsf,vsr,vsf,vsr,vsf]
                cell_db[cname]['slew_axis'][key]  = [ax_r]*8
                cell_db[cname]['load_axis'][key]  = [ay_r]*8

    print(f'{len(cell_db)} cells')
    return cell_db


# ============================================================= GRAPH BUILDER
def build_graph(design_name, files, cell_db, gate_vocab):
    instances, nets, ports = parse_verilog(files['vlog'])
    placements, die, dbu   = parse_def(files['def_'])
    cell_del_sdf, net_del_sdf = parse_sdf(files['sdf'])
    arrivals, rats, slews  = ({},{},{}) if not files['at_sdf'] \
                              else parse_at_sdf(files['at_sdf'])

    die_x0, die_y0, die_x1, die_y1 = die
    die_w = max(die_x1-die_x0, 1.0)
    die_h = max(die_y1-die_y0, 1.0)

    for ct in instances.values():
        gate_vocab.setdefault(ct, len(gate_vocab))

    # output pin set from Liberty (used for feat 1 and cell edges)
    out_pins = {ct: set(op for (_,op) in db.get('lut_delay',{}).keys())
                for ct, db in cell_db.items()}

    # ---- pin index built from Verilog connections ----
    pin_list, pin_idx = [], {}
    def add(inst, port):
        k = (inst, port)
        if k not in pin_idx:
            pin_idx[k] = len(pin_list)
            pin_list.append(k)

    for p in ports: add(p, p)          # scalar ports
    for net_name, info in nets.items():
        if info['driver']:
            add(info['driver'][0], info['driver'][1])
        for si, sp in info['sinks']:
            add(si, sp)

    N = len(pin_list)
    nf      = torch.zeros(N, 10)
    n_ats   = torch.zeros(N, 4)
    n_slews = torch.zeros(N, 4)
    n_nd    = torch.zeros(N, 4)
    n_rats  = torch.zeros(N, 4)
    n_endpt = torch.zeros(N, 1)

    def get_xy(inst):
        if inst in placements: return placements[inst]
        # bus-bit port: 'data_in1[31]' -> look up 'data_in1'
        base = _base_name(inst)
        if base in placements: return placements[base]
        return (die_x0 + die_w*0.5, die_y0 + die_h*0.5)

    for idx, (inst, port) in enumerate(pin_list):
        # feat 0: is_primary_IO
        # B3 fix: check both exact name and base name for bus bits
        base_inst = _base_name(inst)
        nf[idx, 0] = 1.0 if (inst in ports or base_inst in ports) else 0.0

        # feat 1: is_output_pin (drives a net)
        is_out = (inst in ports and ports.get(inst) == 'output') or \
                 (base_inst in ports and ports.get(base_inst) == 'output')
        if not is_out and inst in instances:
            ct = instances[inst]
            # from Liberty
            is_out = port in out_pins.get(ct, set())
            # fallback: sky130 convention
            if not is_out:
                is_out = bool(re.match(r'^[XYQZSCq]', port))
        nf[idx, 1] = 1.0 if is_out else 0.0

        # feats 2-5: die boundary distances
        x, y = get_xy(inst)
        nf[idx, 2] = (x-die_x0)/die_w
        nf[idx, 3] = (die_x1-x)/die_w
        nf[idx, 4] = (y-die_y0)/die_h
        nf[idx, 5] = (die_y1-y)/die_h

        # feats 6-9: pin capacitance from Liberty (0 if not found)
        ct   = instances.get(inst, '')
        cap4 = cell_db.get(ct, {}).get('pin_cap', {}).get(port, [0.0]*4)
        nf[idx, 6:10] = torch.tensor(cap4[:4])

        # labels
        key = (inst, port)
        if key in arrivals: n_ats[idx]   = torch.tensor(arrivals[key])
        if key in rats:     n_rats[idx]  = torch.tensor(rats[key])
        if key in slews:    n_slews[idx] = torch.tensor(slews[key])

    # ---- net edges ----
    net_src_l, net_dst_l, net_ef_l = [], [], []
    for net_name, info in nets.items():
        drv = info['driver']
        if drv is None: continue
        di, dp = drv
        if (di,dp) not in pin_idx: continue
        d_idx = pin_idx[(di,dp)]
        xd, yd = get_xy(di)
        for si, sp in info['sinks']:
            if (si,sp) not in pin_idx: continue
            s_idx = pin_idx[(si,sp)]
            xs, ys = get_xy(si)
            net_src_l.append(d_idx); net_dst_l.append(s_idx)
            net_ef_l.append([(xs-xd)/die_w, (ys-yd)/die_h])
            nd4 = net_del_sdf.get((di,dp,si,sp), [0.0]*4)
            n_nd[s_idx] = torch.tensor(nd4)

    # ---- cell edges ----
    cell_src_l, cell_dst_l, cell_ef_l, cell_cd_l = [], [], [], []
    for inst, ct in instances.items():
        db = cell_db.get(ct, {})
        for (in_pin, out_pin), tbl_d in db.get('lut_delay', {}).items():
            if (inst,in_pin)  not in pin_idx: continue
            if (inst,out_pin) not in pin_idx: continue
            i_idx = pin_idx[(inst,in_pin)]
            o_idx = pin_idx[(inst,out_pin)]
            s_ax = db.get('slew_axis',{}).get((in_pin,out_pin), [[0.0]*7]*8)
            l_ax = db.get('load_axis',{}).get((in_pin,out_pin), [[0.0]*7]*8)
            tbl_d_v = tbl_d
            ef = torch.zeros(EF_CELL_LEN)
            for t in range(N_LUTS):
                b = t*(1+2*LUT_K)
                ef[b] = 1.0
                ef[b+1:b+1+LUT_K]         = torch.tensor(s_ax[t][:LUT_K])
                ef[b+1+LUT_K:b+1+2*LUT_K] = torch.tensor(l_ax[t][:LUT_K])
            for t in range(N_LUTS):
                b = 120+t*LUT_K*LUT_K
                ef[b:b+LUT_K*LUT_K] = torch.tensor(tbl_d_v[t][:LUT_K*LUT_K])
            cd4 = cell_del_sdf.get((inst,in_pin,out_pin), [0.0]*4)
            ef[512:516] = torch.tensor(cd4)
            cell_src_l.append(i_idx); cell_dst_l.append(o_idx)
            cell_ef_l.append(ef);     cell_cd_l.append(cd4)

    # endpoints = pins with no outgoing net edge
    has_out = set(net_src_l)
    for idx in range(N):
        if idx not in has_out: n_endpt[idx,0] = 1.0

    e_cell_delays = (torch.tensor(cell_cd_l, dtype=torch.float32)
                     if cell_cd_l else torch.zeros(0,4))

    return {
        'nf':                nf.float(),
        'net_src':           torch.tensor(net_src_l, dtype=torch.long),
        'net_dst':           torch.tensor(net_dst_l, dtype=torch.long),
        'net_ef':            torch.tensor(net_ef_l, dtype=torch.float32)
                             if net_ef_l else torch.zeros(0,2),
        'cell_src':          torch.tensor(cell_src_l, dtype=torch.long),
        'cell_dst':          torch.tensor(cell_dst_l, dtype=torch.long),
        'cell_ef':           torch.stack(cell_ef_l).float()
                             if cell_ef_l else torch.zeros(0,EF_CELL_LEN),
        'n_ats':             n_ats.float(),
        'n_slews':           n_slews.float(),
        'n_net_delays':      n_nd.float(),
        'n_rats':            n_rats.float(),
        'n_is_timing_endpt': n_endpt.float(),
        'e_cell_delays':     e_cell_delays,
        'die_area':          list(die),
    }


# ====================================================================== MAIN
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root',  default=r'C:\Krish\GCNN\data')
    ap.add_argument('--output_dir', default=r'C:\Krish\GCNN\paper_preprocessed_data')
    ap.add_argument('--lib',        default='')
    ap.add_argument('--designs',    default='')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.designs:
        design_list = [d.strip() for d in args.designs.split(',') if d.strip()]
    else:
        design_list = sorted(
            d for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
            and d not in ('techlib','data'))

    print(f'Designs   : {design_list}')
    print(f'Output dir: {args.output_dir}\n')

    cell_db = {}; lib_used = None
    for d in design_list:
        f = find_files(os.path.join(args.data_root, d), d, args.lib)
        if f['lib']:
            cell_db = parse_liberty(f['lib']); lib_used = f['lib']; break
    if not lib_used and args.lib:
        cell_db = parse_liberty(args.lib); lib_used = args.lib
    if not cell_db:
        print('[WARNING] Liberty not found -- feats 6-9 will be 0, no cell edges')

    gate_vocab = {}; ok = []; fail = []

    for design_name in design_list:
        design_dir = os.path.join(args.data_root, design_name)
        out_path   = os.path.join(args.output_dir, f'{design_name}.pt')
        print(f'\n[{design_name}]')
        files = find_files(design_dir, design_name, args.lib)
        if not files['vlog'] or not files['def_'] or not files['sdf']:
            print('  SKIPPED'); fail.append((design_name,'missing')); continue
        try:
            g = build_graph(design_name, files, cell_db, gate_vocab)
            N      = g['nf'].shape[0]
            E_net  = g['net_src'].shape[0]
            E_cell = g['cell_src'].shape[0]
            nz_at  = (g['n_ats'].abs()>1e-9).any(1).sum().item()
            nz_rat = (g['n_rats'].abs()>1e-9).any(1).sum().item()
            nz_nd  = (g['n_net_delays'].abs()>1e-9).any(1).sum().item()
            nz_cd  = (g['e_cell_delays'].abs()>1e-9).any(1).sum().item() \
                     if g['e_cell_delays'].shape[0]>0 else 0
            # feat sanity
            feat0_nonzero = (g['nf'][:,0]>0.5).sum().item()
            feat1_nonzero = (g['nf'][:,1]>0.5).sum().item()
            print(f'  Pins={N}  Net-edges={E_net}  Cell-edges={E_cell}')
            print(f'  Labels: AT={nz_at}  RAT={nz_rat}  net-delay={nz_nd}  cell-delay={nz_cd}')
            print(f'  Feats: is_IO={feat0_nonzero}  is_out={feat1_nonzero}')
            if E_net==0:  print('  [WARN] Net-edges=0')
            if E_cell==0: print('  [WARN] Cell-edges=0 (Liberty needed)')
            torch.save(g, out_path)
            print(f'  SAVED -> {out_path}')
            ok.append(design_name)
        except Exception as e:
            import traceback; print(f'  ERROR: {e}'); traceback.print_exc()
            fail.append((design_name, str(e)))

    meta = {'gate_vocab':gate_vocab,'n_gates':len(gate_vocab),'liberty':lib_used}
    with open(os.path.join(args.output_dir,'meta.json'),'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n{"="*55}')
    print(f'Done.  OK={len(ok)}  Failed={len(fail)}')
    if ok:   print(f'  OK    : {ok}')
    if fail: print(f'  Failed: {[d for d,_ in fail]}')
    print(f'  Gate vocab: {len(gate_vocab)} cell types')
    print(f'\nNext -- run all 21 designs then upload paper_preprocessed_data to Drive')
    print(f'Then on Colab:')
    print(f'  python train.py --graphs_dir paper_preprocessed_data --checkpoint run1 --seed 1')