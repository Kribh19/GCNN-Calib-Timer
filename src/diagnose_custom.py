import torch
from parse_real_design import build_graph
from interactive_train_real import real_graph_to_tensors, DEVICE

DESIGNS = ['uni_counter4b', 'cla8b', 'seq_detector', 'priority_enc']
DATA_DIR = r'C:\Krish\GCNN\data'

ckpt = torch.load(
    r'results\20260925_200206_train15_test7\model_checkpoint.pt',
    map_location='cpu', weights_only=False)
train_gates = set(ckpt['gate_types'])
print(f'Training gate vocab: {len(train_gates)} types')
print(f'Best epoch: {ckpt["best_epoch"]}, n_params: {ckpt["n_params"]:,}\n')

# Also show stats for a couple of training designs for comparison
TRAIN_SAMPLE = ['xtea', 'spm', 'aes192']

print('='*70)
print('CUSTOM DESIGNS')
print('='*70)
for d in DESIGNS:
    g = build_graph(f'{DATA_DIR}\\{d}', d)
    gate_types_in_design = set(g['gate_of_out'].values())
    unknown = gate_types_in_design - train_gates
    known = gate_types_in_design & train_gates

    n_pins = len(g['pins'])
    n_net_edges = len(g.get('net_edges', []))
    n_cell_edges = len(g.get('cell_edges', []))
    # get level depth from tensor
    tg = real_graph_to_tensors(g, ckpt['gate_types'])
    max_lv = int(tg['pin_level'].max().item())

    print(f'--- {d} ---')
    print(f'  Total pins         : {n_pins}')
    print(f'  Net edges          : {n_net_edges}')
    print(f'  Cell edges         : {n_cell_edges}')
    print(f'  Max topological lvl: {max_lv}')
    print(f'  Gate types used    : {len(gate_types_in_design)}  '
          f'({len(known)} known to model, {len(unknown)} UNKNOWN)')
    if unknown:
        print(f'  Unknown gates      : {sorted(unknown)}')
    # AT true stats
    at_vals = [v for v in tg['AT_true'].tolist() if v != 0.0]
    if at_vals:
        print(f'  AT_true range      : {min(at_vals):.3f} to {max(at_vals):.3f} ns  (mean {sum(at_vals)/len(at_vals):.3f})')
    print()

print('='*70)
print('TEST DESIGNS (for comparison)')
print('='*70)
for d in TRAIN_SAMPLE:
    g = build_graph(f'{DATA_DIR}\\{d}', d)
    gate_types_in_design = set(g['gate_of_out'].values())
    unknown = gate_types_in_design - train_gates
    n_pins = len(g['pins'])
    n_net_edges = len(g.get('net_edges', []))
    print(f'--- {d} (test design) ---')
    print(f'  Total pins         : {n_pins}')
    print(f'  Net edges          : {n_net_edges}')
    print(f'  Unknown gates      : {len(unknown)}')
    print()
