from parse_real_design import build_graph
for d in ['spm', 'synth_ram']:
    g = build_graph(f'C:/Krish/GCNN/data/{d}', d)
    print(d, ':', len(g['true_at_arrival']), 'real AT mapped')
