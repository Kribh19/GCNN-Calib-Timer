import torch
d = torch.load(r'C:\Krish\GCNN\paper_preprocessed_data\xtea.pt', map_location='cpu', weights_only=False)
endpts = d['n_is_timing_endpt'].squeeze() > 0.5
at = d['n_ats'][:,0] > 0
print("endpoints:", endpts.sum().item())
print("endpoints with nonzero AT:", (endpts & at).sum().item())
print("endpoints with zero AT:", (endpts & ~at).sum().item())