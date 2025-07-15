import torch
import numpy as np

Q = torch.load("out/q_table_5k_best.pt")

print(f"{np.count_nonzero(Q.numpy())} non-zero elements ou of {Q.numel()} total elements.")
