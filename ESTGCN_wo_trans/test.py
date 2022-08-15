import torch
import numpy as np
a=[torch.tensor(1).item(),torch.tensor(2).item()]
b=[]
b.append(a)
halt_time=np.array(b)
print(np.mean(halt_time))
