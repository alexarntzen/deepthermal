import torch
import torch.optim as optim
import numpy as np


def argmin(G, y_0, lr=0.1, epochs=1):
    y_opt = y_0.clone().detach().requires_grad_(True)
    optimizer = optim.LBFGS([y_opt], lr=float(lr), max_iter=50000, max_eval=50000, history_size=100,
                            line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)
    for _ in range(epochs):
        def closure():
            optimizer.zero_grad()
            g_forward = G(y_opt)
            print("loss: ", g_forward.item(), "value: ", y_opt.item())
            g_forward.backward()
            return g_forward

        optimizer.step(closure=closure)

    return y_opt.detach()
