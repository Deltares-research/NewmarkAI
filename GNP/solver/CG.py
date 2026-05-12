import time
import torch
import numpy as np
from tqdm import tqdm
from warnings import warn



class Cg():
    def solve(self, A, b, M=None,restart=None, x0=None, max_iters=100,
              timeout=None, rtol=1e-8, progress_bar=True):



        if progress_bar:
            if timeout is None:
                pbar = tqdm(total=max_iters, desc='Solve')
                pbar.update()
            else:
                pbar = tqdm(desc='Solve')
                pbar.update()

        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0

        norm_b = torch.linalg.norm(b)
        hist_abs_res = []
        hist_rel_res = []
        hist_time = []



        # Initial residual
        r = b - A @ x
        if M is not None:
            z = M.apply(r)
        else:
            z = r.clone()
        tic = time.time()
        p = z.clone()
        rz_old = torch.dot(r, z)

        abs_res = torch.linalg.norm(r)
        rel_res = abs_res / norm_b
        hist_abs_res.append(abs_res.item())
        hist_rel_res.append(rel_res.item())
        hist_time.append(time.time() - tic)

        iters = 0

        while True:
            Ap = A @ p
            alpha = rz_old / torch.dot(p, Ap)

            x = x + alpha * p
            r = r - alpha * Ap

            abs_res = torch.linalg.norm(r)
            rel_res = abs_res / norm_b
            hist_abs_res.append(abs_res.item())
            hist_rel_res.append(rel_res.item())
            hist_time.append(time.time() - tic)

            iters += 1

            if (rel_res < rtol) or \
               (timeout is None and iters == max_iters) or \
               (timeout is not None and hist_time[-1] >= timeout):
                break

            if M is not None:
                z = M.apply(r)
                # M = None
            else:
                z = r.clone()

            rz_new = torch.dot(r, z)
            beta = rz_new / rz_old

            p = z + beta * p
            rz_old = rz_new

            if progress_bar:
                pbar.update()

        if progress_bar:
            pbar.close()

        return x, iters, hist_abs_res, hist_rel_res, hist_time
