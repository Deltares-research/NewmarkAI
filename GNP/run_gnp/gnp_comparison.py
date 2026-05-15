#
# The option --num_workers specifies the number of dataloader workers
# for GNP training. The default is 0 (equivalent to 1). This works the
# best for a CPU machine. For a GPU machine, use a larger number.

import os
import time
import pickle
import torch
import argparse
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

from GNP.problems import *
from GNP.solver import  CG
from GNP.precond import *
from GNP.nn import ResGCN
from GNP.utils import scale_A_by_spectral_radius


def estimate_one_norm_condition_number(A):
    from scipy.sparse.linalg import onenormest, LinearOperator, splu

    lu = splu(A)
    n = A.shape[0]

    Ainv = LinearOperator(
        shape=(n, n),
        matvec=lambda x: lu.solve(x),
        rmatvec=lambda x: lu.solve(x)  # important for stability
    )

    Ainv_norm_est = onenormest(Ainv)
    A_norm_est = onenormest(A)

    cond_est = A_norm_est * Ainv_norm_est
    return cond_est

def one_norm_condition_number(A):
    import numpy.linalg as LA
    return LA.cond(A.todense(),1)

#-----------------------------------------------------------------------------
def main():

    # Setup and parameters
    restart = 10                # restart cycle in GMRES
    max_iters = 1000             # maximum number of GMRES iterations
    timeout = None              # timeout in seconds
    rtol = 1e-12                 # relative residual tolerance in GMRES
    training_data = 'x_mix'     # type of training data x, no_x, x_mix
    m = 40                      # Krylov subspace dimension for training data
    num_layers = 8              # number of layers in GNP
    embed = 16                  # embedding dimension in GNP
    hidden = 32                 # hidden dimension in MLPs in GNP
    drop_rate = 0.0             # dropout rate in GNP
    disable_scale_input = False # whether disable the scaling of inputs in GNP
    dtype = torch.float32       # training precision for GNP
    batch_size = 16             # batch size in training GNP
    grad_accu_steps = 1         # gradient accumulation steps in training GNP
    epochs = 2000               # number of epochs in training GNP
    lr = 1e-3                   # learning rate in training GNP
    weight_decay = 0            # weight decay in training GNP
    save_model = True           # whether save model
    hide_solver_bar = False     # whether hide progress bar in linear solver
    hide_training_bar = False   # whether hide progress bar in GNP training

    # Computing device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load problem
    res_dir = r"P:\11212716-026-newmarkai\GNP\figures"

    # case 1 is a matrix with 3.5k DOFS, case 2 has 60k DOFS
    case_nbr =1
    retrain_model = False
    match case_nbr:
        case 1:
            matrix_name = "3_5k_x_3_5k_matrix"
            file_name = r"P:\11212716-026-newmarkai\GNP\matrices_gnp\3_5k_x_3_5k_matrix.pickle"
            trained_model_fn = r"P:\11212716-026-newmarkai\GNP\trained_models\3_5k_matrix.pt"
            with open(file_name, "rb") as f:
                _, _, K, _, _ = pickle.load(f)

        case 2:
            matrix_name = "60k_x_60k_matrix"
            file_name = r"P:\11212716-026-newmarkai\GNP\matrices_gnp\60k_x_60k_matrix.pickle"
            trained_model_fn = r"P:\11212716-026-newmarkai\GNP\trained_models\60k_matrix.pt"
            with open(file_name, "rb") as f:
                matrices= pickle.load(f)
                K = matrices['K'].tocsc()

    print(estimate_one_norm_condition_number(K))

    plt.spy(K, markersize=0.5)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.grid(True)
    plt.savefig(f"{res_dir}/{matrix_name}_sparsity.png")
    plt.close()

    A = torch.sparse_csc_tensor(K.indptr, K.indices, K.data, K.shape,
                                dtype=torch.float64).to(device)


    # Normalize A to avoid hassles
    A = scale_A_by_spectral_radius(A)
    
    # Print problem information
    n = A.shape[0]
    print(f'\nMatrix A: name = scatter_rose, n = {n}, nnz = {A._nnz()}')
        
    # Right-hand side b
    x = gen_x_all_ones(n).to(device)
    b = A @ x
    del x

    out_file_prefix_with_path = r"P:\11212716-026-newmarkai\GNP\newly_trained_models"

    # Cg without preconditioner
    solver = CG.Cg()
    print('\nSolving linear system without preconditioner ...')
    solver.solve(     # dry run; timing is not accurate
        A, b, M=None, restart=restart, max_iters=max_iters,
        timeout=timeout, rtol=rtol, progress_bar=False)
    _, _, _, hist_rel_res, hist_time = solver.solve(
        A, b, M=None, restart=restart, max_iters=max_iters,
        timeout=timeout, rtol=rtol, progress_bar=not hide_solver_bar)
    print(f'Done. Final relative residual = {hist_rel_res[-1]:.4e}')

    # jacobi preconditioner
    M = JacobiPreconditioner(A)
    solver2 = CG.Cg()
    print('\nSolving linear system with Jacobi preconditioner ...')
    solver2.solve(     # dry run; timing is not accurate
        A, b, M=M, restart=restart, max_iters=max_iters,
        timeout=timeout, rtol=rtol, progress_bar=False)
    _, _, _, hist_rel_res_jac, hist_time_jac = solver2.solve(
        A, b, M=M, restart=restart, max_iters=max_iters,
        timeout=timeout, rtol=rtol, progress_bar=not hide_solver_bar)
    print(f'Done. Final relative residual = {hist_rel_res_jac[-1]:.4e}')

    #ilu preconditioner
    # M = ILU(A, None,False, None)
    # solver3 = CG.Cg()
    # print('\nSolving linear system with Ilu preconditioner ...')
    # solver3.solve(     # dry run; timing is not accurate
    #     A, b, M=M, restart=restart, max_iters=max_iters,
    #     timeout=timeout, rtol=rtol, progress_bar=False)
    # _, _, _, hist_rel_res_ilu, hist_time_ilu = solver3.solve(
    #     A, b, M=M, restart=restart, max_iters=max_iters,
    #     timeout=timeout, rtol=rtol, progress_bar=not hide_solver_bar)
    # print(f'Done. Final relative residual = {hist_rel_res_ilu[-1]:.4e}')

    # Cg with GNP: Train preconditioner
    print('\nTraining GNP ...')
    net = ResGCN(A, num_layers, embed, hidden, drop_rate,
                 scale_input=not disable_scale_input, dtype=dtype).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = None
    M = GNP(A, training_data, m, net, device)
    tic = time.time()

    num_workers = 8

    if not retrain_model:
        model_file = trained_model_fn
    else:

        hist_loss, best_loss, best_epoch, model_file = M.train(
            batch_size, grad_accu_steps, epochs, optimizer, scheduler,
            num_workers=num_workers,
            checkpoint_prefix_with_path=\
            out_file_prefix_with_path if save_model else None,
            progress_bar=not hide_training_bar)
        print(f'Done. Training time: {time.time()-tic} seconds')
        print(f'Loss: inital = {hist_loss[0]}, '
              f'final = {hist_loss[-1]}, '
              f'best = {best_loss}, epoch = {best_epoch}')
        if save_model:
            print(f'Best model saved in {model_file}')

        # Investigate training history of the preconditioner
        print('\nPlotting training history ...')
        plt.figure(1)
        plt.semilogy(hist_loss, label='train')
        plt.xlabel("Epochs")
        plt.ylabel("MAE Loss")

        full_path = f"{res_dir}/{matrix_name}_training.png"
        plt.grid(True)
        plt.savefig(full_path)
        plt.close()
        print(f'Figure saved in {full_path}')

    # Load the best checkpoint
    if model_file:
        print(f'\nLoading model from {model_file} ...')
        net.load_state_dict(torch.load(model_file, map_location=device))

        torch.set_float32_matmul_precision('high')  # or 'highest' for even better precision
        net = torch.compile(net, backend="eager")
        M = GNP(A, training_data, m, net, device)
        print('Done.')
    else:
        print('\nNo checkpoint is saved. Use model from the last epoch.')
            
    # Cg with GNP: Linear solve
    print('\nSolving linear system with GNP ...')
    warnings.filterwarnings('error')
    try:
        _, _, _, hist_rel_res_gnp, hist_time_gnp = solver.solve(
            A, b, M=M, restart=restart, max_iters=max_iters,
            timeout=timeout, rtol=rtol, progress_bar=not hide_solver_bar)
    except UserWarning as w:
        print('Warning:', w)
        print('Cg preconditioned by GNP fails')
        hist_rel_res_gnp = None
        hist_time_gnp = None
    except RuntimeError as e:
        print('RuntimeError:', e)
        print('Cg preconditioned by GNP fails')
        hist_rel_res_gnp = None
        hist_time_gnp = None
    else:
        print(f'Done. '
              f'Final relative residual = {hist_rel_res_gnp[-1]:.4e}')
    warnings.resetwarnings()

    # Investigate solution history
    print('\nPlotting solution history ...')
    plt.figure(2)
    plt.semilogy(hist_rel_res, color='C0', label='no precond')
    plt.semilogy(hist_rel_res_jac, color='C3', label='Jacobi')
    # plt.semilogy(hist_rel_res_ilu, color='C5', label='ILU')
    if hist_rel_res_gnp is not None:
        plt.semilogy(hist_rel_res_gnp, color='C7', label='GNP')

    plt.xlabel('Iterations')
    plt.ylabel('Relative Residual')
    plt.grid(True)
    plt.legend()

    full_path = f"{res_dir}/{matrix_name}_solver.png"
    plt.savefig(full_path)
    print(f'Figure saved in {full_path}')
    
    # Compare solution speed
    print('\nPlotting solution history (time to solution) ...')
    plt.figure(3)
    plt.semilogy(hist_time, hist_rel_res, color='C0', label='no precond')
    plt.semilogy(hist_time_jac, hist_rel_res_jac, color='C3', label='Jacobi')
    # plt.semilogy(hist_time_ilu, hist_rel_res_ilu, color='C5', label='ILU')
    if hist_rel_res_gnp is not None:
        plt.semilogy(hist_time_gnp, hist_rel_res_gnp, color='C7', label='GNP')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Relative Residual')
    plt.grid()
    plt.legend()

    full_path = f"{res_dir}/{matrix_name}_time.png"
    plt.savefig(full_path)
    print(f'Figure saved in {full_path}')
    

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
