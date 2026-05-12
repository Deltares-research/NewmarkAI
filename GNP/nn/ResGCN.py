import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from GNP.utils import scale_A_by_spectral_radius


#-----------------------------------------------------------------------------
# An MLP layer.
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, num_layers, hidden, drop_rate,
                 use_batchnorm=False, is_output_layer=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.is_output_layer = is_output_layer

        self.lin = nn.ModuleList()
        self.lin.append( nn.Linear(in_dim, hidden) )
        for i in range(1, num_layers-1):
            self.lin.append( nn.Linear(hidden, hidden) )
        self.lin.append( nn.Linear(hidden, out_dim) )
        if use_batchnorm:
            self.batchnorm = nn.ModuleList()
            for i in range(0, num_layers-1):
                self.batchnorm.append( nn.BatchNorm1d(hidden) )
            if not is_output_layer:
                self.batchnorm.append( nn.BatchNorm1d(out_dim) )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, R):                              # R: (*, in_dim)
        assert len(R.shape) >= 2
        for i in range(self.num_layers):
            R = self.lin[i](R)                            # (*, hidden)
            if i != self.num_layers-1 or not self.is_output_layer:
                if self.use_batchnorm:
                    shape = R.shape
                    R = R.view(-1, shape[-1])
                    R = self.batchnorm[i](R)
                    R = R.view(shape)
                R = self.dropout(F.relu(R))
                                                          # (*, out_dim)
        return R
    

#-----------------------------------------------------------------------------
# A GCN layer.
class GCNConv(nn.Module):

    def __init__(self, AA, in_dim, out_dim):
        super().__init__()
        self.AA = AA  # normalized A
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim, bias=False)


    def forward(self, R):                         # R: (n, batch_size, in_dim)
        assert len(R.shape) == 3
        n, batch_size, in_dim = R.shape
        assert in_dim == self.in_dim
        if in_dim > self.out_dim:
            R = self.fc(R)                           # (n, batch_size, out_dim)
            R = R.view(n, batch_size * self.out_dim) # (n, batch_size * out_dim)
            R = self.AA @ R                          # (n, batch_size * out_dim)
            R = R.view(n, batch_size, self.out_dim)  # (n, batch_size, out_dim)
        else:
            R = R.view(n, batch_size * in_dim)       # (n, batch_size * in_dim)
            R = self.AA @ R                          # (n, batch_size * in_dim)
            R = R.view(n, batch_size, in_dim)        # (n, batch_size, in_dim)
            R = self.fc(R)                           # (n, batch_size, out_dim)
        return R


#-----------------------------------------------------------------------------
# GCN with residual connections.
class ResGCN(nn.Module):
    
    def __init__(self, A, num_layers, embed, hidden, drop_rate,
                 scale_input=True, dtype=torch.float32):
        # A: float64, already on device.
        #
        # For graph convolution, A will be normalized and cast to
        # lower precision and named AA.
        
        super().__init__()
        self.dtype = dtype # used by GNP.precond.GNP
        self.num_layers = num_layers
        self.embed = embed
        self.scale_input = scale_input

        # Note: scale_A_by_spectral_radius() has been called when
        # defining the problem; hence, it is redundant. We keep the
        # code here to leave open the possibility of normalizing A in
        # another manner.
        self.AA = scale_A_by_spectral_radius(A).to(dtype)

        self.mlp_initial = MLP(1, embed, 4, hidden, drop_rate)
        self.mlp_final = MLP(embed, 1, 4, hidden, drop_rate,
                             is_output_layer=True)
        self.gconv = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(num_layers):
            self.gconv.append( GCNConv(self.AA, embed, embed) )
            self.skip.append( nn.Linear(embed, embed) )
            self.batchnorm.append( nn.BatchNorm1d(embed) )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, r):                        # r: (n, batch_size)
        assert len(r.shape) == 2
        n, batch_size = r.shape
        if self.scale_input:
            scaling = torch.linalg.vector_norm(r, dim=0) / np.sqrt(n)
            r = r / scaling  # scaling
        r = r.view(n, batch_size, 1)                # (n, batch_size, 1)
        R = self.mlp_initial(r)                     # (n, batch_size, embed)

        for i in range(self.num_layers):
            R = self.gconv[i](R) + self.skip[i](R)  # (n, batch_size, embed)
            R = R.view(n * batch_size, self.embed)  # (n * batch_size, embed)
            R = self.batchnorm[i](R)                # (n * batch_size, embed)
            R = R.view(n, batch_size, self.embed)   # (n, batch_size, embed)
            R = self.dropout(F.relu(R))             # (n, batch_size, embed)

        z = self.mlp_final(R)                       # (n, batch_size, 1)
        z = z.view(n, batch_size)                   # (n, batch_size)
        if self.scale_input:
            z = z * scaling  # scaling back
        return z

#
# class SpdGCN(nn.Module):
#     def __init__(self, A, embed, dtype=torch.float32):
#         super().__init__()
#         # 1. We strictly allow NO Non-linearities (ReLU, Tanh)
#         #    to preserve the Linear property required by CG.
#         # 2. We define only one "half" of the network (The 'L' factor).
#
#
#
#         self.A = A.to_sparse_coo()
#         self.embed = embed
#         self.dtype = dtype
#         # Simple graph convolution weights: W1, W2
#         self.W1 = nn.Parameter(torch.randn(1, embed, dtype=self.dtype) * 0.01)
#         self.W2 = nn.Parameter(torch.randn(embed, embed, dtype=self.dtype) * 0.01)
#
#         self.epsilon = 1e-6
#         self.dtype = torch.float32
#
#     def safe_sparse_mm(self, A, x):
#         # 1. Ensure x is 2D and Contiguous
#         orig_shape = x.shape
#         if x.ndim > 2:
#             x = x.reshape(orig_shape[0], -1)
#
#         x = x.contiguous().to(self.dtype)
#
#         # 2. Use a 'try-except' fallback to CPU for this operation if CUDA fails
#         # or use the more stable COO mm.
#
#         res = torch.sparse.mm(A, x)
#
#
#         # 3. Restore 3D shape if necessary
#         if len(orig_shape) > 2:
#             res = res.reshape(orig_shape)
#         return res
#
#     def forward_L(self, x_latent):
#         x = self.safe_sparse_mm(self.A, x_latent)
#         x = x @ self.W2
#         x = x @ self.W1.T
#         x = x.squeeze(-1)
#         return self.safe_sparse_mm(self.A, x)
#
#     def forward_L_transpose(self, b):
#         x = self.safe_sparse_mm(self.A, b)
#         x = x.unsqueeze(-1) @ self.W1
#         x = self.safe_sparse_mm(self.A, x)
#         x = x @ self.W2.T
#         return x
#
#     def forward(self, b):
#         # Handle Batch vs Node dimension
#         is_transposed = False
#         if b.shape[0] != self.A.shape[0]:
#             b = b.T
#             is_transposed = True
#
#         latent = self.forward_L_transpose(b)
#         out = self.forward_L(latent)
#
#         res = out + self.epsilon * b
#         return res.T if is_transposed else res
def get_sparse_diagonal(A):
    n = A.shape[0]
    # Convert to COO to access row/col indices easily
    # This is a shallow copy of the values, so it's memory efficient
    A_coo = A.to_sparse_coo()
    A_coo = A_coo.coalesce()
    indices = A_coo.indices()
    values = A_coo.values()

    # Mask where row index == column index
    mask = (indices[0] == indices[1])

    # Create the diagonal tensor
    diag = torch.zeros(n, device=A.device, dtype=A.dtype)
    # Scatter the diagonal values into the correct positions
    diag.scatter_(0, indices[0][mask], values[mask])
    return diag



def extract_block_diagonal(A, block_size):
    n = A.shape[0]
    blocks = []

    for i in range(0, n, block_size):
        blk = A[i:i+block_size, i:i+block_size].to_dense()
        blocks.append(blk)

    return torch.stack(blocks)  # (num_blocks, b, b)

class LearnableJacobi(nn.Module):
    def __init__(self, A,block_size=10, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype  # used by GNP.precond.GNP
        self.block_size = block_size
        # 1. Extract the diagonal of A and store it as a buffer
        # if A.is_sparse:
        #     # For sparse matrices, we extract indices where row == col
        #     # This is a general way to get the diagonal of a sparse tensor
        #     n = A.shape[0]
        #     indices = A._indices()
        #     values = A._values()
        #     mask = (indices[0] == indices[1])
        #     diag_values = torch.zeros(n, device=A.device, dtype=A.dtype)
        #     diag_values[indices[0][mask]] = values[mask]
        #     self.register_buffer('D', diag_values.view(-1, 1))
        # else:
        #     self.register_buffer('D', torch.diag(A).view(-1, 1))

        n = A.shape[0]
        assert n % block_size == 0, "n must be divisible by block_size"

        num_blocks = n // block_size
        self.num_blocks = num_blocks

        # ---- extract block diagonals ----
        blocks = []
        A = A.to_sparse_coo()
        A = A.coalesce()
        indices = A.indices()
        values = A.values()

        n = A.shape[0]
        b = block_size
        num_blocks = n // b

        blocks = torch.zeros(
            num_blocks, b, b,
            device=A.device,
            dtype=values.dtype
        )

        rows, cols = indices
        for r, c, v in zip(rows, cols, values):
            bi = r // b
            bj = c // b
            if bi == bj:
                i = r % b
                j = c % b
                blocks[bi, i, j] = v

        blocks = blocks.to(dtype)
        # blocks = torch.stack(blocks).to(dtype)

        # ---- Cholesky factorization (SPD-safe) ----
        block_chol = torch.linalg.cholesky(blocks)

        self.register_buffer("block_chol", block_chol)

        # ---- learnable scalar per block (log-space) ----
        self.w = nn.Parameter(torch.zeros(num_blocks, dtype=dtype))

    def forward(self, r):
        # solver expects float64
        r = r.to(self.dtype)
        b = self.block_size
        n = r.numel()
        assert n % b == 0
        num_blocks = n // b
        r = r.view(num_blocks, b)

        # solve block systems
        z = torch.cholesky_solve(
            r.unsqueeze(-1),
            self.block_chol
        ).squeeze(-1)

        # learned damping (scalar per block)
        weights = torch.exp(self.w).view(-1, 1)
        z = z / weights

        return z.view(-1).double()