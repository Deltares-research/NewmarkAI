import os
import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class FEMMultiSimDataset(Dataset):
    def __init__(self, root_dir, use_windows=True):
        self.files = []
        self.index = []  # (file_id, t0)

        for sim_dir in sorted(os.listdir(root_dir)):
            h5_path = os.path.join(root_dir, sim_dir, "data.h5")
            if not os.path.isfile(h5_path):
                continue

            self.files.append(h5_path)
            with h5py.File(h5_path, "r") as h5:
                T = h5["series/u"].shape[0]

                if use_windows and "windows" in h5:
                    starts = h5["windows/start"][:]
                else:
                    starts = range(T - 1)

                fid = len(self.files) - 1
                for t0 in starts:
                    self.index.append((fid, int(t0)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fid, t0 = self.index[idx]
        h5 = h5py.File(self.files[fid], "r")

        data = Data()
        data.edge_index = torch.tensor(h5["mesh/edge_index"][:], dtype=torch.long)

        # Node features
        coords = torch.tensor(h5["mesh/coords"][:], dtype=torch.float)
        Mii = torch.tensor(h5["static/Mii"][:], dtype=torch.float)
        Cii = torch.tensor(h5["static/Cii"][:], dtype=torch.float)
        Kii = torch.tensor(h5["static/Kii"][:], dtype=torch.float)
        bc  = torch.tensor(h5["mesh/bc_type"][:], dtype=torch.float)

        data.x_node = torch.cat([coords, Mii, Cii, Kii, bc], dim=1)

        # Edge features
        dist = torch.tensor(h5["static/edge_dist"][:], dtype=torch.float)
        dire = torch.tensor(h5["static/edge_dir"][:], dtype=torch.float)
        Kij  = torch.tensor(h5["static/Kij"][:], dtype=torch.float)
        Mij  = torch.tensor(h5["static/Mij"][:], dtype=torch.float)
        Cij  = torch.tensor(h5["static/Cij"][:], dtype=torch.float)
        data.x_edge = torch.cat([dist, dire, Kij, Mij, Cij], dim=1)

        # Time
        data.u_n   = torch.tensor(h5["series/u"][t0], dtype=torch.float)
        data.v_n   = torch.tensor(h5["series/v"][t0], dtype=torch.float)
        data.a_n   = torch.tensor(h5["series/a"][t0], dtype=torch.float)
        data.u_np1 = torch.tensor(h5["series/u"][t0+1], dtype=torch.float)
        data.v_np1 = torch.tensor(h5["series/v"][t0+1], dtype=torch.float)
        data.a_np1 = torch.tensor(h5["series/a"][t0+1], dtype=torch.float)
        data.f_np1 = torch.tensor(h5["series/force"][t0+1], dtype=torch.float)

        # Global
        meta = h5["meta"]
        data.x_global = torch.tensor([
            meta["dt"][()],
            meta["beta"][()],
            meta["gamma"][()],
        ], dtype=torch.float)

        h5.close()
        return data
