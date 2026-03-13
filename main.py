from torch_geometric.loader import DataLoader
from data_loader import FEMMultiSimDataset

dataset = FEMMultiSimDataset("/home/bruno/software_dev/scatter/results_gnn")
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))

print(batch.x_node.shape)
print(batch.batch.shape)  # node → graph mapping