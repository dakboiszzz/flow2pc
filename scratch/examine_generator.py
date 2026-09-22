import torch
import matplotlib.pyplot as plt
import numpy as np
from repc.flow import ALICFMGeneratorWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load generator
generator = ALICFMGeneratorWrapper("checkpoints/generator.pt").to(device)
generator.eval()

# Load data
data = torch.load("data/knot.pt", map_location=device, weights_only=False)
x0_pool = data[0]
x1_pool = data[-1]

# Sample 5000 random pairs to form the "distribution"
N = 5000
idx0 = torch.randint(0, len(x0_pool), (N,), device=device)
idx1 = torch.randint(0, len(x1_pool), (N,), device=device)
x0 = x0_pool[idx0]
x1 = x1_pool[idx1]

# Evaluate paths at multiple timesteps
steps = 20
times = torch.linspace(0, 1, steps, device=device)
paths = []

with torch.no_grad():
    for t in times:
        t_batch = t.expand(N)
        xt = generator(x0, x1, t_batch)
        paths.append(xt.cpu())

paths = torch.stack(paths, dim=0) # [steps, N, 2]

# Plot 1: The bundle of threads (Alpha=0.05 to show density)
plt.figure(figsize=(10, 8))
for i in range(min(500, N)): # plot first 500 lines
    plt.plot(paths[:, i, 0], paths[:, i, 1], color='blue', alpha=0.03, linewidth=1)
plt.scatter(x0_pool[:, 0].cpu(), x0_pool[:, 1].cpu(), color='green', label='x0 (Start)', s=10)
plt.scatter(x1_pool[:, 0].cpu(), x1_pool[:, 1].cpu(), color='red', label='x1 (End)', s=10)
plt.title("Generator Paths: A Bundle of Deterministic Threads")
plt.legend()
plt.savefig("outputs/generator_threads.png", dpi=150)
plt.close()

# Plot 2: Spatial Density (2D Histogram) at t=0.5
plt.figure(figsize=(10, 8))
xt_mid = paths[steps//2].numpy()
plt.hist2d(xt_mid[:, 0], xt_mid[:, 1], bins=100, cmap='hot')
plt.colorbar(label='Density')
plt.title("Spatial Density of Generator at t=0.5 (Notice the empty space)")
plt.savefig("outputs/generator_density.png", dpi=150)
plt.close()

# Print statistics
print("Script finished generating plots.")
