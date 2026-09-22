import torch
from repc.flow import build_teacher, ALICFMGeneratorWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_config = {
    "name": "alicfm",
    "checkpoint_path": "checkpoints/teacher.pt",
    "num_marginals": 60,
    "map_to_interval": True
}
teacher = build_teacher(teacher_config).to(device)
teacher.eval()

generator = ALICFMGeneratorWrapper("checkpoints/generator.pt").to(device)
generator.eval()

data = torch.load("data/knot.pt", map_location=device, weights_only=False)
x0_pool = data[0]
x1_pool = data[-1]

N = min(500, len(x0_pool))
idx0 = torch.arange(N, device=device)

# Random coupling
torch.manual_seed(42)
idx1_random = torch.randperm(N, device=device)

# "True" coupling (assuming dataset ordered sequentially, x0[i] corresponds to x1[i])
idx1_true = torch.arange(N, device=device)

t = torch.tensor(0.5, device=device).expand(N)

with torch.no_grad():
    path_random = generator(x0_pool[idx0], x1_pool[idx1_random], t)
    vel_random = teacher(t, path_random)
    
    path_true = generator(x0_pool[idx0], x1_pool[idx1_true], t)
    vel_true = teacher(t, path_true)

print(f"Random coupling - Mean path dist from true: {torch.norm(path_random - path_true, dim=-1).mean().item():.2f}")
print(f"Random coupling - Mean Teacher Vel Mag: {torch.norm(vel_random, dim=-1).mean().item():.2f}")
print(f"True coupling   - Mean Teacher Vel Mag: {torch.norm(vel_true, dim=-1).mean().item():.2f}")
