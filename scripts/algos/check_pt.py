# train_orientation.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from skrl.memories.torch import RandomMemory
from skrl.utils.spaces.torch import unflatten_tensorized_space
from networks.networks import GraphEncoder, OrientationModule
from train import env  # или укажи observation_space вручную

DEVICE = torch.device("cuda")
MEMORY_DIR = "/home/xiso/IsaacLab/logs/skrl/memory/memories/26-02-24_17-18-23-800589_memory_0x7418057cde70.pt"
PREPROCESSOR_PATH = "logs/skrl/aloha_sac/memory/preprocessor.pt"
CHECKPOINT_PATH = "/home/xiso/IsaacLab/logs/skrl/aloha_sac/26-02-22_18-54-44-348061_SAC/checkpoints/agent_5000.pt"
BATCH_SIZE = 512
EPOCHS = 50
LR_GRAPH = 3e-4
LR_ORIENT = 1e-3

# ── 1. Загружаем буфер ──────────────────────────────────────────────
memory = RandomMemory(memory_size=1500, num_envs=128, device=DEVICE)
memory.load(directory=MEMORY_DIR)

# ── 2. Загружаем препроцессор и вытаскиваем все состояния ───────────
from networks.networks import DictRunningStandardScaler
obs_space = env.observation_space  # или pickle-файл если env недоступен

preprocessor = DictRunningStandardScaler(
    size=obs_space, img_space=obs_space["img"], device=DEVICE
)
preprocessor.load_state_dict(torch.load(PREPROCESSOR_PATH))
preprocessor.eval()

# Берём все данные из буфера одним куском
all_states = memory.get_tensor_by_name("states")  # [N, flat_dim]

with torch.no_grad():
    processed = preprocessor(all_states, train=False)
    s = unflatten_tensorized_space(obs_space, processed)

img_data       = s["img"].cpu()
graph_data     = s["graph"].cpu()
orient_data    = s["orientation"].cpu()  # GT labels

dataset = TensorDataset(img_data, graph_data, orient_data)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ── 3. Инициализируем модули ─────────────────────────────────────────
graph_encoder = GraphEncoder(
    embeddings_path="source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt"
).to(DEVICE)

orient_module = OrientationModule(
    img_dim=obs_space["img"].shape[0]
).to(DEVICE)

# Опционально: загрузить веса из чекпоинта как стартовую точку
# ckpt = torch.load(CHECKPOINT_PATH)
# graph_encoder.load_state_dict(ckpt["graph_encoder"])
# orient_module.load_state_dict(ckpt["orient_module"])

graph_opt  = torch.optim.AdamW(graph_encoder.parameters(), lr=LR_GRAPH,  weight_decay=1e-4)
orient_opt = torch.optim.AdamW(orient_module.parameters(), lr=LR_ORIENT, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    orient_opt, T_max=EPOCHS * len(loader)
)

# ── 4. Цикл обучения ─────────────────────────────────────────────────
for epoch in range(EPOCHS):
    graph_encoder.train()
    orient_module.train()
    total_loss, total_acc = 0.0, 0.0

    for img, graph_flat, gt_yaw in loader:
        img, graph_flat, gt_yaw = img.to(DEVICE), graph_flat.to(DEVICE), gt_yaw.to(DEVICE)

        graph_emb = graph_encoder(graph_flat)
        _, probs, logits = orient_module(img, graph_emb)
        loss, metrics = orient_module.compute_loss(logits, probs, gt_yaw)

        graph_opt.zero_grad()
        orient_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graph_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(orient_module.parameters(), 1.0)
        graph_opt.step()
        orient_opt.step()
        scheduler.step()

        total_loss += metrics["orient/loss"]
        total_acc  += metrics["orient/acc_relaxed"]

    n = len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | loss: {total_loss/n:.4f} | acc_relaxed: {total_acc/n:.4f}")

# ── 5. Сохраняем ─────────────────────────────────────────────────────
torch.save(graph_encoder.state_dict(), "logs/skrl/aloha_sac/graph_encoder_pretrained.pt")
torch.save(orient_module.state_dict(), "logs/skrl/aloha_sac/orient_module_pretrained.pt")
print("Saved!")