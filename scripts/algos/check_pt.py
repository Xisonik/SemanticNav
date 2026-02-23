import torch

ckpt = torch.load("/home/xiso/IsaacLab/logs/skrl/aloha_sac/26-02-23_11-33-07-648083_SAC/checkpoints/agent_10000.pt", map_location="cpu")

# Смотрим верхний уровень
print(type(ckpt))
print(ckpt.keys())  # если dict

# Все ключи включая вложенные
for k in ckpt.keys():
    print(k)