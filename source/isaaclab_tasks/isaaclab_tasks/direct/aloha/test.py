import torch
p = torch.load("/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt", map_location="cpu")
print(p.keys())
print("name_embs:", type(p.get("name_embs")), getattr(p.get("name_embs"), "shape", None))
print("color_embs:", type(p.get("color_embs")), getattr(p.get("color_embs"), "shape", None))
