import json
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
import torch

# --- настройки путей ---
INPUT_PATH = Path("/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/scene_items.json")         # исходный JSON
OUTPUT_PATH = Path("/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/cdecode_dict.json")      # сюда запишется словарь кодов
CLIP_EMB_PATH = Path("/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt")   # сюда запишем эмбеддинги CLIP


def normalize_name(name: str) -> str:
    """
    Обрезаем по первому '_' и приводим к нижнему регистру:
    'table_2' -> 'table', 'Table' -> 'table'
    """
    return name.split("_", 1)[0].lower()


def build_codebooks(data: dict):
    """
    Строим словари:
      - names:  name -> index (int)
      - colors: color -> index (int)

    Порядок сортированный и ДОЛЖЕН совпадать с порядком,
    в котором мы считаем эмбеддинги CLIP.
    """
    objects = data.get("objects", [])

    # 1) Собираем уникальные нормализованные имена и цвета
    name_set = set()
    color_set = set()

    for obj in objects:
        raw_name = obj.get("name", "")
        norm_name = normalize_name(raw_name)
        if norm_name:
            name_set.add(norm_name)

        color = str(obj.get("info", {}).get("color", "")).lower()
        if color:
            color_set.add(color)

    # 2) Отсортированные списки (детерминированный порядок)
    sorted_names = sorted(name_set)
    sorted_colors = sorted(color_set)

    # 3) Индексы = позиция в этих списках
    # имя -> индекс в name_embs
    name_codes = {name: idx for idx, name in enumerate(sorted_names)}

    # цвет -> индекс в color_embs
    color_codes = {color: idx for idx, color in enumerate(sorted_colors)}

    # 4) Итоговый словарь (без строк "00" и без битовых троек)
    codebook = {
        "names": name_codes,
        "colors": color_codes,
    }

    return codebook, sorted_names, sorted_colors


def save_clip_embeddings(sorted_names, sorted_colors):
    """
    Считаем CLIP-эмбеддинги для всех имён и цветов и сохраняем в .pt.

    Гарантия:
      - name_embs[i] соответствует sorted_names[i] и codebook["names"][sorted_names[i]] == i
      - color_embs[j] соответствует sorted_colors[j] и codebook["colors"][sorted_colors[j]] == j
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    name_texts = sorted_names
    color_texts = sorted_colors
    all_texts = name_texts + color_texts

    if not all_texts:
        torch.save(
            {"names": sorted_names, "colors": sorted_colors,
            "name_embs": torch.empty(0, 512), "color_embs": torch.empty(0, 512)},
            CLIP_EMB_PATH
        )
        return

    # токенизация (это да, норм)
    text_inputs = processor(
        text=all_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        # ВАЖНО: не encode_text, а get_text_features
        embs = model.get_text_features(**text_inputs)          # [len(all_texts), 512]
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-9) # нормализация

    # обратно делим на имена и цвета
    n_names = len(name_texts)
    name_embs = embs[:n_names].cpu()
    color_embs = embs[n_names:].cpu()

    torch.save(
        {"names": sorted_names, "colors": sorted_colors,
        "name_embs": name_embs, "color_embs": color_embs},
        CLIP_EMB_PATH
    )

    print(f"[INFO] Saved CLIP embeddings to {CLIP_EMB_PATH}")
    print(f"       names:  {len(sorted_names)}  →  name_embs.shape = {tuple(name_embs.shape)}")
    print(f"       colors: {len(sorted_colors)}  →  color_embs.shape = {tuple(color_embs.shape)}")


def main():
    # читаем исходный файл
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    codebook, sorted_names, sorted_colors = build_codebooks(data)

    # записываем выходной JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved codebook JSON to {OUTPUT_PATH}")
    print(f"       names:  {len(sorted_names)}")
    print(f"       colors: {len(sorted_colors)}")

    # считаем и сохраняем CLIP-эмбеддинги
    save_clip_embeddings(sorted_names, sorted_colors)


if __name__ == "__main__":
    main()
