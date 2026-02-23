import json
from pathlib import Path

# --- настройки путей ---
INPUT_PATH = Path("/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/scene_items.json")         # сюда положи исходный JSON
OUTPUT_PATH = Path("/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/cdecode_dict.json")    # сюда запишется словарь кодов


def normalize_name(name: str) -> str:
    """
    Обрезаем по первому '_' и приводим к нижнему регистру:
    'table_2' -> 'table', 'Table' -> 'table'
    """
    return name.split("_", 1)[0].lower()


def build_codebooks(data: dict) -> dict:
    objects = data.get("objects", [])

    # 1) Собираем уникальные нормализованные имена и цвета
    name_set = set()
    color_set = set()

    for obj in objects:
        raw_name = obj.get("name", "")
        norm_name = normalize_name(raw_name)
        name_set.add(norm_name)

        color = str(obj.get("info", {}).get("color", "")).lower()
        if color:
            color_set.add(color)

    # 2) Присваиваем кодам именам: "00".."99"
    # сортируем для детерминированности
    sorted_names = sorted(name_set)
    name_codes = {
        name: f"{idx:02d}"
        for idx, name in enumerate(sorted_names)
    }

    # 3) Присваиваем цветам тройки битов по «популярной системе»
    #    используем комбинации от 001 до 111 (7 уникальных кодов)
    #    111 присутствует, как ты просил
    bit_triples = [
        (i >> 2 & 1, i >> 1 & 1, i & 1)
        for i in range(1, 8)  # 001 .. 111
    ]

    sorted_colors = sorted(color_set)
    if len(sorted_colors) > len(bit_triples):
        raise ValueError(
            f"Слишком много уникальных цветов ({len(sorted_colors)}), "
            f"доступно только {len(bit_triples)} тройки."
        )

    color_codes = {
        color: list(bit_triples[idx])
        for idx, color in enumerate(sorted_colors)
    }

    # 4) Формируем итоговый словарь
    codebook = {
        "names": name_codes,
        "colors": color_codes,
    }

    return codebook


def main():
    # читаем исходный файл
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    codebook = build_codebooks(data)

    # записываем выходной JSON
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()