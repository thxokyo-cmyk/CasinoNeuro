"""
Roulette number/color mapping - STRICT VALIDATION.
American roulette: 0, 00, 1-36
"""

# === СТРОГАЯ ТАБЛИЦА ЦВЕТОВ ===
# Красные числа
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}

# Чёрные числа
BLACK_NUMBERS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}

# Зелёные числа
GREEN_NUMBERS = {0}  # 00 тоже зелёное, но хранится как строка

# Полная таблица: число -> цвет
NUMBER_TO_COLOR = {
    "0": "green",
    "00": "green",
    "1": "red",
    "2": "black",
    "3": "red",
    "4": "black",
    "5": "red",
    "6": "black",
    "7": "red",
    "8": "black",
    "9": "red",
    "10": "black",
    "11": "black",
    "12": "red",
    "13": "black",
    "14": "red",
    "15": "black",
    "16": "red",
    "17": "black",
    "18": "red",
    "19": "red",
    "20": "black",
    "21": "red",
    "22": "black",
    "23": "red",
    "24": "black",
    "25": "red",
    "26": "black",
    "27": "red",
    "28": "black",
    "29": "black",
    "30": "red",
    "31": "black",
    "32": "red",
    "33": "black",
    "34": "red",
    "35": "black",
    "36": "red",
}

# Обратная таблица: цвет -> список чисел
COLOR_TO_NUMBERS = {
    "green": ["0", "00"],
    "red": ["1", "3", "5", "7", "9", "12", "14", "16", "18", "19", "21", "23", "25", "27", "30", "32", "34", "36"],
    "black": ["2", "4", "6", "8", "10", "11", "13", "15", "17", "20", "22", "24", "26", "28", "29", "31", "33", "35"],
}


def get_color(number) -> str:
    """Get color for a roulette number"""
    if isinstance(number, str):
        return NUMBER_TO_COLOR.get(number, "unknown")
    else:
        return NUMBER_TO_COLOR.get(str(number), "unknown")


def is_valid_combination(number: str, color: str) -> bool:
    """Check if number-color combination is valid"""
    expected = NUMBER_TO_COLOR.get(number)
    return expected == color


def get_numbers_by_color(color: str) -> list:
    """Get all numbers of a specific color"""
    return COLOR_TO_NUMBERS.get(color, [])


def find_similar_numbers(ocr_text: str, detected_color: str) -> list:
    """
    Find valid numbers that could match OCR result + color.
    Returns list of possible numbers sorted by likelihood.
    
    IMPORTANT: Only returns numbers that match the detected_color!
    """
    valid_numbers = COLOR_TO_NUMBERS.get(detected_color, [])
    
    if not ocr_text:
        return valid_numbers
    
    candidates = []
    
    # Exact match
    if ocr_text in valid_numbers:
        candidates.append((ocr_text, 100))
    
    # Similar looking digits - MUST FILTER BY COLOR
    similar_map = {
        "0": ["0", "6", "8", "9"],
        "1": ["1", "7"],  # Removed "4" - it's black, not red!
        "2": ["2", "7"],
        "3": ["3", "8"],
        "4": ["4", "1", "9"],  # Will be filtered by color below
        "5": ["5", "6"],
        "6": ["6", "0", "8", "5"],
        "7": ["7", "1", "2"],
        "8": ["8", "0", "3", "6"],
        "9": ["9", "0", "4"],  # Will be filtered by color below
    }
    
    # For single digit OCR
    if len(ocr_text) == 1:
        for similar in similar_map.get(ocr_text, []):
            # CRITICAL: Only accept if it matches the detected color
            if similar in valid_numbers and similar != ocr_text:
                score = 80
                # Boost score if visually very similar
                if ocr_text == "1" and similar == "7":
                    score = 90
                elif ocr_text == "7" and similar == "1":
                    score = 90
                candidates.append((similar, score))
    
    # For double digit OCR - try variations
    if len(ocr_text) == 2:
        d1, d2 = ocr_text[0], ocr_text[1]
        
        # Original
        if ocr_text in valid_numbers:
            candidates.append((ocr_text, 100))
        
        # Swap digits
        swapped = d2 + d1
        if swapped in valid_numbers:
            candidates.append((swapped, 70))
        
        # First digit only
        if d1 in valid_numbers:
            candidates.append((d1, 60))
        
        # Second digit only
        if d2 in valid_numbers:
            candidates.append((d2, 60))
        
        # Similar first digit - MUST MATCH COLOR
        for sim1 in similar_map.get(d1, []):
            num = sim1 + d2
            if num in valid_numbers and num != ocr_text:
                candidates.append((num, 50))
        
        # Similar second digit - MUST MATCH COLOR
        for sim2 in similar_map.get(d2, []):
            num = d1 + sim2
            if num in valid_numbers and num != ocr_text:
                candidates.append((num, 50))
    
    # Remove duplicates, sort by score
    seen = set()
    result = []
    for num, score in sorted(candidates, key=lambda x: -x[1]):
        if num not in seen:
            seen.add(num)
            result.append(num)
    
    return result


# === Визуальное сходство цифр ===
# Какие цифры OCR часто путает
OCR_CONFUSION_MAP = {
    # OCR читает -> может быть на самом деле
    "0": ["0", "8", "6", "9"],
    "1": ["1", "7", "4"],
    "2": ["2", "7"],
    "3": ["3", "8", "9"],
    "4": ["4", "1", "9"],
    "5": ["5", "6", "8"],
    "6": ["6", "0", "8", "5"],
    "7": ["7", "1", "2"],
    "8": ["8", "0", "6", "3"],
    "9": ["9", "0", "4", "8"],
    
    # Двузначные ошибки
    "00": ["00", "0"],  # 00 может читаться как 0
    "11": ["11", "1", "17", "71"],
    "17": ["17", "7", "11", "71"],
    "10": ["10", "0", "18"],
    "12": ["12", "2", "17"],
}
# ==============================================================
# 🔄 Обратное соответствие «номер ↔ индекс» для ML-модуля
# ==============================================================

# Порядок, который использовался в старой модели:
# 0, 00, 1-36   → итого 38 позиций
NUMBER_ORDER = ["0", "00"] + [str(i) for i in range(1, 37)]

INDEX_TO_NUMBER = {idx: num for idx, num in enumerate(NUMBER_ORDER)}
NUMBER_TO_INDEX = {num: idx for idx, num in enumerate(NUMBER_ORDER)}


def number_to_index(number: str | int) -> int:
    """
    Преобразовать число рулетки в индекс (0-37).
    Вернёт -1, если число не из набора.
    """
    key = str(number)
    return NUMBER_TO_INDEX.get(key, -1)


def index_to_number(index: int) -> str:
    """
    Обратное преобразование индекса в строку-число рулетки.
    """
    return INDEX_TO_NUMBER.get(index, "UNKNOWN")