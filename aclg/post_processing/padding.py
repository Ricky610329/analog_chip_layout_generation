# aclg/post_processing/padding.py

from aclg.dataclass.component import Component
from typing import List, Tuple, Union
import random

def add_padding(
        components: List[Component], 
        padding: float
) -> List[Component]:
    """
    對一個或多個元件應用邊距（Padding），使其尺寸縮小。
    這個函式會直接修改傳入的元件物件。

    Args:
        components: 一個包含 Component 物件的列表。
        padding: 要在元件四周留下的邊界寬度。

    Returns:
        回傳修改後的元件列表。
    """
    if padding < 0:
        raise ValueError("Padding 值不能為負數。")

    for comp in components:
        # 檢查 padding 是否會導致尺寸變為負數
        if (comp.width - 2 * padding) < 0 or (comp.height - 2 * padding) < 0:
            raise ValueError(f"Padding ({padding}) 對於元件尺寸 ({comp.width}x{comp.height}) 而言太大了。")

        # 中心點 (x, y) 保持不變
        # generate_rule 保持不變

        # 寬度和高度各自減去兩倍的 padding
        comp.width -= 2 * padding
        comp.height -= 2 * padding

    return components

# --- NEW: 新增的進階 Padding 函式 ---
def add_padding_advanced(
        components: List[Component], 
        top: float = 0,
        bottom: float = 0,
        left: float = 0,
        right: float = 0
) -> List[Component]:
    """
    對一個或多個元件應用【非對稱】邊距，可分別指定四個方向的 padding。
    這會同時修改元件的尺寸和中心點位置。
    """
    for comp in components:
        padding_h = left + right
        padding_v = top + bottom

        if (comp.width - padding_h) < 0 or (comp.height - padding_v) < 0:
            raise ValueError(f"Padding (H:{padding_h}, V:{padding_v}) 對於元件尺寸 ({comp.width}x{comp.height}) 而言太大了。")

        # 1. 更新尺寸
        original_width = comp.width
        original_height = comp.height
        comp.width -= padding_h
        comp.height -= padding_v

        # 2. 根據邊距的不平衡來移動中心點
        # 如果 left padding 較大，元件需向右移動；如果 right padding 較大，則向左移動。
        x_shift = (left - right) / 2
        # 如果 top padding 較大，元件需向下移動；如果 bottom padding 較大，則向上移動。
        y_shift = (top - bottom) / 2
        
        comp.x += x_shift
        comp.y += y_shift
        
    return components

# --- MODIFIED: 函式現在可以接受一個範圍來產生隨機大小的 Padding ---
def add_padding_random_oneside(
        components: List[Component],
        padding_value: Union[float, Tuple[float, float]]
) -> List[Component]:
    """
    為列表中的每一個元件，隨機選擇一個邊並應用 Padding。
    Padding 的大小可以是固定的，也可以是一個範圍內的隨機值。

    Args:
        components: 要處理的元件列表。
        padding_value: 
            - 如果是 float: 使用固定的 padding 值。
            - 如果是 tuple (min, max): 為每個元件隨機選擇一個介於 min 和 max 之間的值。
    """
    sides = ['top', 'bottom', 'left', 'right']
    
    for comp in components:
        # 根據傳入的 padding_value 決定實際的 padding 大小
        if isinstance(padding_value, tuple):
            # 如果是元組，從範圍內隨機取一個浮點數
            min_pad, max_pad = padding_value
            if min_pad < 0 or max_pad < 0 or max_pad < min_pad:
                raise ValueError("無效的 padding 範圍。")
            current_padding = random.uniform(min_pad, max_pad)
        else:
            # 否則，使用固定的浮點數值
            if padding_value < 0:
                raise ValueError("Padding 值不能為負數。")
            current_padding = padding_value

        # 隨機選擇一個邊並應用 padding
        chosen_side = random.choice(sides)
        
        # 使用 try-except 來優雅地處理 padding 過大的情況
        try:
            if chosen_side == 'top':
                add_padding_advanced([comp], top=current_padding)
            elif chosen_side == 'bottom':
                add_padding_advanced([comp], bottom=current_padding)
            elif chosen_side == 'left':
                add_padding_advanced([comp], left=current_padding)
            elif chosen_side == 'right':
                add_padding_advanced([comp], right=current_padding)
        except ValueError as e:
            # 如果 padding 對於某個小元件來說太大了，就印出警告並跳過它
            # print(f"Warning: Skipping padding for a component due to error: {e}")
            pass
            
    return components

# --- NEW: 基於對齊方向的 Padding 函式 ---
def add_padding_based_on_alignment(
    components: List[Component],
    padding_range: Tuple[float, float],
    vertical_align_side: str = 'both',
    horizontal_align_side: str = 'both'
) -> List[Component]:
    """
    根據元件的對齊規則 (儲存在 generate_rule)，在其排列軸的兩側施加 Padding。

    Args:
        components: 要處理的元件列表。
        padding_range: 一個 (min, max) 的元組，用於隨機選擇 padding 大小。
        vertical_align_side: 對於垂直排列的元件 (align_top, etc.)，
                             可選 'left', 'right', 'both', 'none'。
        horizontal_align_side: 對於水平排列的元件 (align_left, etc.)，
                               可選 'top', 'bottom', 'both', 'none'。
    """
    # 定義哪些規則屬於垂直或水平排列
    HORIZONTAL_ALIGN_RULES = {'align_top', 'align_bottom', 'align_center_horizontal'}
    VERTICAL_ALIGN_RULES = {'align_left', 'align_right', 'align_center_vertical'}

    for comp in components:
        rule = comp.generate_rule
        current_padding = random.uniform(*padding_range)

        try:
            if any(align_rule in rule for align_rule in VERTICAL_ALIGN_RULES):
                # 這是一個垂直排列的元件，在其左右施加 Padding
                if vertical_align_side == 'left':
                    add_padding_advanced([comp], left=current_padding)
                elif vertical_align_side == 'right':
                    add_padding_advanced([comp], right=current_padding)
                elif vertical_align_side == 'both':
                    add_padding_advanced([comp], left=current_padding, right=current_padding)

            elif any(align_rule in rule for align_rule in HORIZONTAL_ALIGN_RULES):
                # 這是一個水平排列的元件，在其上下施加 Padding
                if horizontal_align_side == 'top':
                    add_padding_advanced([comp], top=current_padding)
                elif horizontal_align_side == 'bottom':
                    add_padding_advanced([comp], bottom=current_padding)
                elif horizontal_align_side == 'both':
                    add_padding_advanced([comp], top=current_padding, bottom=current_padding)
        
        except ValueError:
            # Padding 過大，優雅地跳過
            pass

    return components