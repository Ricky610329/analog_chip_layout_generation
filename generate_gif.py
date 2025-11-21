# generate_gif.py (修正版)

# -*- coding: utf-8 -*-
"""
此腳本用於視覺化類比晶片佈局的完整生成過程，
並將每個主要步驟的快照匯出為一個 GIF 動畫檔案。

執行順序:
1. Level 0: 產生根元件
2. Level 1: 第一次階層式分割
3. Level 2: 第二次階層式分割 (智慧型)
4. GapFiller: (可選) 填補空白區域
5. Netlist: 產生最終的網路連接線
6. 將所有快照合成為一個 GIF。
"""

import os
import random
import yaml
import numpy as np
import glob
import imageio
import math
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

# --- 從專案中匯入必要的模組 ---
# 確保 aclg 套件在 Python 路徑中
from aclg.rules.split.split_ratio import split_by_ratio, SplitOrientation, split_by_ratio_grid
from aclg.rules.split.split_hold import split_hold
from aclg.rules.align import align_components, AlignmentMode
from aclg.dataclass.component import Component

# --- 從 production.ipynb 複製/改寫的核心類別 ---

def load_yaml_config(path='config.yaml') -> Dict[str, Any]:
    """從指定的路徑載入 YAML 設定檔。"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到設定檔 '{path}'。")
        return None
    except yaml.YAMLError as e:
        print(f"❌ 錯誤：解析 YAML 檔案 '{path}' 失敗: {e}")
        return None

class ComponentPlotter:
    """視覺化工具，可以繪製元件、邊(edges)，以及僅繪製被連接的引腳(pins)。"""
    def _draw_recursive(self, ax, component: Component):
        top_left_x, top_left_y = component.get_topleft()
        width, height, level = component.width, component.height, component.level
        LEVEL_COLORS = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#E0BBE4', '#FFD1DC', '#B2DFDB']
        color = LEVEL_COLORS[level % len(LEVEL_COLORS)]
        rect = plt.Rectangle((top_left_x, top_left_y), width, height,
                             linewidth=1.2, edgecolor='black', facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        label = f"L{level}\nID:{component.relation_id}"
        ax.text(component.x, component.y, label, ha='center', va='center', fontsize=8, color='black')
        if component.sub_components:
            for sub_comp in component.sub_components:
                self._draw_recursive(ax, sub_comp)

    def _draw_netlist(self, ax, edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        if not edges: return
        connected_pins = set()
        for p1, p2 in edges:
            connected_pins.add(p1)
            connected_pins.add(p2)
        
        for p1, p2 in edges:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#555555', linestyle='-', linewidth=0.7, alpha=0.6)
            
        for px, py in connected_pins:
            ax.plot(px, py, 'o', color='black', markersize=2.5, alpha=0.8)

    def plot(self, components_to_plot: List[Component], title: str = "Component Layout",
             edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
             output_filename: str = "component_visualization.png",
             canvas_dim: float = None):  # ✨ [新增] 接收畫布尺寸參數
        
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.set_facecolor('#f0f0f0')

        if not components_to_plot:
            ax.set_title("元件列表為空")
            plt.close(fig)
            return

        for comp in components_to_plot:
            self._draw_recursive(ax, comp)
        
        if edges:
            self._draw_netlist(ax, edges)
        
        # ✨ [修改] 設定視野範圍
        if canvas_dim:
            # 如果有傳入 canvas_dim (例如 1000)，就使用固定的畫布範圍 [-500, 500]
            half_dim = canvas_dim / 2
            ax.set_xlim(-half_dim, half_dim)
            ax.set_ylim(-half_dim, half_dim)
            # 注意：這裡故意不繪製紅色虛線框，只鎖定視野
        else:
            # 舊邏輯：自動貼合元件邊界
            root = components_to_plot[0]
            ax.set_xlim(root.x - root.width/2 - 20, root.x + root.width/2 + 20)
            ax.set_ylim(root.y - root.height/2 - 20, root.y + root.height/2 + 20)

        ax.set_aspect('equal', adjustable='box')
        plt.title(title, fontsize=18)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.savefig(output_filename, dpi=120)
        plt.close(fig)

# --- 產生器類別 (保持不變) ---
class Level_0:
    def __init__(self, w_range=(100, 120), h_range=(100, 120)):
        self.x = 0
        self.y = 0
        self.w_range = tuple(w_range)
        self.h_range = tuple(h_range)
        self.generate_rule = 'root'
        self.level = 0
        self.relation_id = 0
    
    def generate(self) -> List[Component]:
        return [Component(
            x=self.x,
            y=self.y,
            width=random.randint(*self.w_range),
            height=random.randint(*self.h_range),
            relation_id=self.relation_id,
            generate_rule=self.generate_rule,
            level=self.level
        )]

class Level_1:
    def __init__(self, **kwargs):
        self.w_h_ratio_bound = tuple(kwargs.get('w_h_ratio_bound', (1/6, 6/1)))
        self.max_tries_per_orientation = kwargs.get('max_tries_per_orientation', 50)
        self.num_splits_range = tuple(kwargs.get('num_splits_range', (2, 5)))
        self.ratio_range = tuple(kwargs.get('ratio_range', (0.3, 1.0)))
        self.split_only_probability = kwargs.get('split_only_probability', 0.5)
        self.align_scale_factor_range = tuple(kwargs.get('align_scale_factor_range', (0.2, 1.0)))
        self.force_align_threshold = kwargs.get('force_align_threshold', 3)
        self.level = 1

    def _find_valid_ratios(self, parent_component: Component, orientation: SplitOrientation, num_splits: int):
        parent_w_h_ratio = parent_component.w_h_ratio()
        min_ratio, max_ratio = self.w_h_ratio_bound
        for _ in range(self.max_tries_per_orientation):
            ratios = [random.uniform(*self.ratio_range) for _ in range(num_splits)]
            total_ratio = sum(ratios)
            all_valid = True
            for r in ratios:
                if orientation == SplitOrientation.HORIZONTAL:
                    sub_w_h_ratio = parent_w_h_ratio * (total_ratio / r)
                else:
                    sub_w_h_ratio = parent_w_h_ratio * (r / total_ratio)
                if not (min_ratio <= sub_w_h_ratio <= max_ratio):
                    all_valid = False
                    break
            if all_valid:
                return ratios
        return None

    def _apply_split(self, parent_component: Component, num_splits: int) -> List[Component]:
        orientations_to_try = [SplitOrientation.VERTICAL, SplitOrientation.HORIZONTAL] if parent_component.w_h_ratio() > 1 else [SplitOrientation.HORIZONTAL, SplitOrientation.VERTICAL]
        for orientation in orientations_to_try:
            valid_ratios = self._find_valid_ratios(parent_component, orientation, num_splits)
            if valid_ratios:
                return split_by_ratio(parent_component, valid_ratios, orientation)
        return split_hold(parent_component)

    def _apply_align(self, parent_component: Component, num_splits: int) -> List[Component]:
        align_mode = random.choice(list(AlignmentMode))
        required_orientation = SplitOrientation.VERTICAL if align_mode in [AlignmentMode.TOP, AlignmentMode.BOTTOM, AlignmentMode.CENTER_H] else SplitOrientation.HORIZONTAL
        
        valid_ratios = self._find_valid_ratios(parent_component, required_orientation, num_splits)
        if not valid_ratios:
            return split_hold(parent_component)
        sub_components = split_by_ratio(parent_component, valid_ratios, required_orientation)

        scale_factors = []
        min_ratio_bound, max_ratio_bound = self.w_h_ratio_bound
        min_scale_bound, max_scale_bound = self.align_scale_factor_range
        for comp in sub_components:
            original_ratio = comp.w_h_ratio()
            if align_mode in [AlignmentMode.TOP, AlignmentMode.BOTTOM, AlignmentMode.CENTER_H]:
                valid_min_s = original_ratio / max_ratio_bound
                valid_max_s = original_ratio / min_ratio_bound
            else:
                valid_min_s = min_ratio_bound / original_ratio
                valid_max_s = max_ratio_bound / original_ratio
            
            final_min_s = max(valid_min_s, min_scale_bound)
            final_max_s = min(valid_max_s, max_scale_bound)
            
            scale = 1.0 if final_min_s > final_max_s else random.uniform(final_min_s, final_max_s)
            scale_factors.append(scale)

        return align_components(sub_components, scale_factors, align_mode) 

    def _process_single_component(self, parent_component: Component) -> List[Component]:
        num_splits = random.randint(*self.num_splits_range) 
        if num_splits > self.force_align_threshold: 
            return self._apply_align(parent_component, num_splits) 
        else:
            if random.random() < self.split_only_probability: 
                return self._apply_split(parent_component, num_splits) 
            else:
                return self._apply_align(parent_component, num_splits) 

    def generate(self, components: List[Component]) -> List[Component]:
        all_results = []
        relation_id = 0
        for component in components:
            processed_sub_components = self._process_single_component(component)
            for sub_comp in processed_sub_components:
                sub_comp.level = self.level
                sub_comp.relation_id = relation_id
            all_results.extend(processed_sub_components)
            relation_id += 1
            component.sub_components = processed_sub_components 
        return all_results 

class Level_2(Level_1): 
    def __init__(self, **kwargs):
        self.large_component_align_probability = kwargs.get('large_component_align_probability', 1.0)
        self.wide_threshold = kwargs.get('wide_threshold', 2.0)
        self.tall_threshold = kwargs.get('tall_threshold', 0.5)
        self.size_thresholds = tuple(kwargs.get('size_thresholds', (0.1, 0.4)))
        self.small_component_hold_probability = kwargs.get('small_component_hold_probability', 0.8)
        self.policy_wide = kwargs.get('policy_wide', {"rows_range": (1, 2), "cols_range": (3, 5)})
        self.policy_tall = kwargs.get('policy_tall', {"rows_range": (3, 5), "cols_range": (1, 2)})
        self.policy_square = kwargs.get('policy_square', {"rows_range": (2, 4), "cols_range": (2, 4)})
        self.w_h_ratio_bound = tuple(kwargs.get('w_h_ratio_bound', (1/6, 6/1)))
        
        self.max_tries_per_orientation = kwargs.get('max_tries', 50)
        
        self.ratio_grid_probability = kwargs.get('ratio_grid_probability', 0.5)
        self.ratio_range = tuple(kwargs.get('ratio_range', (0.3, 0.6)))
        self.large_component_hold_probability = kwargs.get('large_component_hold_probability', 0.7)
        self.simple_split_probability = kwargs.get('simple_split_probability', 0.9)
        self.num_splits_range = tuple(kwargs.get('num_splits_range', (2, 4)))
        self.level = 2

    def _apply_simple_split(self, parent_component: Component) -> List[Component]:
        num_splits = random.randint(*self.num_splits_range)
        orientations_to_try = [SplitOrientation.VERTICAL, SplitOrientation.HORIZONTAL] if parent_component.w_h_ratio() > 1 else [SplitOrientation.HORIZONTAL, SplitOrientation.VERTICAL]
        for orientation in orientations_to_try:
            valid_ratios = self._find_valid_ratios(parent_component, orientation, num_splits)
            if valid_ratios:
                return split_by_ratio(parent_component, valid_ratios, orientation)
        return split_hold(parent_component)
    
    def generate(self, components: List[Component], root_component: Component) -> List[Component]:
        if not components: return []
        all_results = []
        relation_id = 0
        for comp in components:
            processed_sub_components = self._apply_simple_split(comp)
            
            for sub_comp in processed_sub_components: 
                sub_comp.level = self.level 
                sub_comp.relation_id = relation_id 
            all_results.extend(processed_sub_components) 
            relation_id += 1 
            comp.sub_components = processed_sub_components 
        return all_results

class GapFiller:
    def __init__(self, **kwargs):
        self.w_range = tuple(kwargs.get('small_comp_w_range', (6, 14)))
        self.h_range = tuple(kwargs.get('small_comp_h_range', (6, 14)))
        self.spacing = kwargs.get('spacing', 0.5)
        self.level = 4

    def _check_collision(self, new_comp: Component, all_components: List[Component], root_component: Component) -> bool:
        root_left, root_top = root_component.get_topleft()
        root_right, root_bottom = root_component.get_bottomright()
        new_left, new_top = new_comp.get_topleft()
        new_right, new_bottom = new_comp.get_bottomright()

        if not (new_left >= root_left and new_right <= root_right and new_top >= root_top and new_bottom <= root_bottom):
            return True
        for comp in all_components:
            comp_left, comp_top = comp.get_topleft()
            comp_right, comp_bottom = comp.get_bottomright()
            if (new_left < comp_right and new_right > comp_left and new_top < comp_bottom and new_bottom > comp_top):
                return True
        return False

    def fill(self, existing_leaf_components: List[Component], root_component: Component, num_to_place: int) -> List[Component]:
        if not existing_leaf_components or num_to_place == 0: return []
        
        best_host, max_edge_len, best_edge_type = None, -1.0, '' 
        for comp in existing_leaf_components: 
            if comp.width > max_edge_len: 
                max_edge_len, best_host, best_edge_type = comp.width, comp, random.choice(['top', 'bottom'])
            if comp.height > max_edge_len:
                max_edge_len, best_host, best_edge_type = comp.height, comp, random.choice(['left', 'right'])
        
        if not best_host: return []

        gap_components = []
        all_components = existing_leaf_components.copy()
        h_left, h_top = best_host.get_topleft()
        h_right, h_bottom = best_host.get_bottomright()
        cursor = h_left if best_edge_type in ['top', 'bottom'] else h_top

        for _ in range(num_to_place):
            new_w, new_h = random.uniform(*self.w_range), random.uniform(*self.h_range)
            new_comp = Component(x=0, y=0, width=new_w, height=new_h, level=self.level, relation_id=-1)

            if best_edge_type == 'bottom':
                if cursor + new_w > h_right: break
                new_comp.x, new_comp.y = cursor + new_w / 2, h_bottom + new_h / 2
            elif best_edge_type == 'top':
                if cursor + new_w > h_right: break
                new_comp.x, new_comp.y = cursor + new_w / 2, h_top - new_h / 2
            elif best_edge_type == 'right':
                if cursor + new_h > h_bottom: break
                new_comp.x, new_comp.y = h_right + new_w / 2, cursor + new_h / 2
            elif best_edge_type == 'left':
                if cursor + new_h > h_bottom: break
                new_comp.x, new_comp.y = h_left - new_w / 2, cursor + new_h / 2

            if not self._check_collision(new_comp, all_components, root_component):
                gap_components.append(new_comp)
                all_components.append(new_comp)
                cursor += (new_w + self.spacing) if best_edge_type in ['top', 'bottom'] else (new_h + self.spacing)
            else:
                break
        return gap_components

class NetlistGenerator:
    def __init__(self, **kwargs):
        self.pin_alpha = kwargs.get('pin_dist_alpha', 2.5)
        self.min_pins = kwargs.get('min_pins_per_comp', 2)
        self.max_pins = kwargs.get('max_pins_per_comp', 50)
        self.s = kwargs.get('edge_scale_param', 15.0)
        self.gamma = kwargs.get('edge_gamma_multiplier', 0.05)
        self.max_p = kwargs.get('max_edge_prob', 0.9)

    def _generate_pins_for_components(self, components: List[Component]) -> List[List[Tuple[float, float]]]:
        all_pins = []
        num_pins_array = np.random.zipf(self.pin_alpha, len(components)) + self.min_pins
        for i, comp in enumerate(components):
            comp_pins = []
            num_pins = int(np.clip(num_pins_array[i], self.min_pins, self.max_pins))
            left, top = comp.get_topleft()
            right, bottom = comp.get_bottomright()
            for _ in range(num_pins):
                pin_x, pin_y = random.uniform(left, right), random.uniform(top, bottom)
                comp_pins.append((pin_x, pin_y))
            all_pins.append(comp_pins)
        return all_pins

    def generate(self, components: List[Component]) -> Tuple[List[Any], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        if not components: return [], []
        all_pins = self._generate_pins_for_components(components)
        
        edges = []
        for i in range(len(components) - 1):
            if all_pins[i] and all_pins[i+1]:
                p1 = random.choice(all_pins[i])
                p2 = random.choice(all_pins[i+1])
                edges.append((p1, p2))
        
        print(f"[*] Netlist 產生完畢，最終總共有 {len(edges)} 條邊。")
        return all_pins, edges

def main():
    """主執行函式"""
    config = load_yaml_config('config.yaml')
    if config is None:
        return

    gif_cfg = config.get('gif_settings', {})
    main_cfg = config.get('main_execution', {})
    
    # ✨ [新增] 從 config 讀取畫布大小，預設 1000
    ml_cfg = config.get('ml_preparation', {})
    target_canvas_dim = ml_cfg.get('target_canvas_dim', 1000.0)
    
    output_dir = gif_cfg.get('output_directory', 'generation_visualizations')
    gif_filename = gif_cfg.get('gif_filename', 'generation_process.gif')
    duration = gif_cfg.get('frame_duration_seconds', 2.0)
    cleanup = gif_cfg.get('cleanup_frames', True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    seed = gif_cfg.get('seed', 'random')
    if seed == 'random':
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    print(f"使用種子: {seed} 進行生成...")

    level_0_gen = Level_0(**config.get('Level_0', {}))
    level_1_gen = Level_1(**config.get('Level_1', {}))
    level_2_gen = Level_2(**config.get('Level_2', {}))
    gap_filler = GapFiller(**config.get('GapFiller', {}))
    netlist_gen = NetlistGenerator(**config.get('NetlistGenerator', {}))
    plotter = ComponentPlotter()
    frame_files = []

    # --- Stage 1: Level 0 ---
    print("➡️ [Stage 1/5] Generating Level 0 (Root Component)...")
    root_components = level_0_gen.generate()
    root_component = root_components[0]
    frame_path = os.path.join(output_dir, "01_level_0.png")
    
    # ✨ [修改] 傳入 canvas_dim
    plotter.plot(root_components, title="Stage 1: Root Component (Level 0)", output_filename=frame_path, canvas_dim=target_canvas_dim)
    frame_files.append(frame_path)

    # --- Stage 2: Level 1 ---
    print("➡️ [Stage 2/5] Generating Level 1 (First Split)...")
    level_1_components = level_1_gen.generate(root_components)
    frame_path = os.path.join(output_dir, "02_level_1.png")
    plotter.plot(root_components, title="Stage 2: First Split (Level 1)", output_filename=frame_path, canvas_dim=target_canvas_dim)
    frame_files.append(frame_path)

    # --- Stage 3: Level 2 ---
    print("➡️ [Stage 3/5] Generating Level 2 (Second Split)...")
    level_2_components = level_2_gen.generate(level_1_components, root_component)
    frame_path = os.path.join(output_dir, "03_level_2.png")
    plotter.plot(root_components, title="Stage 3: Second Split (Level 2)", output_filename=frame_path, canvas_dim=target_canvas_dim)
    frame_files.append(frame_path)

    # --- Stage 4: GapFiller ---
    print("➡️ [Stage 4/5] Running GapFiller...")
    gap_components = []
    occupied_area = sum(c.width * c.height for c in level_2_components)
    total_area = root_component.width * root_component.height

    if total_area > 0 and (total_area - occupied_area) / total_area > main_cfg.get('gap_filler_activation_threshold', 0.2):
        num_gaps = main_cfg.get('num_gaps_to_fill', 0)
        gap_components = gap_filler.fill(level_2_components, root_component, num_gaps)
        print(f"   -> Filled {len(gap_components)} gap components.")
    else:
        print("   -> No gap filling required, skipped.")

    frame_path = os.path.join(output_dir, "04_gap_filler.png")
    plotter.plot(root_components + gap_components, title="Stage 4: Gap Filling (GapFiller)", output_filename=frame_path, canvas_dim=target_canvas_dim)
    frame_files.append(frame_path)

    # --- Stage 5: Netlist ---
    print("➡️ [Stage 5/5] Generating Netlist...")
    final_leaf_components = level_2_components + gap_components
    _, edges = netlist_gen.generate(final_leaf_components)
    frame_path = os.path.join(output_dir, "05_netlist.png")
    plotter.plot(
        root_components + gap_components,
        title="Stage 5: Final Layout and Netlist",
        edges=edges,
        output_filename=frame_path,
        canvas_dim=target_canvas_dim # ✨ 傳入畫布大小
    )
    frame_files.append(frame_path)

    
    # --- 3. 合成 GIF ---
    print("\n正在將所有快照合成為 GIF...")
    gif_path = os.path.join(output_dir, gif_filename)
    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"GIF 成功儲存至: {gif_path}")

    # --- 4. 清理 ---
    if cleanup:
        print("🗑️ 正在清理暫存圖片...")
        for filename in frame_files:
            os.remove(filename)
        print("   -> 清理完畢。")

    print("\n所有任務執行完畢！")

if __name__ == "__main__":
    main()