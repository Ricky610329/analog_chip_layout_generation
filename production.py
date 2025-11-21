# -*- coding: utf-8 -*-

# ==============================================================================
# Cell 1: 匯入基礎模組
# ==============================================================================
import os
import math
import random
import json
import yaml
from datetime import datetime
from collections import defaultdict
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

# --- 從專案中匯入必要的模組 ---
# 確保 aclg 套件在 Python 路徑中
from aclg.rules.split.split_ratio import split_by_ratio, SplitOrientation, split_by_ratio_grid
from aclg.rules.split.split_basic import split_horizontal, split_vertical
from aclg.rules.split.split_hold import split_hold
from aclg.rules.spacing import spacing_grid, spacing_vertical, spacing_horizontal
from aclg.post_processing.padding import add_padding, add_padding_advanced, add_padding_random_oneside, add_padding_based_on_alignment
from aclg.rules.symetric.symmetric_1 import split_symmetric_1_horizontal, split_symmetric_1_vertical
from aclg.rules.align import align_components, AlignmentMode
from aclg.dataclass.component import Component

# ==============================================================================
# Cell 2: YAML 設定檔載入函式
# ==============================================================================
def load_yaml_config(path='config.yaml'):
    """
    從指定的路徑載入 YAML 設定檔。

    Args:
        path (str): YAML 檔案的路徑。

    Returns:
        dict: 包含設定參數的字典；如果檔案不存在或解析失敗則回傳 None。
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # 使用 safe_load 更安全，避免執行任意程式碼
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到設定檔 '{path}'。")
        return None
    except yaml.YAMLError as e:
        print(f"錯誤：解析 YAML 檔案 '{path}' 失敗: {e}")
        return None

# ==============================================================================
# Cell 3: NetlistGenerator 類別
# ==============================================================================
class NetlistGenerator:
    """
    (最終版) 產生 Netlist，採用三階段策略確保：
    1. 連接優先考慮距離近的 Pin (使用 K-近鄰權重隨機選擇增加多樣性)。
    2. 所有 Pin 都有連接。
    3. 所有元件最終形成一個單一連通圖。
    4. [新增] 大元件的 Pin 數量不超過總元件數的 1.5 倍。
    """
    def __init__(self,
                 pin_distribution_rules: Dict[str, Any] = None,
                 pin_dist_alpha: float = 2.5,
                 min_pins_per_comp: int = 2,
                 max_pins_per_comp: int = 50,
                 edge_scale_param: float = 15.0,
                 edge_gamma_multiplier: float = 0.05,
                 max_edge_prob: float = 0.9,
                 k_nearest_neighbors: int = 5):
        
        if pin_distribution_rules:
            rules = pin_distribution_rules
            base_probs = rules.get('base_probabilities', {})
            self.prob_2_pin = base_probs.get('2_pin', 0.55)
            self.prob_3_pin = base_probs.get('3_pin', 0.10)
            self.prob_4_pin = base_probs.get('4_pin', 0.30)
            self.large_comp_area_threshold = rules.get('large_comp_area_threshold', 1000.0)
            self.large_comp_high_pin_prob = rules.get('large_comp_high_pin_prob', 0.80)
            self.large_pin_count_range = tuple(rules.get('large_pin_count_range', [5, 10]))
            self.prob_large_pin = 1.0 - (self.prob_2_pin + self.prob_3_pin + self.prob_4_pin)

        self.s = edge_scale_param
        self.gamma = edge_gamma_multiplier
        self.max_p = max_edge_prob
        self.k_nearest = k_nearest_neighbors
        
        self.pin_alpha = pin_dist_alpha
        self.min_pins = min_pins_per_comp
        self.max_pins = max_pins_per_comp
    
    def _get_pin_count_for_component(self, component: Component, total_num_components: int) -> int:
        """根據元件面積和規則，決定單一元件的 Pin 腳數量。"""
        area = component.width * component.height
        is_large = area > self.large_comp_area_threshold

        # 計算 Pin 數量的動態上限
        max_allowed_pins = math.floor(total_num_components * 1.5)
        max_allowed_pins = max(2, max_allowed_pins)

        def get_large_pin_count():
            min_pins_cfg, max_pins_cfg = self.large_pin_count_range
            effective_max = min(max_pins_cfg, max_allowed_pins)
            if effective_max < min_pins_cfg:
                return effective_max
            return random.randint(min_pins_cfg, effective_max)

        if is_large:
            if random.random() < self.large_comp_high_pin_prob:
                return get_large_pin_count()
            else:
                pin_choices, base_total = [2, 3, 4], self.prob_2_pin + self.prob_3_pin + self.prob_4_pin
                probabilities = [self.prob_2_pin / base_total, self.prob_3_pin / base_total, self.prob_4_pin / base_total]
                return random.choices(pin_choices, weights=probabilities, k=1)[0]
        else:
            pin_choices = [2, 3, 4, 'large']
            probabilities = [self.prob_2_pin, self.prob_3_pin, self.prob_4_pin, self.prob_large_pin]
            choice = random.choices(pin_choices, weights=probabilities, k=1)[0]
            
            if choice == 'large':
                return get_large_pin_count()
            else:
                return choice

    def _generate_pins_for_components(self, components: List[Component]) -> List[List[Tuple[float, float]]]:
        """[最終完整版] 為所有元件產生引腳座標。"""
        total_num_components = len(components)
        symmetric_groups = defaultdict(list)
        comp_to_idx_map = {id(comp): i for i, comp in enumerate(components)}
        for comp in components:
            if comp.symmetric_group_id != -1:
                symmetric_groups[comp.symmetric_group_id].append(comp)
        all_pins = [[] for _ in components]
        processed_indices = set()
        for group_id, group_members in symmetric_groups.items():
            if len(group_members) != 2: continue
            master_comp, slave_comp = group_members[0], group_members[1]
            master_idx, slave_idx = comp_to_idx_map[id(master_comp)], comp_to_idx_map[id(slave_comp)]
            processed_indices.add(master_idx); processed_indices.add(slave_idx)
            
            num_pins = self._get_pin_count_for_component(master_comp, total_num_components)
            
            master_pins = []
            m_left, m_top = master_comp.get_topleft()
            m_right, m_bottom = master_comp.get_bottomright()
            for _ in range(num_pins):
                master_pins.append((random.uniform(m_left, m_right), random.uniform(m_top, m_bottom)))
            all_pins[master_idx] = master_pins
            slave_pins, delta_x, delta_y = [], abs(master_comp.x - slave_comp.x), abs(master_comp.y - slave_comp.y)
            if delta_y < delta_x:
                for m_pin_x, m_pin_y in master_pins: slave_pins.append((slave_comp.x - (m_pin_x - master_comp.x), m_pin_y + (slave_comp.y - master_comp.y)))
            elif delta_x < delta_y:
                for m_pin_x, m_pin_y in master_pins: slave_pins.append((m_pin_x + (slave_comp.x - master_comp.x), slave_comp.y - (m_pin_y - master_comp.y)))
            else:
                slave_pins = list(master_pins)
            all_pins[slave_idx] = slave_pins
            
        for i, comp in enumerate(components):
            if i in processed_indices: continue
            num_pins = self._get_pin_count_for_component(comp, total_num_components)
            comp_pins = []
            left, top = comp.get_topleft()
            right, bottom = comp.get_bottomright()
            for _ in range(num_pins):
                comp_pins.append((random.uniform(left, right), random.uniform(top, bottom)))
            all_pins[i] = comp_pins
        return all_pins

    def _generate_probabilistic_edges(self, all_pins: List[List[Tuple[float, float]]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        edges, pin_to_comp_map = [], {pin: i for i, comp_pins in enumerate(all_pins) for pin in comp_pins}
        all_pin_coords = list(pin_to_comp_map.keys())
        for i in range(len(all_pin_coords)):
            for j in range(i + 1, len(all_pin_coords)):
                p1, p2 = all_pin_coords[i], all_pin_coords[j]
                if pin_to_comp_map[p1] == pin_to_comp_map[p2]: continue
                l1_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                prob = min(self.gamma * math.exp(-l1_distance / self.s), self.max_p)
                if random.random() < prob: edges.append((p1, p2))
        return edges

    def _ensure_all_pins_connected(self, all_pins: List[List[Tuple[float, float]]], edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        pin_to_comp_map = {pin: i for i, comp_pins in enumerate(all_pins) for pin in comp_pins}
        all_pin_coords = list(pin_to_comp_map.keys())
        if not all_pin_coords: return
        connected_pins = {p for edge in edges for p in edge}
        unconnected_pins = [p for p in all_pin_coords if p not in connected_pins]
        if not unconnected_pins: return
        print(f"[*] 發現 {len(unconnected_pins)} 個未連接的 Pin，進行多樣化局部連接...")
        for p1 in unconnected_pins:
            if p1 in connected_pins: continue
            p1_comp_idx = pin_to_comp_map[p1]
            candidates = []
            for p2 in all_pin_coords:
                if p1 is p2 or pin_to_comp_map.get(p2) == p1_comp_idx: continue
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                candidates.append({'pin': p2, 'dist': dist})
            if not candidates: continue
            candidates.sort(key=lambda x: x['dist'])
            top_k_candidates = candidates[:self.k_nearest]
            candidate_pins = [c['pin'] for c in top_k_candidates]
            weights = [1.0 / (c['dist'] + 1e-9) for c in top_k_candidates]
            if candidate_pins:
                chosen_p2 = random.choices(candidate_pins, weights=weights, k=1)[0]
                edges.append((p1, chosen_p2))
                connected_pins.add(p1); connected_pins.add(chosen_p2)

    def _ensure_single_connected_component(self, components: List[Component], all_pins: List[List[Tuple[float, float]]], edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        num_components = len(components)
        if num_components < 2: return
        adj, pin_to_comp_map = {i: set() for i in range(num_components)}, {pin: i for i, comp_pins in enumerate(all_pins) for pin in comp_pins}
        for p1, p2 in edges:
            comp_idx1, comp_idx2 = pin_to_comp_map.get(p1), pin_to_comp_map.get(p2)
            if comp_idx1 is not None and comp_idx2 is not None:
                adj[comp_idx1].add(comp_idx2); adj[comp_idx2].add(comp_idx1)
        visited, components_groups = set(), []
        for i in range(num_components):
            if i not in visited:
                group, q = [], [i]
                visited.add(i)
                while q:
                    u = q.pop(0)
                    group.append(u)
                    for v in adj[u]:
                        if v not in visited: visited.add(v); q.append(v)
                components_groups.append(group)
        if len(components_groups) <= 1:
            print("[*] 所有元件已連通，無需橋接。")
            return
        print(f"[*] 發現 {len(components_groups)} 個獨立的元件群，開始最終橋接...")
        main_group = components_groups[0]
        for i in range(1, len(components_groups)):
            group_to_bridge = components_groups[i]
            min_dist, best_bridge_edge = float('inf'), None
            for u_idx in main_group:
                for v_idx in group_to_bridge:
                    for p1 in all_pins[u_idx]:
                        for p2 in all_pins[v_idx]:
                            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                            if dist < min_dist: min_dist, best_bridge_edge = dist, (p1, p2)
            if best_bridge_edge:
                edges.append(best_bridge_edge)
                main_group.extend(group_to_bridge)

    def generate(self, components: List[Component]) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        if not components: return [], []
        print(f"[*] 開始為 {len(components)} 個元件產生 Netlist...")
        all_pins = self._generate_pins_for_components(components)
        edges = self._generate_probabilistic_edges(all_pins)
        print(f"[*] 初始機率性產生了 {len(edges)} 條邊。")
        self._ensure_all_pins_connected(all_pins, edges)
        self._ensure_single_connected_component(components, all_pins, edges)
        print(f"[*] Netlist 產生完畢，最終總共有 {len(edges)} 條邊。")
        return all_pins, edges

# ==============================================================================
# Cell 4: ComponentPlotter 類別
# ==============================================================================
class ComponentPlotter:
    """
    視覺化工具，可以繪製元件、邊(edges)，以及僅繪製被連接的引腳(pins)。
    新版支援繪製非階層性新增的元件 (如 GapFiller)。
    """
    def _draw_recursive(self, ax, component: Component):
        top_left_x, top_left_y = component.get_topleft()
        width, height, level = component.width, component.height, component.level
        LEVEL_COLORS = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#E0BBE4', '#FFD1DC', '#B2DFDB']
        color = LEVEL_COLORS[level % len(LEVEL_COLORS)]
        rect = plt.Rectangle((top_left_x, top_left_y), width, height,
                             linewidth=1.2, edgecolor='black', facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        
        label = f"L{level}\nID:{component.relation_id}"
        if component.symmetric_group_id != -1:
            label += f"\nS:{component.symmetric_group_id}"
            
        ax.text(component.x, component.y, label, ha='center', va='center', fontsize=8, color='black')
        
        if component.sub_components:
            for sub_comp in component.sub_components:
                self._draw_recursive(ax, sub_comp)

    def _draw_netlist(self, ax, edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        if not edges:
            return

        connected_pins = set()
        for p1, p2 in edges:
            connected_pins.add(p1)
            connected_pins.add(p2)
        
        print(f"[*] 正在繪製 {len(edges)} 條邊...")
        for p1, p2 in edges:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#555555', linestyle='-', linewidth=0.7, alpha=0.6)
            
        print(f"[*] 正在繪製 {len(connected_pins)} 個已連接的引腳...")
        for px, py in connected_pins:
            ax.plot(px, py, 'o', color='black', markersize=2.5, alpha=0.8)

    def plot(self, components_to_plot: List[Component], title: str = "Component Layout",
             edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
             output_filename: str = "component_visualization.png",
             canvas_dim: float = None):  # ✨ [修改 1] 新增 canvas_dim 參數
        """
        繪製元件和 Netlist (邊與已連接的引腳)。
        """
        fig, ax = plt.subplots(1, figsize=(12, 12)) # 稍微調整圖表大小
        
        if not components_to_plot:
            ax.set_title("元件列表為空")
            plt.close(fig) # 避免顯示空圖
            return

        # 1. 繪製所有元件
        for comp in components_to_plot:
            self._draw_recursive(ax, comp)
        
        # 2. 繪製連線
        if edges:
            self._draw_netlist(ax, edges)
        
        # ✨ [修改 2] 設定視野範圍
        # 如果有傳入 canvas_dim (例如 1000)，就使用固定的畫布範圍 [-500, 500]
        if canvas_dim:
            half_dim = canvas_dim / 2
            ax.set_xlim(-half_dim, half_dim)
            ax.set_ylim(-half_dim, half_dim)
            
            # 畫出畫布邊界框，讓視覺上更清楚範圍
            import matplotlib.patches as patches
            canvas_rect = patches.Rectangle((-half_dim, -half_dim), canvas_dim, canvas_dim,
                                            linewidth=2, edgecolor='red', facecolor='none', 
                                            linestyle='--', alpha=0.5, label='Canvas Boundary')
            ax.add_patch(canvas_rect)
            # ax.legend(loc='upper right') # 選擇性開啟圖例
            
        else:
            # 舊邏輯：如果沒設定 canvas_dim，則自動貼合元件邊界 (Fallback)
            root = components_to_plot[0]
            padding = 20
            ax.set_xlim(root.x - root.width/2 - padding, root.x + root.width/2 + padding)
            ax.set_ylim(root.y - root.height/2 - padding, root.y + root.height/2 + padding)

        ax.set_aspect('equal', adjustable='box')
        plt.title(title, fontsize=16)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.savefig(output_filename, dpi=150)
        print(f"✅ 繪圖完成！圖片已儲存至 {output_filename}")
        plt.close(fig)

# ==============================================================================
# Cell 5: Level_0 類別
# ==============================================================================
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

# ==============================================================================
# Cell 6: Level_1 類別
# ==============================================================================
class Level_1:
    """
    進階版的 Level_1 處理器，修正了對齊邏輯，嚴格遵守分割方向與對齊模式的綁定關係。
    """
    def __init__(
        self,
        w_h_ratio_bound: tuple[float, float] = (1/6, 6/1),
        max_tries_per_orientation: int = 50,
        num_splits_range: tuple[int, int] = (2, 5),
        ratio_range: tuple[float, float] = (0.3, 1.0),
        split_only_probability: float = 0.5,
        align_scale_factor_range: tuple[float, float] = (0.2, 1.0),
        force_align_threshold: int = 3,
        symmetric_split_probability: float = 0.3,
        adaptive_symmetric_target_ratio: float = 1.5
    ):
        self.w_h_ratio_bound = w_h_ratio_bound
        self.max_tries_per_orientation = max_tries_per_orientation
        self.num_splits_range = num_splits_range
        self.ratio_range = ratio_range
        self.split_only_probability = split_only_probability
        self.align_scale_factor_range = align_scale_factor_range
        self.force_align_threshold = force_align_threshold
        self.symmetric_split_probability = symmetric_split_probability
        self.adaptive_symmetric_target_ratio = adaptive_symmetric_target_ratio
        self.level = 1

    def _apply_adaptive_symmetric_split(self, parent_component: Component) -> List[Component]:
        """
        [新版] 執行三明治切割，並保留中間元件，形成三元對稱結構。
        """
        parent_w = parent_component.width
        parent_h = parent_component.height
        target_ratio = self.adaptive_symmetric_target_ratio

        if parent_component.w_h_ratio() > 1:
            ideal_child_w = parent_h * target_ratio
            if (parent_w / 2) > ideal_child_w and (1 - 2 * (ideal_child_w / parent_w)) > 0:
                ratio = ideal_child_w / parent_w
                sub_components = split_by_ratio(parent_component, [ratio, 1 - 2 * ratio, ratio], SplitOrientation.VERTICAL)
                if len(sub_components) == 3:
                    center_comp = sub_components[1]
                    aspect_ratio = max(center_comp.width, center_comp.height) / min(center_comp.width, center_comp.height)
                    
                    if aspect_ratio > 3:
                        sub_components[0].generate_rule = "symmetric_adaptive_side"
                        sub_components[2].generate_rule = "symmetric_adaptive_side"
                        return [sub_components[0], sub_components[2]]
                    else:
                        sub_components[0].generate_rule = "symmetric_adaptive_side"
                        sub_components[2].generate_rule = "symmetric_adaptive_side"
                        sub_components[1].generate_rule = "symmetric_adaptive_center"
                        return sub_components
            
        elif parent_component.w_h_ratio() < 1:
            ideal_child_h = parent_w * target_ratio
            if (parent_h / 2) > ideal_child_h and (1 - 2 * (ideal_child_h / parent_h)) > 0:
                ratio = ideal_child_h / parent_h
                sub_components = split_by_ratio(parent_component, [ratio, 1 - 2 * ratio, ratio], SplitOrientation.HORIZONTAL)
                if len(sub_components) == 3:
                    center_comp = sub_components[1]
                    aspect_ratio = max(center_comp.width, center_comp.height) / min(center_comp.width, center_comp.height)
                    if aspect_ratio > 3:
                        sub_components[0].generate_rule = "symmetric_adaptive_side"
                        sub_components[2].generate_rule = "symmetric_adaptive_side"
                        return [sub_components[0], sub_components[2]]
                    else:
                        sub_components[0].generate_rule = "symmetric_adaptive_side"
                        sub_components[2].generate_rule = "symmetric_adaptive_side"
                        sub_components[1].generate_rule = "symmetric_adaptive_center"
                        return sub_components

        if parent_component.w_h_ratio() > 1:
            return split_symmetric_1_horizontal(parent_component) 
        else:
            return split_symmetric_1_vertical(parent_component)

    def _find_valid_ratios(self, parent_component: Component, orientation: SplitOrientation, num_splits: int):
        parent_w_h_ratio = parent_component.w_h_ratio()
        min_ratio, max_ratio = self.w_h_ratio_bound
        for _ in range(self.max_tries_per_orientation):
            ratios = [random.uniform(*self.ratio_range) for _ in range(num_splits)]
            total_ratio = sum(ratios)
            all_valid = True
            for r in ratios:
                sub_w_h_ratio = 0
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
        if parent_component.w_h_ratio() > 1:
            orientations_to_try = [SplitOrientation.VERTICAL, SplitOrientation.HORIZONTAL]
        else:
            orientations_to_try = [SplitOrientation.HORIZONTAL, SplitOrientation.VERTICAL]

        for orientation in orientations_to_try:
            valid_ratios = self._find_valid_ratios(parent_component, orientation, num_splits)
            if valid_ratios:
                return split_by_ratio(parent_component, valid_ratios, orientation)
        return split_hold(parent_component)

    def _apply_align(self, parent_component: Component, num_splits: int) -> List[Component]:
        align_mode = random.choice(list(AlignmentMode))
        if align_mode in [AlignmentMode.TOP, AlignmentMode.BOTTOM, AlignmentMode.CENTER_H]:
            required_orientation = SplitOrientation.VERTICAL
        else:
            required_orientation = SplitOrientation.HORIZONTAL

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
            
            if final_min_s > final_max_s:
                scale = 1.0 
            else:
                scale = random.uniform(final_min_s, final_max_s)
            
            scale_factors.append(scale)

        return align_components(sub_components, scale_factors, align_mode)

    def _process_single_component(self, parent_component: Component) -> List[Component]:
        if random.random() < self.symmetric_split_probability:
            return self._apply_adaptive_symmetric_split(parent_component)
        
        num_splits = random.randint(*self.num_splits_range)
        if num_splits > self.force_align_threshold:
            return self._apply_align(parent_component, num_splits)
        else:
            if random.random() < self.split_only_probability:
                return self._apply_split(parent_component, num_splits)
            else:
                return self._apply_align(parent_component, num_splits)

    def generate(self, components: List[Component], start_group_id: int) -> Tuple[List[Component], int]:
        all_results = []
        relation_id = 0
        symmetric_group_counter = start_group_id

        for component in components:
            processed_sub_components = self._process_single_component(component)
            
            is_valid = True
            min_r, max_r = self.w_h_ratio_bound
            for sub_comp in processed_sub_components:
                if not (min_r <= sub_comp.w_h_ratio() <= max_r):
                    is_valid = False
                    break
            
            if not is_valid:
                processed_sub_components = split_hold(component)

            is_symmetric_pair = (len(processed_sub_components) == 2 and 
                                processed_sub_components[0].generate_rule in ["symmetric_1", "symmetric_adaptive_side"])
            
            is_adaptive_trio = (len(processed_sub_components) == 3 and 
                                processed_sub_components[0].generate_rule == "symmetric_adaptive_side")

            if is_symmetric_pair:
                for sub_comp in processed_sub_components:
                    sub_comp.symmetric_group_id = symmetric_group_counter
                symmetric_group_counter += 1
            elif is_adaptive_trio:
                processed_sub_components[0].symmetric_group_id = symmetric_group_counter
                processed_sub_components[2].symmetric_group_id = symmetric_group_counter
                symmetric_group_counter += 1
            
            for sub_comp in processed_sub_components:
                sub_comp.level = self.level
                sub_comp.relation_id = relation_id
            all_results.extend(processed_sub_components)
            relation_id += 1
            component.sub_components = processed_sub_components
            
        return all_results, symmetric_group_counter

# ==============================================================================
# Cell 7: Level_2 類別
# ==============================================================================
class Level_2:
    """
    Level 2 產生器 (上下文感知與多樣化策略版)。
    """
    def __init__(
        self,
        large_component_align_probability: float = 1.0,
        wide_threshold: float = 2.0,
        tall_threshold: float = 0.5,
        size_thresholds: Tuple[float, float] = (0.1, 0.4),
        small_component_hold_probability: float = 0.8,
        policy_wide: Dict[str, Any] = None,
        policy_tall: Dict[str, Any] = None,
        policy_square: Dict[str, Any] = None,
        w_h_ratio_bound: tuple[float, float] = (1/6, 6/1),
        max_tries: int = 50,
        ratio_grid_probability: float = 0.5,
        ratio_range: tuple[float, float] = (0.3, 0.6),
        large_component_hold_probability: float = 0.7,
        simple_split_probability: float = 0.9,
        num_splits_range: tuple[int, int] = (2, 4),
        symmetric_split_probability: float = 0.0,
        adaptive_symmetric_target_ratio: float = 1.5
    ):
        self.large_component_align_probability = large_component_align_probability
        self.wide_threshold = wide_threshold
        self.tall_threshold = tall_threshold
        self.size_thresholds = size_thresholds
        self.small_component_hold_probability = small_component_hold_probability
        self.policy_wide = policy_wide or {"rows_range": (1, 2), "cols_range": (3, 5),"h_ratios_num_range": (1, 2), "v_ratios_num_range": (3, 5)}
        self.policy_tall = policy_tall or {"rows_range": (3, 5), "cols_range": (1, 2),"h_ratios_num_range": (3, 5), "v_ratios_num_range": (1, 2)}
        self.policy_square = policy_square or {"rows_range": (2, 4), "cols_range": (2, 4),"h_ratios_num_range": (2, 4), "v_ratios_num_range": (2, 4)}
        self.w_h_ratio_bound = w_h_ratio_bound
        self.max_tries = max_tries
        self.ratio_grid_probability = ratio_grid_probability
        self.ratio_range = ratio_range
        self.level = 2
        self.large_component_hold_probability = large_component_hold_probability
        self.simple_split_probability = simple_split_probability
        self.num_splits_range = num_splits_range
        self.symmetric_split_probability = symmetric_split_probability
        self.adaptive_symmetric_target_ratio = adaptive_symmetric_target_ratio

    def _apply_forced_split(self, comp: Component) -> List[Component]:
        min_r, max_r = self.w_h_ratio_bound
        ratio = comp.w_h_ratio()
        children = []
        if ratio > max_r:
            num_splits = math.ceil(ratio / max_r)
            children = spacing_horizontal(comp, num_splits)
        elif ratio < min_r:
            num_splits = math.ceil(min_r / ratio)
            children = spacing_vertical(comp, num_splits)
        
        for child in children:
            child.level = self.level
            child.relation_id = comp.relation_id
            child.generate_rule = "forced_split"
        return children

    def _apply_adaptive_symmetric_split(self, parent_component: Component) -> List[Component]:
        parent_w = parent_component.width
        parent_h = parent_component.height
        target_ratio = self.adaptive_symmetric_target_ratio

        if parent_component.w_h_ratio() > 1:
            ideal_child_w = parent_h * target_ratio
            if (parent_w / 2) > ideal_child_w and (1 - 2 * (ideal_child_w / parent_w)) > 0:
                ratio = ideal_child_w / parent_w
                sub_components = split_by_ratio(parent_component, [ratio, 1 - 2 * ratio, ratio], SplitOrientation.VERTICAL)
                if len(sub_components) == 3:
                    center_comp = sub_components[1]
                    aspect_ratio = max(center_comp.width, center_comp.height) / min(center_comp.width, center_comp.height)
                    if aspect_ratio > 3:
                        sub_components[0].generate_rule = "symmetric_adaptive_side"
                        sub_components[2].generate_rule = "symmetric_adaptive_side"
                        return [sub_components[0], sub_components[2]]
                    else:
                        sub_components[0].generate_rule = "symmetric_adaptive_side"
                        sub_components[2].generate_rule = "symmetric_adaptive_side"
                        sub_components[1].generate_rule = "symmetric_adaptive_center"
                        return sub_components
        
        elif parent_component.w_h_ratio() < 1:
            ideal_child_h = parent_w * target_ratio
            if (parent_h / 2) > ideal_child_h and (1 - 2 * (ideal_child_h / parent_h)) > 0:
                ratio = ideal_child_h / parent_h
                sub_components = split_by_ratio(parent_component, [ratio, 1 - 2 * ratio, ratio], SplitOrientation.HORIZONTAL)
                if len(sub_components) == 3:
                    center_comp = sub_components[1]
                    aspect_ratio = max(center_comp.width, center_comp.height) / min(center_comp.width, center_comp.height)
                    if aspect_ratio > 3:
                        sub_components[0].generate_rule = "symmetric_adaptive_side"
                        sub_components[2].generate_rule = "symmetric_adaptive_side"
                        return [sub_components[0], sub_components[2]]
                    else:
                        sub_components[0].generate_rule = "symmetric_adaptive_side"
                        sub_components[2].generate_rule = "symmetric_adaptive_side"
                        sub_components[1].generate_rule = "symmetric_adaptive_center"
                        return sub_components

        if parent_component.w_h_ratio() > 1:
            return split_symmetric_1_horizontal(parent_component) 
        else:
            return split_symmetric_1_vertical(parent_component)

    def _find_valid_ratios(self, parent_component: Component, orientation: SplitOrientation, num_splits: int):
        parent_w_h_ratio = parent_component.w_h_ratio()
        min_ratio, max_ratio = self.w_h_ratio_bound
        for _ in range(self.max_tries):
            ratios = [random.uniform(0.3, 1.0) for _ in range(num_splits)]
            total_ratio = sum(ratios)
            all_valid = True
            for r in ratios:
                sub_w_h_ratio = 0
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

    def _apply_simple_split(self, parent_component: Component) -> List[Component]:
        num_splits = random.randint(*self.num_splits_range)
        if parent_component.w_h_ratio() > 1:
            orientations_to_try = [SplitOrientation.VERTICAL, SplitOrientation.HORIZONTAL]
        else:
            orientations_to_try = [SplitOrientation.HORIZONTAL, SplitOrientation.VERTICAL]
        for orientation in orientations_to_try:
            valid_ratios = self._find_valid_ratios(parent_component, orientation, num_splits)
            if valid_ratios:
                return split_by_ratio(parent_component, valid_ratios, orientation)
        return split_hold(parent_component)

    def _apply_advanced_align(self, parent_component: Component, siblings_bbox: Dict[str, float]) -> List[Component]:
        valid_align_modes = []; epsilon = 1e-6
        p_left, p_top = parent_component.get_topleft()
        p_right, p_bottom = parent_component.get_bottomright()
        is_top_edge = abs(p_top - siblings_bbox['min_y']) < epsilon
        is_bottom_edge = abs(p_bottom - siblings_bbox['max_y']) < epsilon
        is_left_edge = abs(p_left - siblings_bbox['min_x']) < epsilon
        is_right_edge = abs(p_right - siblings_bbox['max_x']) < epsilon
        if is_top_edge: valid_align_modes.append(AlignmentMode.BOTTOM)
        if is_bottom_edge: valid_align_modes.append(AlignmentMode.TOP)
        if is_left_edge: valid_align_modes.append(AlignmentMode.RIGHT)
        if is_right_edge: valid_align_modes.append(AlignmentMode.LEFT)
        if not (is_top_edge or is_bottom_edge): valid_align_modes.append(AlignmentMode.CENTER_V)
        if not (is_left_edge or is_right_edge): valid_align_modes.append(AlignmentMode.CENTER_H)
        if not valid_align_modes:
            align_mode = AlignmentMode.CENTER_H if parent_component.w_h_ratio() <= 1 else AlignmentMode.CENTER_V
        else:
            align_mode = random.choice(valid_align_modes)
        if align_mode in [AlignmentMode.TOP, AlignmentMode.BOTTOM, AlignmentMode.CENTER_H]:
            orientation = SplitOrientation.VERTICAL
        else:
            orientation = SplitOrientation.HORIZONTAL
        num_splits = random.randint(2, 4)
        valid_ratios = [random.uniform(0.5, 1.0) for _ in range(num_splits)]
        sub_components = split_by_ratio(parent_component, valid_ratios, orientation)
        scale_factors = [random.uniform(0.6, 0.95) for _ in range(num_splits)]
        return align_components(sub_components, scale_factors, align_mode)

    def _apply_grid_split(self, parent_component: Component, size_ratio: float) -> List[Component]:
        shape_ratio = parent_component.w_h_ratio()
        base_policy = self.policy_square
        if shape_ratio > self.wide_threshold: base_policy = self.policy_wide
        elif shape_ratio < self.tall_threshold: base_policy = self.policy_tall
        final_policy = self._get_dynamic_policy(base_policy, size_ratio)
        if random.random() < self.ratio_grid_probability:
            return self._apply_ratio_grid(parent_component, final_policy)
        else:
            return self._apply_spacing_grid(parent_component, final_policy)

    def generate(self, components: List[Component], root_component: Component, start_group_id: int) -> Tuple[List[Component], int]:
        if not components or not root_component:
            return [], start_group_id
        root_area = root_component.width * root_component.height
        min_x = min(c.get_topleft()[0] for c in components); max_x = max(c.get_bottomright()[0] for c in components)
        min_y = min(c.get_topleft()[1] for c in components); max_y = max(c.get_bottomright()[1] for c in components)
        siblings_bbox = {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}
        small_thresh, large_thresh = self.size_thresholds
        alignment_candidates = [c for c in components if (c.width * c.height) / root_area > large_thresh]
        component_to_align = random.choice(alignment_candidates) if alignment_candidates and random.random() < self.large_component_align_probability else None
        all_results = []
        relation_id = 0
        symmetric_group_counter = start_group_id
        for comp in components:
            if comp.symmetric_group_id != -1:
                group_id_to_break = comp.symmetric_group_id
                for member_comp in components:
                    if member_comp.symmetric_group_id == group_id_to_break:
                        member_comp.symmetric_group_id = -1
                        member_comp.generate_rule = "symmetry_broken_by_L2"
            
            processed_sub_components = []
            min_r, max_r = self.w_h_ratio_bound
            
            if not (min_r <= comp.w_h_ratio() <= max_r):
                processed_sub_components = self._apply_forced_split(comp)
            else:
                if comp is component_to_align:
                    processed_sub_components = self._apply_advanced_align(comp, siblings_bbox)
                elif random.random() < self.symmetric_split_probability:
                    processed_sub_components = self._apply_adaptive_symmetric_split(comp)
                else:
                    size_ratio = (comp.width * comp.height) / root_area
                    is_large = size_ratio > large_thresh
                    is_small = size_ratio < small_thresh
                    if is_large and random.random() < self.large_component_hold_probability:
                        processed_sub_components = split_hold(comp)
                    elif is_small and random.random() < self.small_component_hold_probability:
                        processed_sub_components = split_hold(comp)
                    else:
                        if random.random() < self.simple_split_probability:
                            processed_sub_components = self._apply_simple_split(comp)
                        else:
                            processed_sub_components = self._apply_grid_split(comp, size_ratio)
            
            is_valid = True
            for sub_comp in processed_sub_components:
                if not (min_r <= sub_comp.w_h_ratio() <= max_r):
                    is_valid = False
                    break
            
            if not is_valid:
                processed_sub_components = split_hold(comp)

            is_symmetric_pair = (len(processed_sub_components) == 2 and 
                                processed_sub_components[0].generate_rule in ["symmetric_1", "symmetric_adaptive_side"])
            is_adaptive_trio = (len(processed_sub_components) == 3 and 
                                processed_sub_components[0].generate_rule == "symmetric_adaptive_side")

            if is_symmetric_pair:
                for sub_comp in processed_sub_components:
                    sub_comp.symmetric_group_id = symmetric_group_counter
                symmetric_group_counter += 1
            elif is_adaptive_trio:
                processed_sub_components[0].symmetric_group_id = symmetric_group_counter
                processed_sub_components[2].symmetric_group_id = symmetric_group_counter
                symmetric_group_counter += 1
            
            for sub_comp in processed_sub_components:
                sub_comp.level = self.level
                sub_comp.relation_id = relation_id
            all_results.extend(processed_sub_components)
            relation_id += 1
            comp.sub_components = processed_sub_components
            
        return all_results, symmetric_group_counter

    def _get_dynamic_policy(self, base_policy: Dict[str, Any], size_ratio: float) -> Dict[str, Any]:
        small_thresh, large_thresh = self.size_thresholds; dynamic_policy = base_policy.copy();
        if size_ratio < small_thresh: scale_factor = 0.5 
        elif size_ratio > large_thresh: scale_factor = 1.5 
        else: return dynamic_policy
        for key in ["rows_range", "cols_range", "h_ratios_num_range", "v_ratios_num_range"]:
            min_val, max_val = dynamic_policy[key]; new_min = max(1, int(min_val * scale_factor)); new_max = max(new_min, int(max_val * scale_factor)); dynamic_policy[key] = (new_min, new_max)
        return dynamic_policy
    def _apply_ratio_grid(self, parent_component: Component, policy: Dict[str, Any]) -> List[Component]:
        h_ratios_num_range = policy["h_ratios_num_range"]; v_ratios_num_range = policy["v_ratios_num_range"]
        for _ in range(self.max_tries):
            num_h = random.randint(*h_ratios_num_range); num_v = random.randint(*v_ratios_num_range)
            h_ratios = [random.uniform(*self.ratio_range) for _ in range(num_h)]; v_ratios = [random.uniform(*self.ratio_range) for _ in range(num_v)]
            return split_by_ratio_grid(parent_component, h_ratios, v_ratios)
        return split_hold(parent_component)
    def _apply_spacing_grid(self, parent_component: Component, policy: Dict[str, Any]) -> List[Component]:
        rows_range = policy["rows_range"]; cols_range = policy["cols_range"]
        for _ in range(self.max_tries):
            rows = random.randint(*rows_range); cols = random.randint(*cols_range)
            return spacing_grid(parent_component, rows, cols)
        return split_hold(parent_component)

# ==============================================================================
# Cell 8: GapFiller 類別
# ==============================================================================
class GapFiller:
    """
    Finds the longest available edge in a layout and places a row of 
    small, aligned components along it.
    """
    def __init__(self,
                 small_comp_w_range: tuple[float, float] = (6, 14),
                 small_comp_h_range: tuple[float, float] = (6, 14),
                 spacing: float = 0.5):
        self.w_range = small_comp_w_range
        self.h_range = small_comp_h_range
        self.spacing = spacing
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
            if (new_left < comp_right and new_right > comp_left and
                new_top < comp_bottom and new_bottom > comp_top):
                return True
        return False

    def fill(self, existing_leaf_components: List[Component], root_component: Component, num_to_place: int) -> List[Component]:
        if not existing_leaf_components or num_to_place == 0:
            return []

        best_host = None
        best_edge_type = ''
        max_edge_len = -1.0
        
        for comp in existing_leaf_components:
            if comp.width > max_edge_len:
                max_edge_len = comp.width
                best_host = comp
                best_edge_type = random.choice(['top', 'bottom'])
            if comp.height > max_edge_len:
                max_edge_len = comp.height
                best_host = comp
                best_edge_type = random.choice(['left', 'right'])

        if not best_host:
            return []

        gap_components = []
        all_components = existing_leaf_components.copy()
        h_left, h_top = best_host.get_topleft()
        h_right, h_bottom = best_host.get_bottomright()

        cursor = 0
        if best_edge_type in ['bottom', 'top']:
            cursor = h_left
        elif best_edge_type in ['left', 'right']:
            cursor = h_top

        for _ in range(num_to_place):
            new_w = random.uniform(*self.w_range)
            new_h = random.uniform(*self.h_range)
            new_comp = Component(x=0, y=0, width=new_w, height=new_h, level=self.level, relation_id=-1)
            
            if best_edge_type == 'bottom':
                if cursor + new_w > h_right: break
                new_comp.x = cursor + new_w / 2
                new_comp.y = h_bottom + new_h / 2
            elif best_edge_type == 'top':
                if cursor + new_w > h_right: break
                new_comp.x = cursor + new_w / 2
                new_comp.y = h_top - new_h / 2
            elif best_edge_type == 'right':
                if cursor + new_h > h_bottom: break
                new_comp.x = h_right + new_w / 2
                new_comp.y = cursor + new_h / 2
            elif best_edge_type == 'left':
                if cursor + new_h > h_bottom: break
                new_comp.x = h_left - new_w / 2
                new_comp.y = cursor + new_h / 2

            if not self._check_collision(new_comp, all_components, root_component):
                gap_components.append(new_comp)
                all_components.append(new_comp)
                if best_edge_type in ['top', 'bottom']:
                    cursor += new_w + self.spacing
                else:
                    cursor += new_h + self.spacing
            else:
                break
        
        return gap_components

# ==============================================================================
# Cell 9: JSON 匯出函式
# ==============================================================================
def component_to_dict(component: Component) -> Dict[str, Any]:
    """
    遞迴地將一個 Component 物件及其所有子元件轉換成字典格式。
    """
    if component is None:
        return None
    
    sub_components_list = []
    if component.sub_components:
        sub_components_list = [component_to_dict(sub) for sub in component.sub_components]

    return {
        "x": component.x,
        "y": component.y,
        "width": component.width,
        "height": component.height,
        "level": component.level,
        "relation_id": component.relation_id,
        "generate_rule": component.generate_rule,
        "symmetric_group_id": component.symmetric_group_id,
        "sub_components": sub_components_list
    }

def export_layout_to_json(
    layout_id: int,
    seed_used: int,
    root_component: Component,
    gap_components: List[Component],
    final_leaf_components: List[Component],
    edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    output_path: str
):
    """
    將完整的佈局資料（包含使用的種子）匯出成一個 JSON 檔案。
    """
    root_dict = component_to_dict(root_component)
    gap_dicts = [component_to_dict(comp) for comp in gap_components]
    leaf_dicts = [component_to_dict(comp) for comp in final_leaf_components]
    
    layout_data = {
        "layout_id": layout_id,
        "seed_used": seed_used,
        "root_component": root_dict,
        "gap_components": gap_dicts,
        "final_leaf_components": leaf_dicts,
        "netlist_edges": edges
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=4)
        print(f"📄 佈局資料已成功儲存至 {output_path}")
    except Exception as e:
        print(f"❌ 儲存 JSON 檔案至 {output_path} 時發生錯誤: {e}")

# ==============================================================================
# Cell 10: 主要執行函式
# ==============================================================================
def main_execution_batch_from_yaml():
    """
    [新版] 實現了跨層級的對稱群組 ID 管理和最終驗證。
    """
    config = load_yaml_config('config.yaml')
    if config is None:
        return
        
    path_config = config.get('path_settings', {})
    main_config = config.get('main_execution', {})
    
    raw_output_dir = path_config.get('raw_output_directory', 'raw_layouts')
    num_to_generate = main_config.get('num_layouts_to_generate', 1)
    file_basename = os.path.basename(raw_output_dir)
    image_subdir = path_config.get('image_subdirectory', 'images')
    json_subdir = path_config.get('json_subdirectory', 'json_data')
    image_output_folder = os.path.join(raw_output_dir, image_subdir)
    json_output_folder = os.path.join(raw_output_dir, json_subdir)
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(json_output_folder, exist_ok=True)
    
    print(f"📂 圖片將儲存於: '{image_output_folder}'")
    print(f"📂 JSON 資料將儲存於: '{json_output_folder}'")
    print(f"🚀 批次產生任務啟動，預計產生 {num_to_generate} 套資料...")
    print("-" * 50)

    # ✨ [修改 1] 從 config 讀取畫布大小，預設為 1000
    ml_config = config.get('ml_preparation', {})
    target_canvas_dim = ml_config.get('target_canvas_dim', 1000.0)

    for i in range(num_to_generate):
        current_seed = random.randint(0, 2**32 - 1)
        random.seed(current_seed)
        np.random.seed(current_seed)
        
        print(f"=============== 正在產生資料組 #{i+1}/{num_to_generate} (Seed: {current_seed}) ===============")

        level_0_generator = Level_0(**config.get('Level_0', {}))
        level_1_generator = Level_1(**config.get('Level_1', {}))
        level_2_generator = Level_2(**config.get('Level_2', {}))
        gap_filler = GapFiller(**config.get('GapFiller', {}))
        netlist_generator = NetlistGenerator(**config.get('NetlistGenerator', {}))

        symmetric_group_counter = 0
        
        root_components = level_0_generator.generate()
        root_component = root_components[0]
        
        level_1_components, symmetric_group_counter = level_1_generator.generate(root_components, symmetric_group_counter)
        level_2_components, symmetric_group_counter = level_2_generator.generate(level_1_components, root_component, symmetric_group_counter)

        symmetric_groups = defaultdict(list)
        for l2_comp in level_2_components:
            if l2_comp.symmetric_group_id != -1:
                symmetric_groups[l2_comp.symmetric_group_id].append(l2_comp)

        for group_id, members in symmetric_groups.items():
            if len(members) != 2:
                for member in members:
                    member.symmetric_group_id = -1
                    member.generate_rule = "symmetry_invalidated_post_check"

        num_gaps_to_fill = main_config.get('num_gaps_to_fill', 0)
        gap_filler_threshold = main_config.get('gap_filler_activation_threshold', 0.2)
        gap_components = []
        occupied_area = sum(c.width * c.height for c in level_2_components)
        total_area = root_component.width * root_component.height
        if total_area > 0 and (total_area - occupied_area) / total_area > gap_filler_threshold:
            gap_components = gap_filler.fill(level_2_components, root_component, num_gaps_to_fill)

        final_leaf_components = level_2_components + gap_components
        _, edges = netlist_generator.generate(final_leaf_components)

        plotter = ComponentPlotter()
        components_to_plot = root_components + gap_components
        current_title = f"{main_config.get('output_title', 'Layout')} #{i} (Seed: {current_seed})"
        png_filename = f"{file_basename}_{i}.png"
        png_output_path = os.path.join(image_output_folder, png_filename)
        # ✨ [修改 2] 呼叫 plot 時傳入 canvas_dim
        plotter.plot(
            components_to_plot, 
            title=current_title, 
            edges=edges, 
            output_filename=png_output_path,
            canvas_dim=target_canvas_dim  # <--- 傳入 1000
        )
    
        json_filename = f"{file_basename}_{i}.json"
        json_output_path = os.path.join(json_output_folder, json_filename)
        
        export_layout_to_json(
            layout_id=i,
            seed_used=current_seed,
            root_component=root_component,
            gap_components=gap_components,
            final_leaf_components=final_leaf_components,
            edges=edges,
            output_path=json_output_path
        )
        print("-" * 50)

    print(f"✨ 所有批次任務執行完畢！ ✨")

# ==============================================================================
# 主程式進入點
# ==============================================================================
if __name__ == "__main__":
    # 執行使用 YAML 設定檔的批次產生流程
    main_execution_batch_from_yaml()