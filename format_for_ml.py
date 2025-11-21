# format_for_ml.py

import os
import json
import glob
import yaml
from typing import List, Dict, Any, Tuple
from collections import defaultdict

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 設定檔讀取錯誤: {e}")
        return None

def find_parent_component_index(pin_coords: Tuple[float, float], components: List[Dict[str, Any]]) -> int:
    px, py = pin_coords
    for i, comp in enumerate(components):
        left = comp['x'] - comp['width'] / 2
        right = comp['x'] + comp['width'] / 2
        top = comp['y'] - comp['height'] / 2
        bottom = comp['y'] + comp['height'] / 2
        if (left - 1e-6) <= px <= (right + 1e-6) and (top - 1e-6) <= py <= (bottom + 1e-6):
            return i
    return None

def format_single_layout(input_path: str, output_path: str, target_canvas_dim: float):
    print(f"🔄 正在處理: {os.path.basename(input_path)}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 讀取錯誤 {input_path}: {e}")
        return

    leaf_components = data.get("final_leaf_components", [])
    root_component = data.get("root_component", None)
    
    # ✨ [新增] 讀取原始資料中的 Seed 和 ID
    seed_used = data.get("seed_used", "Unknown")
    layout_id = data.get("layout_id", "Unknown")

    if not leaf_components:
        print(f"⚠️ 警告：無 final_leaf_components，跳過。")
        return

    netlist_edges = data.get("netlist_edges", [])

    # 1. 設定錨點
    if root_component:
        content_center_x = root_component['x']
        content_center_y = root_component['y']
    else:
        # Fallback
        min_x = min(comp['x'] - comp['width'] / 2 for comp in leaf_components)
        max_x = max(comp['x'] + comp['width'] / 2 for comp in leaf_components)
        min_y = min(comp['y'] - comp['height'] / 2 for comp in leaf_components)
        max_y = max(comp['y'] + comp['height'] / 2 for comp in leaf_components)
        content_center_x = (min_x + max_x) / 2
        content_center_y = (min_y + max_y) / 2
    
    # 2. 計算縮放因子
    scale_factor = target_canvas_dim / 2.0
    if scale_factor == 0: scale_factor = 1.0

    ml_nodes = []
    ml_targets = []
    ml_sub_components = []
    
    # 3. 正規化元件
    for comp in leaf_components:
        norm_w = comp['width'] / scale_factor
        norm_h = comp['height'] / scale_factor
        ml_nodes.append([norm_w, norm_h])

        shifted_x = comp['x'] - content_center_x
        shifted_y = comp['y'] - content_center_y
        norm_x = shifted_x / scale_factor
        norm_y = shifted_y / scale_factor
        ml_targets.append([norm_x, norm_y])
        
        ml_sub_components.append([
            {
                "offset": [0.0, 0.0],
                "dims": [comp['width'], comp['height']]
            }
        ])

    # 4. 正規化連線
    basic_component_edges = []
    for edge in netlist_edges:
        src_pin_abs = tuple(edge[0])
        dest_pin_abs = tuple(edge[1])
        
        src_comp_idx = find_parent_component_index(src_pin_abs, leaf_components)
        dest_comp_idx = find_parent_component_index(dest_pin_abs, leaf_components)
        
        if src_comp_idx is not None and dest_comp_idx is not None:
            src_comp = leaf_components[src_comp_idx]
            dest_comp = leaf_components[dest_comp_idx]
            
            src_offset_x = (src_pin_abs[0] - src_comp['x']) / scale_factor
            src_offset_y = (src_pin_abs[1] - src_comp['y']) / scale_factor
            dest_offset_x = (dest_pin_abs[0] - dest_comp['x']) / scale_factor
            dest_offset_y = (dest_pin_abs[1] - dest_comp['y']) / scale_factor
            
            basic_component_edges.append([
                [src_comp_idx, dest_comp_idx],
                [src_offset_x, src_offset_y, dest_offset_x, dest_offset_y]
            ])

    # 5. 對稱群組
    symmetry_groups_map = defaultdict(list)
    for i, comp in enumerate(leaf_components):
        group_id = comp.get("symmetric_group_id", -1)
        if group_id != -1:
            symmetry_groups_map[group_id].append(i)
    ml_symmetry_groups = [indices for indices in symmetry_groups_map.values() if len(indices) == 2]

    # 6. 儲存結果
    ml_data = {
        "metadata": {
            "normalization_mode": "fixed_canvas_from_config",
            "target_canvas_dim": target_canvas_dim,
            "scale_factor": scale_factor,
            # ✨ [新增] 將 Seed 和 Layout ID 存入 Metadata
            "seed_used": seed_used,
            "layout_id": layout_id
        },
        "node": ml_nodes,
        "target": ml_targets,
        "edges": { "basic_component_edge": basic_component_edges, "align_edge": [], "group_edge": [] },
        "sub_components": ml_sub_components,
        "symmetry_groups": ml_symmetry_groups
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ml_data, f, indent=2)
        print(f"✅ 轉換成功: {os.path.basename(output_path)} (Seed: {seed_used})")
    except Exception as e:
        print(f"❌ 寫入錯誤: {e}")

def main():
    config = load_config()
    if not config: return

    path_cfg = config.get('path_settings', {})
    ml_cfg = config.get('ml_preparation', {})
    target_canvas_dim = ml_cfg.get('target_canvas_dim', 1000.0)

    print(f"--- 開始執行佈局資料正規化 (Target Canvas: {target_canvas_dim}) ---")

    raw_dir = path_cfg.get('raw_output_directory')
    ml_dir = path_cfg.get('ml_ready_output_directory')
    
    if not raw_dir or not ml_dir:
        print("❌ 錯誤：路徑設定遺失。")
        return

    input_folder = os.path.join(raw_dir, path_cfg.get('json_subdirectory', 'json_data'))
    os.makedirs(ml_dir, exist_ok=True)
    
    input_files = glob.glob(os.path.join(input_folder, '*.json'))
    
    if not input_files:
        print(f"⚠️ 找不到檔案。")
        return

    print(f"🔍 發現 {len(input_files)} 個原始檔案。")
    for input_file_path in input_files:
        basename = os.path.basename(input_file_path).replace('.json', '')
        suffix = basename.split('_')[-1]
        output_filename = f"formatted_{suffix}.json"
        output_file_path = os.path.join(ml_dir, output_filename)
        
        format_single_layout(input_file_path, output_file_path, target_canvas_dim)

    print("✨ 正規化完成！現在請執行 format_visualization.py。 ✨")

if __name__ == "__main__":
    main()