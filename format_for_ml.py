# -*- coding: utf-8 -*-
"""
此腳本用於將 `production.ipynb` 產生的佈局 JSON 資料，
轉換為機器學習模型（如 Chip Placement with Diffusion Models 論文中所述）所需的格式。
"""

import os
import json
import glob
import yaml
from typing import List, Dict, Any, Tuple

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """從指定的路徑載入 YAML 設定檔。"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到設定檔 '{config_path}'。")
        return None
    except yaml.YAMLError as e:
        print(f"❌ 錯誤：解析 YAML 檔案 '{config_path}' 失敗: {e}")
        return None

def find_parent_component_index(pin_coords: Tuple[float, float], components: List[Dict[str, Any]]) -> int:
    """
    根據 pin 的絕對座標，在元件列表中找到其所屬的父元件索引。
    """
    px, py = pin_coords
    for i, comp in enumerate(components):
        left = comp['x'] - comp['width'] / 2
        right = comp['x'] + comp['width'] / 2
        top = comp['y'] - comp['height'] / 2
        bottom = comp['y'] + comp['height'] / 2
        # 增加一個微小的容錯區間
        if (left - 1e-6) <= px <= (right + 1e-6) and (top - 1e-6) <= py <= (bottom + 1e-6):
            return i
    return None

def format_single_layout(input_path: str, output_path: str):
    """
    將單一的 layout.json 檔案轉換為 ML-ready 格式。
    """
    print(f"🔄 正在處理: {os.path.basename(input_path)}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    root_comp = data.get("root_component", {})
    canvas_width = root_comp.get("width")
    canvas_height = root_comp.get("height")

    if not canvas_width or not canvas_height:
        print(f"⚠️ 警告：無法從 {input_path} 取得畫布尺寸，跳過此檔案。")
        return

    leaf_components = data.get("final_leaf_components", [])
    netlist_edges = data.get("netlist_edges", [])
    
    # 初始化 ML 格式的各個部分
    ml_nodes = []
    ml_targets = []
    ml_sub_components = []
    
    # 處理 nodes, targets, 和 sub_components
    for comp in leaf_components:
        # Node: 正規化的 [width, height]
        norm_w = comp['width'] / canvas_width
        norm_h = comp['height'] / canvas_height
        ml_nodes.append([norm_w, norm_h])

        # Target: 正規化的 [x, y]，範圍 [-1, 1]
        norm_x = comp['x'] / (canvas_width / 2)
        norm_y = comp['y'] / (canvas_height / 2)
        ml_targets.append([norm_x, norm_y])
        
        # Sub-components: 論文格式要求
        ml_sub_components.append([
            {
                "offset": [0.0, 0.0],
                "dims": [comp['width'], comp['height']]
            }
        ])

    # 處理 edges
    basic_component_edges = []
    for edge in netlist_edges:
        src_pin_abs, dest_pin_abs = tuple(edge[0]), tuple(edge[1])
        
        src_comp_idx = find_parent_component_index(src_pin_abs, leaf_components)
        dest_comp_idx = find_parent_component_index(dest_pin_abs, leaf_components)
        
        if src_comp_idx is not None and dest_comp_idx is not None:
            src_comp = leaf_components[src_comp_idx]
            dest_comp = leaf_components[dest_comp_idx]

            # 計算正規化的 pin 偏移量 (相對於畫布半徑)
            src_offset_x = (src_pin_abs[0] - src_comp['x']) / (canvas_width / 2)
            src_offset_y = (src_pin_abs[1] - src_comp['y']) / (canvas_height / 2)
            dest_offset_x = (dest_pin_abs[0] - dest_comp['x']) / (canvas_width / 2)
            dest_offset_y = (dest_pin_abs[1] - dest_comp['y']) / (canvas_height / 2)
            
            basic_component_edges.append([
                [src_comp_idx, dest_comp_idx],
                [src_offset_x, src_offset_y, dest_offset_x, dest_offset_y]
            ])

    # 組合最終的 ML-ready JSON 物件
    ml_data = {
        "node": ml_nodes,
        "target": ml_targets,
        "edges": {
            "basic_component_edge": basic_component_edges,
            "align_edge": [], # 根據範例，保留空列表
            "group_edge": []  # 根據範例，保留空列表
        },
        "sub_components": ml_sub_components
    }

    # 寫入檔案
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ml_data, f, indent=2) # 使用 indent=2 節省空間
        print(f"✅ 成功轉換並儲存至: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"❌ 寫入 {output_path} 時發生錯誤: {e}")

def main():
    """主執行函式"""
    print("--- 開始執行佈局資料轉換任務 ---")
    config = load_config()
    if not config:
        return

    path_cfg = config.get('path_settings', {})
    raw_dir = path_cfg.get('raw_output_directory')
    ml_dir = path_cfg.get('ml_ready_output_directory')
    json_subdir = path_cfg.get('json_subdirectory', 'json_data')
    
    if not raw_dir or not ml_dir:
        print("❌ 錯誤：config.yaml 中缺少 'raw_output_directory' 或 'ml_ready_output_directory' 設定。")
        return

    input_folder = os.path.join(raw_dir, json_subdir)
    output_folder = ml_dir

    # 確保輸出目錄存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 尋找所有原始 layout JSON 檔案
    input_files = glob.glob(os.path.join(input_folder, '*.json'))

    if not input_files:
        print(f"⚠️ 在 '{input_folder}' 中找不到任何 .json 檔案。")
        return
        
    print(f"🔍 發現 {len(input_files)} 個 JSON 檔案，準備進行轉換...")
    print("-" * 40)

    for input_file_path in input_files:
        basename = os.path.basename(input_file_path)
        # 產生對應的輸出檔名，例如 raw_layouts_1.json -> formatted_1.json
        parts = basename.split('_')
        if len(parts) > 1 and parts[-1].replace('.json', '').isdigit():
            file_index = parts[-1].replace('.json', '')
            output_filename = f"formatted_{file_index}.json"
        else:
            # 如果檔名格式不符，則使用原始檔名
            output_filename = f"formatted_{basename}"
            
        output_file_path = os.path.join(output_folder, output_filename)
        format_single_layout(input_file_path, output_file_path)
        print("-" * 20)

    print("✨ 所有檔案轉換完畢！ ✨")

if __name__ == "__main__":
    main()
