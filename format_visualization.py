# format_visualization.py

# -*- coding: utf-8 -*-
import os
import json
import glob
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Tuple

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 讀取設定檔失敗: {e}")
        return None

def plot_formatted_layout(data: Dict[str, Any], output_path: str):
    nodes = data.get("node", [])
    targets = data.get("target", [])
    edges = data.get("edges", {}).get("basic_component_edge", [])
    symmetry_groups = data.get("symmetry_groups", [])
    metadata = data.get("metadata", {})

    # 1. 讀取 Metadata (僅用於標題資訊，不需用於縮放)
    seed_used = metadata.get("seed_used", "Unknown")
    layout_id = metadata.get("layout_id", "?")
    
    if not nodes or not targets:
        return

    # 設定圖片大小
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('white')

    # 2. 準備顏色
    SYMMETRY_COLORS = ["#f7a29b", "#8cf795", "#aca5fa", "#e9be7d", "#8ef7cd", "#d091fa"]
    component_colors = {}
    for i, pair in enumerate(symmetry_groups):
        color = SYMMETRY_COLORS[i % len(SYMMETRY_COLORS)]
        if len(pair) == 2:
            component_colors[pair[0]] = color
            component_colors[pair[1]] = color

    # 3. 繪製元件 (直接使用正規化數值)
    for i in range(len(nodes)):
        # 讀取正規化數值 [-1, 1]
        norm_w, norm_h = nodes[i]
        norm_x, norm_y = targets[i]
        
        # --- [修改] 不進行反正規化，直接使用 ---
        # 寬高與中心點
        w = norm_w
        h = norm_h
        center_x = norm_x
        center_y = norm_y
        
        # 計算左上角座標 (正規化空間)
        top_left_x = center_x - w / 2
        top_left_y = center_y - h / 2
        
        # 決定顏色
        if i in component_colors:
            face_color = component_colors[i]
        else:
            face_color = "#B0DCFF"

        rect = patches.Rectangle(
            (top_left_x, top_left_y), w, h,
            linewidth=1.2, edgecolor='black', facecolor=face_color, alpha=0.9
        )
        ax.add_patch(rect)
        
        # 標示 ID
        ax.text(center_x, center_y, f"ID:{i}", ha='center', va='center', fontsize=8, color='black')

    # 4. 繪製連線 (直接使用正規化數值)
    for edge_info in edges:
        indices, offsets = edge_info
        src_idx, dest_idx = indices
        # Offset 也是正規化過的
        src_off_x, src_off_y, dest_off_x, dest_off_y = offsets

        if src_idx < len(nodes) and dest_idx < len(nodes):
            # --- [修改] 計算 Pin 的正規化絕對位置 ---
            # 直接將 元件中心(norm) + 偏移量(norm)
            src_pin_x = targets[src_idx][0] + src_off_x
            src_pin_y = targets[src_idx][1] + src_off_y
            dest_pin_x = targets[dest_idx][0] + dest_off_x
            dest_pin_y = targets[dest_idx][1] + dest_off_y

            ax.plot([src_pin_x, dest_pin_x], [src_pin_y, dest_pin_y], color='#555555', linestyle='-', linewidth=0.8, alpha=0.6)
            ax.plot(src_pin_x, src_pin_y, 'o', color='black', markersize=2.5)
            ax.plot(dest_pin_x, dest_pin_y, 'o', color='black', markersize=2.5)

    # 5. 設定視野範圍與外觀
    ax.set_aspect('equal', adjustable='box')
    
    # ✨ [修改] 強制設定範圍為 [-1, 1]
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    
    # 開啟格線
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 標題與軸標籤
    plt.title(f"Normalized Layout #{layout_id} (Seed: {seed_used})", fontsize=16)
    plt.xlabel("Normalized X [-1, 1]")
    plt.ylabel("Normalized Y [-1, 1]")
    
    # 加上外框線
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"🖼️  正規化視覺化已儲存: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"❌ 寫入錯誤: {e}")
    finally:
        plt.close(fig)

def main():
    print("--- 開始執行 ML-ready 資料視覺化 (正規化檢視模式) ---")
    config = load_config()
    if not config: return

    path_cfg = config.get('path_settings', {})
    ml_dir = path_cfg.get('ml_ready_output_directory')
    viz_dir = path_cfg.get('visualization_output_directory')

    if not ml_dir or not viz_dir:
        print("❌ 路徑設定錯誤")
        return
        
    os.makedirs(viz_dir, exist_ok=True)
    input_files = glob.glob(os.path.join(ml_dir, 'formatted_*.json'))

    print(f"🔍 處理 {len(input_files)} 個檔案...")
    
    for input_file in input_files:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            base_name = os.path.basename(input_file).replace('.json', '')
            output_image_path = os.path.join(viz_dir, f"{base_name}_normalized_vis.png")
            plot_formatted_layout(content, output_image_path)
            
        except Exception as e:
            print(f"處理失敗 {os.path.basename(input_file)}: {e}")

    print("✨ 完成！ ✨")

if __name__ == "__main__":
    main()