# 類比晶片佈局程序化生成器 (Analog Chip Layout Procedural Generator)

這是一個基於 Python 的程序化生成工具，旨在透過一系列可組合、由 `config.yaml` 驅動的規則，階層式地生成複雜的二維佈局，並為其建立連通的網表(Netlist)，模擬類比晶片設計中的元件排列與連接。

此專案流程包含：
1.  **階層式佈局生成**：從一個根元件 (Level 0) 開始，透過 `Level_1` 和 `Level_2` 產生器，逐步將其分割成結構合理的子元件。
2.  **間隙填補 (Gap Filling)**：在已生成的佈局中，自動尋找並填補空白區域，增加佈局的密度與真實性。
3.  **網表生成 (Netlist Generation)**：為所有最終元件產生引腳 (pins)，並透過機率模型建立邊 (edges)，同時確保整個佈局的元件是完全連通的。
4.  **資料格式化 (ML-Ready Formatting)**：將生成的原始 JSON 資料轉換為適合機器學習模型（特別是擴散模型）使用的正規化格式。

## 核心概念

1.  **階層式生成 (Hierarchical Generation)**：佈局的生成是分層的。首先建立一個代表整個晶片區域的根元件 (`Level_0`)，然後由 `Level_1` 產生器對其進行初步分割，最後由 `Level_2` 產生器在前一層的基礎上進行更精細、更複雜的上下文感知 (Context-Aware) 佈局操作。

2.  **規則驅動設計 (Rule-Driven Design)**：所有的佈局變化都是透過應用「規則」來實現的。這些規則被模組化（如分割、對齊、留白），使得演算法的擴充與修改變得容易。

3.  **上下文感知 (Context-Aware)**：在 `Level_2` 的生成器中，系統會分析某元件相對於其「同級元件」的位置，以做出更智慧的決策。例如，一個靠上邊界的元件，系統會避免對它進行「向上對齊」。

4.  **網表生成與連通性保證 (Netlist & Connectivity)**：`NetlistGenerator` 模組會為元件機率性地建立連接，並透過圖遍歷演算法來尋找孤立的元件群，然後建立「橋接邊」來確保最終的網表是單一連通分量。

5.  **組態檔驅動 (Configuration-Driven)**：整個生成流程的所有參數，從元件尺寸、分割機率到網表生成規則，全都由 `config.yaml` 統一管理，方便使用者快速調整與實驗。

6.  **端到端資料管線 (End-to-End Data Pipeline)**：專案不僅生成視覺化的佈局，更提供了一個完整的流程，從 `production.ipynb` 產生原始資料，再由 `format_for_ml.py` 將其轉換為機器學習模型可以直接使用的格式。

## 專案結構
```
analog_chip_layout_generation  
├── aclg    # 核心演算法封裝 (Package)  
│   ├── dataclass   # 定義核心資料結構 (Component)  
│   │   ├── component.py  
│   │   └── __init__.py  
│   ├── drop        # 隨機丟棄元件的規則  
│   │   ├── random_drop.py  
│   │   └── __init__.py  
│   ├── post_processing # 後處理步驟 (如：Padding)  
│   │   ├── padding.py  
│   │   └── __init__.py  
│   ├── rules       # 各式佈局生成規則   
│   │   ├── align   # 對齊規則  
│   │   │   └── __init__.py  
│   │   ├── spacing  # 均分/網格分割規則  
│   │   │   └── __init__.py  
│   │   ├── split    # 基礎/比例分割規則  
│   │   │   ├── split_basic.py  
│   │   │   ├── split_hold.py  
│   │   │   ├── split_ratio.py  
│   │   │   └── __init__.py  
│   │   ├── symetric     # 對稱分割規則  
│   │   │   ├── symmetric_1.py  
│   │   │   └── __init__.py  
│   │   └── __init__.py  
│   └── __init__.py  
├── .gitattributes  
├── .gitignore  
├── config.yaml         # 核心設定檔，所有參數都在此定義  
├── production.ipynb    # 主要執行檔案、範例與視覺化展示  
├── format_for_ml.py    # 主要執行檔案 (2): 將原始 JSON 轉換為 ML 格式  
└── README.md  
```

## 模組詳解

### `aclg.dataclass.component`

-   **`Component`**: 使用 `@dataclass` 定義的核心物件。代表一個矩形元件，包含中心座標 `x`, `y`、`width`、`height`、階層 `level` 等屬性。提供了 `get_topleft()`, `get_bottomright()`, 和 `w_h_ratio()` 等輔助方法。

### `aclg.rules`

這是所有佈局生成規則的核心所在。

-   **`split`**:
    -   `split_basic`: 提供最基本的水平 `split_horizontal` 和垂直 `split_vertical` 分割功能。
    -   `split_ratio`: 更強大的分割工具。`split_by_ratio` 可根據一個比例列表，將元件一次性切成多個子元件。`split_by_ratio_grid` 則可以同時根據水平和垂直的比例列表，直接生成二維網格佈局。
    -   `split_hold`: 一個特殊的「無操作」規則，它會直接回傳原始元件，用於在某些條件下停止對該元件的進一步分割。

-   **`align`**:
    -   `align_components`: 此函式可以對一組已經存在的元件進行縮放和對齊。它支援靠上、下、左、右以及水平/垂直置中等多種對齊模式 (`AlignmentMode`)。

-   **`spacing`**:
    -   `spacing_grid`: 將一個元件平均分割成 `rows` x `cols` 的網格。
    -   `spacing_horizontal` 和 `spacing_vertical`: `spacing_grid` 的特例，分別用於生成單行或單列的均分佈局。

-   **`symetric`**:
    -   `split_symmetric_1_horizontal` 和 `split_symmetric_1_vertical`: 將元件從正中央進行對稱分割（切割比例為 0.5）。

### `aclg.post_processing.padding`

-   `add_padding`: 對元件應用邊距（Padding），使其在保持中心點不變的情況下，按指定數值縮小尺寸。

### `production.ipynb` - 產生器與主流程

這是主要的執行腳本，定義了三層產生器並串連整個流程。

-   **`ComponentPlotter`**: 一個視覺化工具類別，使用 `matplotlib` 將 `Component` 的階層結構遞迴地繪製出來，並用不同顏色區分層級。

-   **`Level_0`**: 根元件產生器。功能很簡單，就是在 `(0,0)` 位置產生一個指定尺寸範圍內的隨機大小的矩形，作為所有佈局的基礎。

-   **`Level_1`**: 初階分割器。它接收上一層的元件，並根據一系列規則進行分割或對齊。
    -   它會優先嘗試符合長寬比限制 (`w_h_ratio_bound`) 的分割方式。
    -   它有一個機率 (`split_only_probability`) 決定是執行「純分割」還是「分割後對齊」。
    -   當子元件數量超過閾值 (`force_align_threshold`) 時，會強制執行對齊操作，以創造更複雜的結構。

-   **`Level_2`**: 進階的**上下文感知**產生器。這是此專案最核心的智慧所在。
    -   **單次對齊原則**: 在處理一批同級元件時，它會先找出尺寸較大的候選者，並從中隨機選擇**最多一個**元件進行進階的 `_apply_advanced_align` 操作，其餘元件則執行常規的網格分割。
    -   **智慧對齊 (`_apply_advanced_align`)**:
        -   在執行對齊前，它會先計算所有同級元件的整體邊界框 (`siblings_bbox`)。
        -   接著，它會判斷目標元件是否緊靠邊界框的上下左右。
        -   最後，它會過濾掉不合理的對齊選項（如：一個已經在最右側的元件不應該再向右對齊），並從合理的選項中隨機挑選一個執行。
    -   **動態策略 (`_get_dynamic_policy`)**: 對於執行常規網格分割的元件，它會根據元件相對於根元件的面積大小，動態調整網格分割的密度。
-   **`GapFiller`**: (可選) 尋找並填補佈局中的空白區域。
-   **`NetlistGenerator`**: 為所有最終元件產生引腳與連線。
-   **輸出**: 將每一組生成的佈局儲存為一張 PNG 圖片 (`raw_layouts/images`) 和一個詳細的 JSON 檔案 (`raw_layouts/json_data`)。JSON 中會記錄該次生成所使用的 `seed`，以供重現。

-   **`format_for_ml.py`**: 資料生成的第二步。此腳本會讀取 `raw_layouts/json_data` 中的原始 JSON 檔案，並將其轉換為機器學習模型所需的格式：
    -   **正規化**: 將元件的尺寸和中心座標正規化。尺寸被正規化到 `[0, 1]`，中心點座標被正規化到 `[-1, 1]`。
    -   **格式轉換**: 輸出包含 `node` (正規化尺寸), `target` (正規化座標), `edges` (引腳的相對偏移量) 等欄位的 JSON 檔案，儲存於 `dataset_ml_ready/`。


## 如何使用

您可以直接在 `production.ipynb` 中執行主要的程式碼儲存格來產生並檢視結果。主要流程如下：

1.  **安裝相依套件**:
    確保您的 Python 環境中已安裝 `matplotlib` 和 `pyyaml`。
    ```bash
    pip install matplotlib pyyaml
    ```

2.  **調整設定**:
    打開 `config.yaml` 檔案。根據您的需求，修改其中的參數。。

3.  **產生原始佈局與網表**:
    執行 `production.ipynb`。

4.  **轉換為機器學習格式**:
    執行 `format_for_ml.py` 腳本。它會自動找到上一步產生的所有 JSON 檔案並進行轉換。
    ```
    python format_for_ml.py
    ```


## 輸出

1. `raw_layouts/images/raw_layouts_*.png`:  
原始佈局的視覺化圖片，包含不同層級的元件、ID、以及生成的網表（引腳與連線）。

2. `raw_layouts/json_data/raw_layouts_*.json`:  
詳細的原始資料。包含完整的元件階層樹、`GapFiller` 新增的元件、每個元件的絕對座標與尺寸，以及網表的邊列表和使用的隨機種子。

3. `dataset_ml_ready/formatted_*.json`:  
最終提供給機器學習模型的資料。所有幾何資訊都經過正規化，並且格式符合常見的圖神經網路或擴散模型輸入要求。