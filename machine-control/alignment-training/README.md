# 網球姿態模型訓練

這裡收錄陳和欽在碩士研究中開發的 PyTorch 程式。模型讀取成對的清晰與模糊縫線圖，預測兩個旋轉角度和一個滾動量。兩個角度各用正弦與餘弦表示，加上正規化的滾動量，總共五個輸出值。

| 檔案 | 實作內容 |
| --- | --- |
| [model.py](model.py) | 用兩個單通道 EfficientNet-B3 分支擷取特徵，合併後預測初始旋轉、後續旋轉與滾動量。 |
| [dataset.py](dataset.py) | 配對兩張影像，將 IA、R、A 編碼成模型的五個訓練目標。 |
| [train.py](train.py) | 從初始角度與滾動量的聯合分布計算加權抽樣，訓練雙分支模型，並記錄驗證損失及角度誤差。 |
| [generate_labels.py](generate_labels.py) 和 [generate_blur_images.py](generate_blur_images.py) | 從檔名讀取動作參數，並以高斯模糊和縮放產生第二路輸入。 |
| [export_to_onnx.py](export_to_onnx.py) | 將訓練所得的 PyTorch 權重匯出為雙輸入的 ONNX 模型。 |

## 準備資料並執行

使用 Python 3.13，從此目錄執行以下命令。訓練需要自行準備已二值化的縫線圖，放入 `data/images_processed/`。檔名格式為 `<任意前綴>_IA_<初始旋轉角度>_R_<滾動量>_A_<後續旋轉角度>.png`，例如 `sample_Top_IA_15_R_2800_A_30.png`。IA、R、A 填整數，訓練資料需涵蓋多種滾動量。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --cpu --epochs 1 --batch-size 2 --workers 0
```

這組參數適合在小資料集上確認流程。分割後的訓練集至少要有兩筆影像，且須包含至少兩種 R 值。正式訓練時，可調整 [config.py](config.py) 或以 `python train.py --help` 查看參數。要使用 NVIDIA GPU，可安裝對應 CUDA 的 PyTorch 組合，並省略 `--cpu`。[PyTorch 安裝說明](https://pytorch.org/get-started/locally/)

訓練時會建立模糊影像、標籤 CSV、TensorBoard 紀錄與最佳權重。可執行 `.\.venv\Scripts\python.exe -m tensorboard.main --logdir logs` 查看訓練曲線。匯出時，程式會讀取 `saved_models/` 中最後修改的最佳權重，寫出 `model.onnx`。

```powershell
.\.venv\Scripts\python.exe export_to_onnx.py
```

本目錄提供訓練與匯出原始碼。重現論文結果需要當時的資料與實驗條件。[對位原理與部署](../../docs/engineering.md) · [實機測試](../../docs/evaluation.md)
