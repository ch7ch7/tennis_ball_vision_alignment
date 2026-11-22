# 網球殘膠檢測模型訓練

網球第一次微調後，機台要判斷朝上的球面是否有影響商標壓印的殘膠。這個 U-Net 讀取彩色球面影像，逐像素估計殘膠位置。Python 負責訓練與 ONNX 匯出，C# 在機台上執行推論並決定是否翻面。[C# 機台控制說明](../README.md) · [殘膠辨識實機測試](../../docs/evaluation.md)

## 一張球面照片如何變成訓練資料？

在原圖上圈出殘膠並修整邊界，可得到同尺寸的黑白 PNG。白色像素代表殘膠，黑色代表背景；無殘膠的球面對應一張全黑遮罩。研究中使用的 427 張影像由陳和欽完成標註。[半自動遮罩標註工具](annotation-tool/README.md)

資料載入程式依檔名配對原圖與遮罩，調整亮度通道的局部對比度，並將影像和遮罩縮放至 256 × 256。訓練時依有膠、無膠分組切分訓練與驗證資料，使用加權二元交叉熵與 Dice loss 訓練分割模型。驗證損失最低的權重用於匯出 ONNX。[資料前處理](dataset.py) · [U-Net 結構](model.py) · [訓練流程](train.py) · [ONNX 匯出](export_onnx.py)

## 使用自己的影像訓練

從本目錄執行，將原圖放在 `dataset/train_images/`，對應遮罩放在 `dataset/masks/`，例如 `train_images/ball01.png` 搭配 `masks/ball01.png`。使用下層標註工具時，將已標註的原圖和輸出遮罩分別複製到這兩個目錄。請準備有膠和無膠兩類樣本，每類至少兩組，讓訓練與驗證各有樣本。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe export_onnx.py
```

訓練輪數、影像尺寸與批次大小可在 [config.py](config.py) 調整。訓練會寫入 TensorBoard 紀錄和最佳權重。匯出程式預設選取最近更新的最佳權重，在本目錄產生 `model.onnx`。
