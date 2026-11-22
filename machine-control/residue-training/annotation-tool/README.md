# 半自動遮罩標註工具

陳和欽開發這套工具，為印刷前的殘膠檢測製作像素遮罩。它也用於標註另一項分割任務中的大塊溢膠（開發時稱為「耳朵」）。[殘膠模型如何讀取遮罩](../README.md)

魔術棒會選取點選位置附近、顏色相近且連通的像素。遇到輪廓不規則的區域，可切換筆刷或直接擴張、縮減遮罩。輸出的黑白 PNG 保留原圖尺寸，也能重新載入繼續標註。[工具原始碼](annotate.py)

## 試用

使用有桌面環境的 Python。下列 PowerShell 命令會將自製的示意圖複製到工具的輸入目錄。想標註自己的資料時，用同一目錄放入 PNG 工作副本。工具的刪除功能會同時移除輸入影像與對應的遮罩。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force dataset/raw_images
Copy-Item examples/sample.png dataset/raw_images/
.\.venv\Scripts\python.exe annotate.py
```

| 操作 | 用途 |
| --- | --- |
| 滑鼠左鍵點選或拖曳 | 加入選區。試用時可以先點選示意圖上的淺色斑塊。 |
| Tab、+、- | 切換魔術棒／筆刷，或調整選取半徑。 |
| ]、[ | 擴張或縮減遮罩。 |
| Z、R | 復原上一筆選區，或清空目前遮罩。 |
| S、A、D | 儲存目前的遮罩、儲存並切到上一張或下一張。 |
| 長按空白鍵 | 在 Windows 上暫時顯示原圖。 |
| Q 或 Esc | 離開。離開前用 S 儲存當前遮罩。 |

預設遮罩存入 `dataset/masks/`，與輸入檔名相同。程式會記錄上次開啟的影像，重新啟動後可接著標註。若要更改容許色差、4／8 連通性、視窗大小或輸出位置，可把 [設定範本](magicwand_config.example.json) 複製為 `magicwand_config.json` 再編輯。
