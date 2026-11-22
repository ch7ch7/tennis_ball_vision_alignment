# magic_wand_labeler.py
import cv2
import numpy as np
from pathlib import Path
import json
import os
import time

# ===== Config: 讀取 magicwand_config.json，找不到則用預設 =====
CONFIG_PATH = Path("magicwand_config.json")
DEFAULTS = {
    "IMG_DIR": "dataset/raw_images",
    "SAVE_DIR": "dataset/masks",
    "TOLERANCE": 20,
    "SELECT_RADIUS": 100,
    "CONNECTIVITY": 8,                 # 8 或 4
    "RESIZE_SHAPE": [600, 600],      # [W, H]
    "RADIUS_STEP": 5,
    "STATE_FILE": "dataset/masks/_magicwand_state.json",
    "MORPH_KERNEL_SIZE": 3            # 膨脹/侵蝕的內核大小 (預設 3x3)
}

def load_config():
    cfg = DEFAULTS.copy()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user = json.load(f)
            if isinstance(user, dict):
                cfg.update(user)
        except Exception as e:
            print(f"[WARN] Failed to read {CONFIG_PATH}: {e}\nUsing defaults.")
    # normalize
    if isinstance(cfg.get("RESIZE_SHAPE"), (list, tuple)) and len(cfg["RESIZE_SHAPE"]) == 2:
        cfg["RESIZE_SHAPE"] = (int(cfg["RESIZE_SHAPE"][0]), int(cfg["RESIZE_SHAPE"][1]))
    else:
        cfg["RESIZE_SHAPE"] = (1200, 1200)
    cfg["CONNECTIVITY"] = int(cfg.get("CONNECTIVITY", 8))
    cfg["TOLERANCE"] = int(cfg.get("TOLERANCE", 20))
    cfg["SELECT_RADIUS"] = int(cfg.get("SELECT_RADIUS", 100))
    cfg["RADIUS_STEP"] = int(cfg.get("RADIUS_STEP", 5))
    cfg["MORPH_KERNEL_SIZE"] = int(cfg.get("MORPH_KERNEL_SIZE", 3)) # 新增
    cfg["IMG_DIR"] = str(cfg.get("IMG_DIR", "dataset/raw_images"))
    cfg["SAVE_DIR"] = str(cfg.get("SAVE_DIR", "dataset/masks"))
    cfg["STATE_FILE"] = str(cfg.get("STATE_FILE", "dataset/masks/_magicwand_state.json"))
    return cfg

CFG = load_config()
IMG_DIR         = CFG["IMG_DIR"]
SAVE_DIR        = CFG["SAVE_DIR"]
TOLERANCE       = CFG["TOLERANCE"]
SELECT_RADIUS   = CFG["SELECT_RADIUS"]
CONNECTIVITY    = CFG["CONNECTIVITY"]
RESIZE_SHAPE    = CFG["RESIZE_SHAPE"]
RADIUS_STEP     = CFG["RADIUS_STEP"]
STATE_FILE      = Path(CFG["STATE_FILE"])
MORPH_KSIZE     = CFG["MORPH_KERNEL_SIZE"]

# ===== 全域定義：形態學處理內核 =====
_morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KSIZE, MORPH_KSIZE))

# ===== 空白鍵長按偵測（Windows；其他平台回傳 False） =====
try:
    import ctypes
    _user32 = ctypes.windll.user32
    def _is_space_down() -> bool:
        return (_user32.GetAsyncKeyState(0x20) & 0x8000) != 0
except Exception:
    def _is_space_down() -> bool:
        return False

# ===== 載入資料夾影像 =====
image_files = sorted(Path(IMG_DIR).glob("*.png"))
if not image_files:
    raise SystemExit(f"No .png files found in {IMG_DIR}")

# ----- 狀態保存/載入：記住上次影像 -----
def load_last_index(files):
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        last_stem = data.get("last_stem", "")
        if last_stem:
            for i, p in enumerate(files):
                if p.stem == last_stem:
                    return i
    except Exception:
        pass
    return 0

def write_last_state(stem):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_stem": stem}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] cannot write state: {e}")

# 以「上次關閉的影像」為起點
index = load_last_index(image_files)

# ===== 全域工作狀態 =====
img = None
h, w = None, None
acc_mask = None                
scale_x, scale_y = None, None  
compare_mode = False           
undo_stack = []
hover_xy = None                

# 🔹 新增：全圖純黑像素的布林遮罩，加速運算
is_pure_black = None

use_brush_mode = False         # False=魔術棒(連通), True=純筆刷(無連通, 全塗)

# 互動繪製
is_drawing = False             
stroke_mask = None             
last_seed_xy = None            
DRAG_STEP_PX = 0               

# 刪除鍵雙擊偵測
DEL_DOUBLE_MS = 600
last_del_time = 0.0

# ===== 功能函式 =====
def save_current_mask():
    out_dir = Path(SAVE_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (image_files[index].stem + ".png")
    try:
        saved = cv2.imwrite(str(out_path), acc_mask)
    except cv2.error as err:
        print(f"[Save] {out_path}: {err}")
        return False
    if not saved:
        print(f"[Save] Cannot write mask: {out_path}")
        return False
    print(f"[AutoSave] {out_path}")
    return True

def load_existing_mask(stem: str, shape_hw):
    h0, w0 = shape_hw
    m_path = Path(SAVE_DIR) / (stem + ".png")
    if not m_path.exists():
        return np.zeros((h0, w0), np.uint8)
    m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((h0, w0), np.uint8)
    if m.shape[:2] != (h0, w0):
        m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8) * 255
    print(f"[LoadMask] {m_path}")
    return m

def current_display_mask():
    if is_drawing and stroke_mask is not None:
        return np.maximum(acc_mask, stroke_mask)
    return acc_mask

def load_image(idx):
    global img, h, w, acc_mask, scale_x, scale_y, hover_xy, index, is_pure_black
    index = idx
    path = str(image_files[idx])
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")

    h, w = img.shape[:2]

    # 🔹 每次載入新圖時，先算好整張圖的純黑遮罩，後續操作直接套用
    is_pure_black = np.all(img == [0, 0, 0], axis=-1)

    acc_mask = load_existing_mask(Path(path).stem, (h, w))
    # 🔹 預防機制：強制清除舊有 mask 中可能標到的純黑區
    if is_pure_black is not None:
        acc_mask[is_pure_black] = 0

    undo_stack.clear()
    hover_xy = None

    scale_x = w / float(RESIZE_SHAPE[0])
    scale_y = h / float(RESIZE_SHAPE[1])

    write_last_state(Path(path).stem)
    refresh_display()
    print(f"Loaded: {path} ({idx+1}/{len(image_files)})")

def refresh_display():
    if img is None: return

    if compare_mode:
        base = img.copy()
        suffix = " (ORIGINAL VIEW: hold SPACE)"
    else:
        overlay = img.copy()
        disp_mask = current_display_mask()
        overlay[disp_mask == 255] = (0, 0, 255)
        base = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        suffix = ""

    if hover_xy is not None:
        cx, cy = hover_xy
        cv2.circle(base, (int(cx), int(cy)), int(SELECT_RADIUS), (0, 255, 255), 1, cv2.LINE_AA)

    resized = cv2.resize(base, RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("MagicWand", resized)

    mode_str = "「純筆刷全塗」" if use_brush_mode else f"「魔術棒連通({CONNECTIVITY})」"
    
    title = (
        f"MagicWand [{index+1}/{len(image_files)}]{suffix} | "
        f"模式(TAB):{mode_str} | +/-:半徑({SELECT_RADIUS}px) | "
        f"]:擴張, [:縮減 | A/D:前後一張 | Z:復原 | S:存檔"
    )
    try:
        cv2.setWindowTitle("MagicWand", title)
    except Exception:
        pass

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def adjust_radius(delta):
    global SELECT_RADIUS
    SELECT_RADIUS = int(clamp(SELECT_RADIUS + delta, 1, max(h, w)))
    refresh_display()
    print(f"SELECT_RADIUS = {SELECT_RADIUS}px")

def pick_region_local_bgr_connected(seed_x, seed_y, tol, radius, connectivity=8):
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    inner = ff_mask[1:-1, 1:-1]
    inner[:] = 1
    cv2.circle(inner, (seed_x, seed_y), int(radius), color=0, thickness=-1)

    flags = (connectivity
             | cv2.FLOODFILL_FIXED_RANGE
             | cv2.FLOODFILL_MASK_ONLY
             | (255 << 8))

    tmp = img.copy()
    cv2.floodFill(tmp, ff_mask, (seed_x, seed_y), (0, 0, 255), (tol,)*3, (tol,)*3, flags)
    region = (ff_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
    return region

def apply_morphology(op_type):
    global acc_mask
    if acc_mask is None: return

    if op_type == "dilate":
        new_acc = cv2.dilate(acc_mask, _morph_kernel, iterations=1)
        print("[Morphology] Dilated (expanded) mask +1px")
    elif op_type == "erode":
        new_acc = cv2.erode(acc_mask, _morph_kernel, iterations=1)
        print("[Morphology] Eroded (shrunk) mask -1px")
    else:
        return

    # 🔹 無論是膨脹還是侵蝕，都強制清空純黑區域
    if is_pure_black is not None:
        new_acc[is_pure_black] = 0

    undo_stack.append(acc_mask)
    acc_mask = new_acc

    refresh_display()

def begin_stroke(ox, oy):
    global is_drawing, stroke_mask, last_seed_xy
    is_drawing = True
    stroke_mask = np.zeros((h, w), np.uint8)
    last_seed_xy = None
    add_seed_to_stroke(ox, oy)

def add_seed_to_stroke(ox, oy):
    global stroke_mask, last_seed_xy
    if last_seed_xy is not None:
        dx = ox - last_seed_xy[0]
        dy = oy - last_seed_xy[1]
        if (dx*dx + dy*dy) < (DRAG_STEP_PX * DRAG_STEP_PX):
            return

    if use_brush_mode:
        region = np.zeros((h, w), np.uint8)
        cv2.circle(region, (ox, oy), SELECT_RADIUS, 255, -1)
    else:
        region = pick_region_local_bgr_connected(ox, oy, TOLERANCE, SELECT_RADIUS, CONNECTIVITY)

    # 🔹 無論是魔術棒還是筆刷塗抹，強制清空純黑區域
    if is_pure_black is not None:
        region[is_pure_black] = 0

    stroke_mask = np.maximum(stroke_mask, region)
    last_seed_xy = (ox, oy)
    refresh_display()

def end_stroke():
    global is_drawing, stroke_mask, acc_mask
    if not is_drawing:
        return
    if stroke_mask is not None and stroke_mask.any():
        undo_stack.append(acc_mask)
        acc_mask = np.maximum(acc_mask, stroke_mask)
    is_drawing = False
    stroke_mask = None
    refresh_display()

def cancel_stroke():
    global is_drawing, stroke_mask
    is_drawing = False
    stroke_mask = None
    refresh_display()

def on_mouse(event, x, y, flags, param):
    global hover_xy
    ox = int(round(x * scale_x))
    oy = int(round(y * scale_y))
    ox = clamp(ox, 0, w - 1); oy = clamp(oy, 0, h - 1)

    if event == cv2.EVENT_MOUSEMOVE:
        hover_xy = (ox, oy)
        if not compare_mode and (flags & cv2.EVENT_FLAG_LBUTTON):
            if not is_drawing:
                begin_stroke(ox, oy)
            else:
                add_seed_to_stroke(ox, oy)
        else:
            refresh_display()
        return

    if compare_mode:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        begin_stroke(ox, oy)

    elif event == cv2.EVENT_LBUTTONUP:
        end_stroke()

def delete_current_image_and_mask():
    global index

    if not image_files:
        print("[Delete] No images.")
        return

    cur_img = image_files[index]
    cur_mask = Path(SAVE_DIR) / (cur_img.stem + ".png")

    try:
        os.remove(str(cur_img))
        print(f"[Delete] image removed: {cur_img}")
    except FileNotFoundError:
        print(f"[Delete] image not found: {cur_img}")
    except Exception as e:
        print(f"[Delete] image remove failed: {e}")
        return

    if cur_mask.exists():
        try:
            os.remove(str(cur_mask))
            print(f"[Delete] mask removed:  {cur_mask}")
        except Exception as e:
            print(f"[Delete] mask remove failed: {e}")
    else:
        print(f"[Delete] mask not found:  {cur_mask}")

    del image_files[index]
    if not image_files:
        print("[Delete] All images removed. Exit.")
        cv2.destroyAllWindows()
        raise SystemExit(0)

    if index >= len(image_files):
        index = len(image_files) - 1

    load_image(index)
    try:
        cv2.resizeWindow("MagicWand", RESIZE_SHAPE[0], RESIZE_SHAPE[1])
    except Exception:
        pass

# ===== 啟動 =====
cv2.namedWindow("MagicWand", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("MagicWand", on_mouse)
load_image(index)
cv2.resizeWindow("MagicWand", RESIZE_SHAPE[0], RESIZE_SHAPE[1])

print("-" * 60)
print("操作指南:")
print("  滑鼠左鍵拖曳      : 連續標註")
print("  TAB 鍵        : 切換「魔術棒連通」/「純筆刷全塗」模式 (重要!)")
print("   ]  鍵 (右中括號) : 一鍵擴張 (膨脹 mask +1px) - 解決邊緣未選到")
print("   [  鍵 (左中括號) : 一鍵縮減 (侵蝕 mask -1px) - 修正不小心塗出去")
print("  Z 鍵              : 上一步 (撤銷)")
print("  R 鍵              : 清除所有 mask")
print("  S / A / D 鍵      : 存檔 / 上一張 / 下一張")
print("  +/- 鍵            : 調整筆刷/選取半徑")
print("  C / 4 / 8 鍵      : 調整連通性 (魔術棒模式下)")
print("  DEL 連按兩次       : 刪除當前影像+mask")
print("  Q / ESC 鍵        : 離開程序")
print("  長按空白鍵         : 顯示原圖 (比對用)")
print("-" * 60)

prev_space = False
last_del_time = 0.0
while True:
    key = cv2.waitKeyEx(20)

    if key in (27, ord('q'), ord('Q')):
        break
    elif key in (ord('r'), ord('R')):
        cancel_stroke()
        undo_stack.append(acc_mask)
        acc_mask = np.zeros((h, w), np.uint8)
        refresh_display()
        print("Cleared mask to all zeros.")
    elif key in (ord('s'), ord('S')):
        end_stroke()
        save_current_mask()
    elif key in (ord('z'), ord('Z')):
        cancel_stroke()
        if undo_stack:
            acc_mask = undo_stack.pop()
            refresh_display()
            print("Undo last region.")
        else:
            print("No region to undo.")
    elif key in (ord('a'), ord('A')):   
        end_stroke()
        if not save_current_mask():
            continue
        load_image((index - 1) % len(image_files))
        try:
            cv2.resizeWindow("MagicWand", RESIZE_SHAPE[0], RESIZE_SHAPE[1])
        except Exception:
            pass
    elif key in (ord('d'), ord('D')):   
        end_stroke()
        if not save_current_mask():
            continue
        load_image((index + 1) % len(image_files))
        try:
            cv2.resizeWindow("MagicWand", RESIZE_SHAPE[0], RESIZE_SHAPE[1])
        except Exception:
            pass
    elif key in (ord('+'), ord('=')):
        adjust_radius(+RADIUS_STEP)
    elif key in (ord('-'), ord('_')):
        adjust_radius(-RADIUS_STEP)

    elif key in (ord('c'), ord('C')):
        CONNECTIVITY = 4 if CONNECTIVITY == 8 else 8
        print(f"CONNECTIVITY -> {CONNECTIVITY}")
        refresh_display()
    elif key == ord('4'):
        CONNECTIVITY = 4
        print(f"CONNECTIVITY -> {CONNECTIVITY}")
        refresh_display()
    elif key == ord('8'):
        CONNECTIVITY = 8
        print(f"CONNECTIVITY -> {CONNECTIVITY}")
        refresh_display()

    elif key == 9:  
        use_brush_mode = not use_brush_mode
        state = "BRUSH (No connected)" if use_brush_mode else "MAGIC WAND (Connected)"
        print(f"Mode changed to: {state}")
        refresh_display()

    elif key == 93: 
        cancel_stroke()
        end_stroke() 
        apply_morphology("dilate")
    elif key == 91: 
        cancel_stroke()
        end_stroke() 
        apply_morphology("erode")

    elif key in (127, 3014656):
        now = time.time()
        if (now - last_del_time) <= (DEL_DOUBLE_MS / 1000.0):
            cancel_stroke()
            delete_current_image_and_mask()
            last_del_time = 0.0
        else:
            print("Press DEL again within 0.6s to DELETE current image & mask.")
            last_del_time = now

    space_now = _is_space_down()
    if space_now != prev_space:
        compare_mode = space_now
        refresh_display()
        prev_space = space_now

cv2.destroyAllWindows()
