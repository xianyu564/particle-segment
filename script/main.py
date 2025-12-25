import os, re, json, glob, datetime
import numpy as np
import cv2
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import label as sk_label

# -----------------------------
# JSON helper
# -----------------------------
def to_jsonable(obj):
    import numpy as np
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj

# -----------------------------
# Overlay masking (time & scale zones)
# 根据你图的布局：时间左上；比例尺右下
# -----------------------------
def mask_overlays(gray, time_roi=(0, 0, 300, 160), scale_roi=(650, 820, 380, 220)):
    work = gray.copy()
    med = int(np.median(work))
    x, y, w, h = time_roi
    work[y:y+h, x:x+w] = med
    x, y, w, h = scale_roi
    work[y:y+h, x:x+w] = med
    return work

# -----------------------------
# Read time mark (优先：文件名；其次：OCR)
# 如果你文件名是 FM_t23.jpg 这种，建议用它做“帧序号/时间”
# -----------------------------
def read_time_seconds_from_filename(path, fps_or_dt_seconds=5):
    # 例如 FM_t23.jpg -> 23 * 5 sec = 115 sec（你可改 dt）
    m = re.search(r"_t(\d+)", os.path.basename(path))
    if not m:
        return None
    idx = int(m.group(1))
    return idx * fps_or_dt_seconds

def read_time_seconds_ocr(gray, roi=(0, 0, 300, 160)):
    # 如果你已经装好 tesseract 才能用；否则就别调用它
    import pytesseract
    x, y, w, h = roi
    patch = gray[y:y+h, x:x+w].copy()
    patch = cv2.resize(patch, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    _, bw = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    txt = pytesseract.image_to_string(bw, config="--psm 6").strip().lower()
    # parse "110 sec"
    m = re.search(r"(\d+)\s*sec", txt)
    if m:
        return int(m.group(1)), txt
    return None, txt

# -----------------------------
# Scale bar length detection (只读白色横条像素长度)
# 默认右下角；不依赖 OCR 文本 “5 um”
# -----------------------------
def read_scale_bar(gray, roi=(650, 820, 380, 220), real_um=5.0):
    x, y, w, h = roi
    patch = gray[y:y+h, x:x+w].copy()

    # 找“白色比例尺横条”：阈值取高亮
    thr = np.percentile(patch, 99)
    bw = (patch >= thr).astype(np.uint8) * 255

    # 只保留比较粗的横向结构：形态学开运算（横向核）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3))
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    # 找最长的水平连通段
    cnts, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # 选“最宽”的轮廓作为横条
    best = max(cnts, key=lambda c: cv2.boundingRect(c)[2])
    bx, by, bwid, bhei = cv2.boundingRect(best)
    bar_px = int(bwid)

    if bar_px <= 0:
        return None

    um_per_px = float(real_um) / float(bar_px)
    return {
        "real_um": float(real_um),
        "bar_px": int(bar_px),
        "um_per_px": float(um_per_px),
        "roi": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    }

# -----------------------------
# Segmentation: Otsu + morphology + distance + watershed
# 目标：一颗粒子 -> 一个 label
# -----------------------------
from skimage.feature import peak_local_max

def _norm_u8(img):
    """Normalize float/uint to uint8 for saving."""
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-9:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn)
    return (arr * 255).astype(np.uint8)

def _norm_u8_force(img):
    arr = img.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-9:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn)
    return (arr * 255).astype(np.uint8)


def save_dbg(debug_dir, stem, name, img, cmap=None):
    """Save grayscale or color debug image."""
    if debug_dir is None:
        return
    os.makedirs(debug_dir, exist_ok=True)
    out = os.path.join(debug_dir, f"{stem}__{name}.png")
    if img is None:
        return
    if img.ndim == 2:
        cv2.imwrite(out, _norm_u8(img))
    else:
        cv2.imwrite(out, img)


def _remove_small_cc(bw_bool, min_area=200):
    """
    Remove small connected components from a binary mask.
    bw_bool: bool array
    """
    if min_area is None or min_area <= 0:
        return bw_bool

    bw_u8 = (bw_bool.astype(np.uint8) * 255)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw_u8, connectivity=8)

    keep = np.zeros(n, dtype=bool)
    keep[0] = False  # background
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            keep[i] = True

    return keep[labels]

def _remove_small_cc_u8(bw_u8, min_area=200):
    """
    Remove small connected components from a binary image.
    bw_u8: uint8 (0/255)
    return: uint8 (0/255)
    """
    if min_area is None or min_area <= 0:
        return bw_u8

    bw_u8 = (bw_u8 > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw_u8, connectivity=8)

    out = np.zeros_like(bw_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            out[labels == i] = 255
    return out


def _keep_large_cc_u8(bw_u8, min_area=3000):
    """Only keep large connected components (uint8 0/255)."""
    bw_u8 = (bw_u8 > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw_u8, connectivity=8)
    out = np.zeros_like(bw_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            out[labels == i] = 255
    return out


def _triangle_threshold_value(u8):
    """Return triangle threshold value (int). u8 must be uint8."""
    # OpenCV triangle uses THRESH_TRIANGLE; it returns threshold used
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return int(t)


def segment_particles(
    gray,
    time_roi=(0, 0, 260, 140),
    scale_roi=(700, 820, 324, 204),

    # ---- pre-process ----
    use_clahe=True,
    clahe_clip=0.1,          # 太大容易把背景抬亮 -> “压不住白”
    clahe_grid=(8, 8),
    median_ksize=3,          # 不是 gaussian；对盐椒点更有效。不要太大(3/5)

    # ---- threshold ----
    thr_method="otsu",   # "triangle" or "otsu"
    thr_offset=10,            # >0 更敏感(阈值更低); <0 更保守(阈值更高，用来“压白”)
    auto_clip_white=True,    # 自动“压白”
    fg_ratio_max=0.18,       # 前景占比太高就自动抬阈值
    fg_ratio_min=0.003,      # 前景占比太低就自动降阈值(保细边)
    auto_step=2,             # 每次调阈值的步长
    auto_max_iter=40,        # 最多迭代次数

    # ---- binary cleanup ----
    min_cc_area=800,         # 关键：压小点爆炸（你图上 300~800 都可试）
    close_kernel=1,          # 默认关掉 close（你说 close 不合理）
    close_iter=1,
    open_kernel=3,           # 很轻的 open；主要用于边缘毛刺
    open_iter=1,
    fill_holes=True,

    # ---- watershed ----
    ws_min_distance=60,
    ws_threshold_rel=0.4,

    debug_dir=None,
    stem="img",
    raw_bgr=None,   # <--- 新增：传入原始彩色图
):
    save_dbg(debug_dir, stem, "00_raw", gray)

    # 1) mask overlays
    work = mask_overlays(gray, time_roi=time_roi, scale_roi=scale_roi)
    save_dbg(debug_dir, stem, "01_mask_overlays", work)

    # 2) median denoise (NOT gaussian)
    if median_ksize and median_ksize >= 3 and (median_ksize % 2 == 1):
        work = cv2.medianBlur(work, int(median_ksize))
    save_dbg(debug_dir, stem, f"02_median_k{median_ksize}", work)

    # 3) CLAHE (optional)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid))
        work = clahe.apply(work)
    save_dbg(debug_dir, stem, f"03_clahe_clip{clahe_clip}", work)

    # 4) threshold base
    if thr_method.lower() == "otsu":
        t_base, _ = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t_base = int(t_base)
    else:
        t_base = _triangle_threshold_value(work)

    # apply offset (IMPORTANT: offset>0 makes threshold LOWER -> more white)
    t = int(np.clip(t_base - int(thr_offset), 0, 255))

    # 4.1 auto “clip white” by monitoring fg ratio
    #     if too white -> increase threshold (t += step)
    #     if too black -> decrease threshold (t -= step)
    if auto_clip_white:
        for _ in range(int(auto_max_iter)):
            _, bw_try = cv2.threshold(work, int(t), 255, cv2.THRESH_BINARY)
            fg_ratio = float((bw_try > 0).mean())
            if fg_ratio > float(fg_ratio_max):
                t = min(255, t + int(auto_step))   # raise threshold to suppress white
                continue
            if fg_ratio < float(fg_ratio_min):
                t = max(0, t - int(auto_step))     # lower threshold to keep faint edges
                continue
            break

    _, bw = cv2.threshold(work, int(t), 255, cv2.THRESH_BINARY)
    bw = bw.astype(np.uint8)
    save_dbg(debug_dir, stem, f"04_thr_{thr_method}_t{t_base}_off{thr_offset}_use{t}", bw)

    # 5) remove small CC first (压小点爆炸)
    bw = _remove_small_cc_u8(bw, min_area=min_cc_area)
    save_dbg(debug_dir, stem, f"05_rm_small_cc_min{min_cc_area}", bw)

    # 6) GAP BRIDGING before fill holes: close only on LARGE objects
    #    目的：把“括号形”亮边缘补成闭环，fill holes 才能填满
    bridge_min_area = 3000     # 只对大颗粒补缝，避免噪声越补越多
    bridge_kernel = 5          # 关键旋钮：3/5 先试；太大容易把相邻颗粒粘死
    bridge_iter = 2            # 一般 1 就够，必要再到 2

    bw_big = _keep_large_cc_u8(bw, min_area=bridge_min_area)
    save_dbg(debug_dir, stem, f"06a_keep_big_min{bridge_min_area}", bw_big)

    # 用 CROSS 核补“细缺口”更克制，较不容易横向膨胀
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (bridge_kernel, bridge_kernel))
    bw_bridge = cv2.morphologyEx(bw_big, cv2.MORPH_CLOSE, k, iterations=int(bridge_iter))
    save_dbg(debug_dir, stem, f"06b_bridge_close_cross_k{bridge_kernel}_it{bridge_iter}", bw_bridge)

    # closing 可能带来少量碎屑，再兜底去一次小连通域
    bw_bridge = _remove_small_cc_u8(bw_bridge, min_area=min_cc_area)
    save_dbg(debug_dir, stem, f"06c_bridge_rm_small_min{min_cc_area}", bw_bridge)

    # 7) fill holes（此时边缘更闭合，才能填满内部）
    bw_bool = (bw_bridge > 0)
    if fill_holes:
        bw_bool = ndi.binary_fill_holes(bw_bool)
    bw_fill = (bw_bool.astype(np.uint8) * 255)
    save_dbg(debug_dir, stem, f"07_fill_holes_{int(fill_holes)}", bw_fill)


    # 8) light open
    bw_u8 = bw_fill
    if open_kernel and open_kernel > 1:
        bw_u8 = cv2.morphologyEx(
            bw_fill, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_kernel), int(open_kernel))),
            iterations=int(open_iter),
        )
    bw_bool = bw_u8 > 0
    save_dbg(debug_dir, stem, f"08_open_k{open_kernel}_it{open_iter}", bw_u8)

    # 9) distance map
    dist = ndi.distance_transform_edt(bw_bool)
    save_dbg(debug_dir, stem, "09_distance", dist)

    # 10) markers
    peaks = peak_local_max(
        dist,
        min_distance=int(ws_min_distance),
        threshold_rel=float(ws_threshold_rel),
        labels=bw_bool.astype(np.uint8),
        exclude_border=False,
    )

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i

    mk_vis = cv2.cvtColor(_norm_u8(dist), cv2.COLOR_GRAY2BGR)
    
    def _draw_filled_triangle(img, x, y, size=10, color=(255, 0, 0)):  # BGR: 蓝色
        h, w = img.shape[:2]
        x = int(x); y = int(y)
        s = int(size)
        # 上尖三角形：顶点在上
        pts = np.array([
            [x,   y - s],
            [x - s, y + s],
            [x + s, y + s],
        ], dtype=np.int32)
    
        # 裁剪到图像范围
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    
        cv2.fillConvexPoly(img, pts, color)
    
    for (r, c) in peaks:
        _draw_filled_triangle(mk_vis, c, r, size=10, color=(255, 0, 0))  # 蓝色实心三角
    
    save_dbg(debug_dir, stem, f"10_markers_n{len(peaks)}_md{ws_min_distance}_tr{ws_threshold_rel}", mk_vis)


    if markers.max() == 0:
        markers = sk_label(bw_bool)

    # 11) watershed
    labels_ws = watershed(-dist, markers, mask=bw_bool).astype(np.int32)
    lab_vis = _norm_u8_force(labels_ws)   # 不要先转 uint8
    save_dbg(debug_dir, stem, f"11_watershed_labels_k{labels_ws.max()}", lab_vis)

    print("bw pixels:", int(bw_bool.sum()))
    print("dist max:", float(dist.max()))
    print("labels max:", int(labels_ws.max()))

    # 12) boundary overlay
    edges = cv2.Canny(lab_vis, 50, 150)
    
    # 加粗：膨胀边缘
    edge_thick = 3  # 线宽大概 3~5 之间调
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_thick, edge_thick))
    edges_thick = cv2.dilate(edges, k, iterations=1)
    
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[edges_thick > 0] = (255, 0, 0)  # 蓝色
    save_dbg(debug_dir, stem, "12_edges_on_raw", overlay)

    # 13) boundary overlay on ORIGINAL BGR (raw)
    if raw_bgr is not None:
        overlay_raw = raw_bgr.copy()
        overlay_raw[edges_thick > 0] = (255, 0, 0)  # 蓝色
        save_dbg(debug_dir, stem, "13_edges_on_raw_bgr", overlay_raw)

    return labels_ws, bw_u8, work




# -----------------------------
# a/c extraction: minAreaRect from each label
# -----------------------------
def measure_ac(
    labels_ws,
    um_per_px,
    min_area=0,
    include_border=True,
    margin=0,
):
    """
    Measure each label with minAreaRect.
    a = short axis, c = long axis (in px and um)

    Args:
      labels_ws: int32 label image (0=background)
      um_per_px: float
      min_area: int, filter tiny labels (0 means keep all)
      include_border: if False, drop objects touching border by 'margin'
      margin: int, margin in px used when include_border=False

    Returns:
      list of dicts (one per label)
    """
    H, W = labels_ws.shape[:2]
    particles = []

    labs = np.unique(labels_ws)
    labs = labs[labs != 0]  # drop background

    for lab in labs:
        lab = int(lab)
        mask = (labels_ws == lab).astype(np.uint8)  # 0/1

        area_px = int(np.count_nonzero(mask))
        if area_px <= int(min_area):
            continue

        # find contour from mask
        cnts, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        ctn = max(cnts, key=cv2.contourArea)

        # min area rect
        rect = cv2.minAreaRect(ctn)  # ((cx,cy),(w,h),angle)
        (cx, cy), (rw, rh), ang = rect

        # IMPORTANT: a=short, c=long
        short_px = float(min(rw, rh))
        long_px  = float(max(rw, rh))

        # optional border filtering
        if not include_border:
            box = cv2.boxPoints(rect)  # 4x2 float
            if (box[:, 0].min() < margin or box[:, 0].max() > (W - 1 - margin) or
                box[:, 1].min() < margin or box[:, 1].max() > (H - 1 - margin)):
                continue

        particles.append({
            "label": lab,
            "x_px": float(cx),
            "y_px": float(cy),

            "a_px": short_px,                 # short axis
            "c_px": long_px,                  # long axis
            "a_um": float(short_px * um_per_px),
            "c_um": float(long_px  * um_per_px),

            "area_px": area_px,
            "angle_deg": float(ang),
        })

    # sort by label id for stability
    particles.sort(key=lambda d: d["label"])
    return particles


# -----------------------------
# Debug overlay: draw minAreaRect boxes (比画圆更符合你的粒子)
# -----------------------------
def save_debug_overlay_all_labels(raw_bgr, labels_ws, out_path):
    vis = raw_bgr.copy()
    labs = np.unique(labels_ws)
    labs = labs[labs != 0]

    for lab in labs:
        mask = (labels_ws == int(lab)).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        ctn = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(ctn)
        box = cv2.boxPoints(rect).astype(int)

        # 蓝色框
        cv2.drawContours(vis, [box], 0, (255, 0, 0), 2)

    cv2.imwrite(out_path, vis)


# -----------------------------
# Process one image
# -----------------------------
def process_one(path,
                real_um=5.0,
                time_dt_seconds=5,
                time_roi=(0, 0, 300, 160),
                scale_roi=(650, 820, 380, 220),
                margin=5,
                min_area=5000,
                debug_dir=None,
                # 分割参数
                use_clahe=True,
                clahe_clip=0.1,
                clahe_grid=(8, 8),
                median_ksize=3,
                thr_method="otsu",
                thr_offset=10,
                auto_clip_white=True,
                fg_ratio_max=0.18,
                fg_ratio_min=0.003,
                auto_step=2,
                auto_max_iter=40,
                min_cc_area=800,
                close_kernel=1,
                close_iter=1,
                open_kernel=3,
                open_iter=1,
                fill_holes=True,
                ws_min_distance=60,
                ws_threshold_rel=0.4,
                bridge_min_area=3000,
                bridge_kernel=5,
                bridge_iter=2,
                # 其他参数
                include_border=True):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    now_time = datetime.datetime.now().isoformat(timespec="seconds")

    # time: prefer filename
    tsec = read_time_seconds_from_filename(path, fps_or_dt_seconds=time_dt_seconds)
    time_info = {"seconds": int(tsec) if tsec is not None else None, "source": "filename"}
    # 如果你想用 OCR 做对照，可以打开下面两行
    t_ocr, raw = read_time_seconds_ocr(gray, roi=time_roi)
    time_info = {"seconds": int(t_ocr) if t_ocr is not None else time_info["seconds"], "raw": raw, "source": "ocr_or_filename"}

    scale = read_scale_bar(gray, roi=scale_roi, real_um=real_um)
    if scale is None:
        raise RuntimeError("Scale bar not detected. Please adjust scale_roi or threshold logic.")
    um_per_px = scale["um_per_px"]

    stem = os.path.splitext(os.path.basename(path))[0]
    labels_ws, bw, work = segment_particles(
        gray,
        time_roi=time_roi,
        scale_roi=scale_roi,
        # 分割参数
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
        median_ksize=median_ksize,
        thr_method=thr_method,
        thr_offset=thr_offset,
        auto_clip_white=auto_clip_white,
        fg_ratio_max=fg_ratio_max,
        fg_ratio_min=fg_ratio_min,
        auto_step=auto_step,
        auto_max_iter=auto_max_iter,
        min_cc_area=min_cc_area,
        close_kernel=close_kernel,
        close_iter=close_iter,
        open_kernel=open_kernel,
        open_iter=open_iter,
        fill_holes=fill_holes,
        ws_min_distance=ws_min_distance,
        ws_threshold_rel=ws_threshold_rel,
        bridge_min_area=bridge_min_area,
        bridge_kernel=bridge_kernel,
        bridge_iter=bridge_iter,
        debug_dir=debug_dir,   # 复用你传入的 debug_dir
        stem=stem,
        raw_bgr=bgr,   # <--- 这里传进去
    )


    particles = measure_ac(
        labels_ws=labels_ws,
        um_per_px=um_per_px,
        min_area=min_area,     # 使用配置中的参数
        include_border=include_border,  # 使用配置中的参数
        margin=margin,         # 使用配置中的参数
    )

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        dbg_path = os.path.join(debug_dir, os.path.basename(path).replace(".jpg", "_boxes_alllabels.png"))
        save_debug_overlay_all_labels(bgr, labels_ws, dbg_path)


    return {
        "now_time": now_time,
        "file": os.path.basename(path),
        "image_shape": {"height_px": int(gray.shape[0]), "width_px": int(gray.shape[1])},
        "time_mark": time_info,
        "scale": scale,
        "params": {
            "real_um": float(real_um),
            "time_dt_seconds": int(time_dt_seconds),
            "margin": int(margin),
            "min_area": int(min_area)
        },
        "particles": particles
    }

# -----------------------------
# Process folder -> JSON
# -----------------------------
def process_folder(in_dir, pattern="FM_t*.jpg", out_json="results.json", debug_dir="debug_boxes",
                   real_um=5.0, time_dt_seconds=5, time_roi=(0, 0, 300, 160), scale_roi=(650, 820, 380, 220),
                   margin=5, min_area=5000,
                   # 分割参数
                   use_clahe=True, clahe_clip=0.1, clahe_grid=(8, 8), median_ksize=3,
                   thr_method="otsu", thr_offset=10, auto_clip_white=True,
                   fg_ratio_max=0.18, fg_ratio_min=0.003, auto_step=2, auto_max_iter=40,
                   min_cc_area=800, close_kernel=1, close_iter=1, open_kernel=3, open_iter=1, fill_holes=True,
                   ws_min_distance=60, ws_threshold_rel=0.4,
                   bridge_min_area=3000, bridge_kernel=5, bridge_iter=2,
                   # 其他参数
                   include_border=True):
    paths = sorted(glob.glob(os.path.join(in_dir, pattern)))
    all_res = []
    
    want = {"FM_t01.jpg", "FM_t10.jpg", "FM_t23.jpg"}
    for p in paths:
        if os.path.basename(p) in want:
            print("processing:", p)
            all_res.append(process_one(p,
                                     real_um=real_um,
                                     time_dt_seconds=time_dt_seconds,
                                     time_roi=time_roi,
                                     scale_roi=scale_roi,
                                     margin=margin,
                                     min_area=min_area,
                                     debug_dir=debug_dir,
                                     # 分割参数
                                     use_clahe=use_clahe,
                                     clahe_clip=clahe_clip,
                                     clahe_grid=clahe_grid,
                                     median_ksize=median_ksize,
                                     thr_method=thr_method,
                                     thr_offset=thr_offset,
                                     auto_clip_white=auto_clip_white,
                                     fg_ratio_max=fg_ratio_max,
                                     fg_ratio_min=fg_ratio_min,
                                     auto_step=auto_step,
                                     auto_max_iter=auto_max_iter,
                                     min_cc_area=min_cc_area,
                                     close_kernel=close_kernel,
                                     close_iter=close_iter,
                                     open_kernel=open_kernel,
                                     open_iter=open_iter,
                                     fill_holes=fill_holes,
                                     ws_min_distance=ws_min_distance,
                                     ws_threshold_rel=ws_threshold_rel,
                                     bridge_min_area=bridge_min_area,
                                     bridge_kernel=bridge_kernel,
                                     bridge_iter=bridge_iter,
                                     # 其他参数
                                     include_border=include_border))

# =============================================================================
#     for p in paths[1,23:24]:
#         all_res.append(process_one(p, debug_dir=debug_dir, **kwargs))
# =============================================================================

    safe_res = to_jsonable(all_res)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(safe_res, f, ensure_ascii=False, indent=2)
    return out_json



# -----------------------------
# 配置参数 (Config Parameters)
# 所有可调参数集中在这里，方便调整和查看说明
# -----------------------------
CONFIG = {
    # ===== 文件路径配置 =====
    "IN_DIR": r"..\samples\samples",           # 你的图片所在文件夹
    "OUT_JSON": r".\out_particles.json",       # 输出JSON文件路径
    "DEBUG_DIR": r".\debug_overlay",           # 调试图片输出文件夹 (设为None就不输出debug图)

    # ===== 文件处理配置 =====
    "pattern": "FM_t*.jpg",                    # 文件名匹配模式
    "real_um": 5.0,                            # 比例尺实际长度(微米)
    "time_dt_seconds": 5,                      # 时间间隔(秒)，用于从文件名解析时间

    # ===== ROI区域配置 =====
    "time_roi": (0, 0, 260, 140),             # 时间标记区域 (x, y, w, h)
    "scale_roi": (700, 820, 324, 204),        # 比例尺区域 (x, y, w, h)

    # ===== 图像预处理参数 =====
    "use_clahe": True,                         # 是否使用CLAHE增强对比度
    "clahe_clip": 0.1,                         # CLAHE clip limit，太大容易把背景抬亮 -> "压不住白"
    "clahe_grid": (8, 8),                      # CLAHE网格大小
    "median_ksize": 3,                         # 中值滤波核大小，不是gaussian；对盐椒点更有效。不要太大(3/5)

    # ===== 阈值处理参数 =====
    "thr_method": "otsu",                      # 阈值方法: "triangle" or "otsu"
    "thr_offset": 10,                          # 阈值偏移，>0更敏感(阈值更低); <0更保守(阈值更高，用来"压白")
    "auto_clip_white": True,                   # 自动"压白"，通过监控前景占比
    "fg_ratio_max": 0.18,                      # 前景占比上限，太高就自动抬阈值
    "fg_ratio_min": 0.003,                     # 前景占比下限，太低就自动降阈值(保细边)
    "auto_step": 2,                            # 自动调整阈值的步长
    "auto_max_iter": 40,                       # 自动调整的最大迭代次数

    # ===== 二值化清理参数 =====
    "min_cc_area": 800,                        # 最小连通域面积，关键：压小点爆炸（你图上300~800都可试）
    "close_kernel": 1,                         # 闭运算核大小，默认关掉close（你说close不合理）
    "close_iter": 1,                           # 闭运算迭代次数
    "open_kernel": 3,                          # 开运算核大小，很轻的open；主要用于边缘毛刺
    "open_iter": 1,                            # 开运算迭代次数
    "fill_holes": True,                        # 是否填充孔洞

    # ===== 分水岭参数 =====
    "ws_min_distance": 60,                     # 分水岭最小距离
    "ws_threshold_rel": 0.4,                   # 分水岭相对阈值

    # ===== 桥接参数 (GAP BRIDGING) =====
    "bridge_min_area": 3000,                   # 只对大颗粒补缝，避免噪声越补越多
    "bridge_kernel": 5,                        # 关键旋钮：3/5先试；太大容易把相邻颗粒粘死
    "bridge_iter": 2,                          # 一般1就够，必要再到2

    # ===== 测量参数 =====
    "margin": 60,                              # 边界边距，用于过滤触边粒子
    "min_area": 5000,                          # 最小粒子面积过滤 (0=不过滤)
    "include_border": True,                    # 是否包含触边粒子 (True=不丢边界)

    # ===== 可选功能 =====
    "assume_bright_particle": True,            # True表示粒子边缘更亮（你这张更像这样）
    "invert": False,                           # trackpy的"亮/暗特征"开关，先保持False
}

# -----------------------------
# 10) 主程序入口
# -----------------------------
if __name__ == "__main__":
    out = process_folder(
        CONFIG["IN_DIR"],
        pattern=CONFIG["pattern"],
        out_json=CONFIG["OUT_JSON"],
        debug_dir=CONFIG["DEBUG_DIR"],
        real_um=CONFIG["real_um"],
        time_dt_seconds=CONFIG["time_dt_seconds"],
        time_roi=CONFIG["time_roi"],
        scale_roi=CONFIG["scale_roi"],
        margin=CONFIG["margin"],
        min_area=CONFIG["min_area"],
        # 分割参数
        use_clahe=CONFIG["use_clahe"],
        clahe_clip=CONFIG["clahe_clip"],
        clahe_grid=CONFIG["clahe_grid"],
        median_ksize=CONFIG["median_ksize"],
        thr_method=CONFIG["thr_method"],
        thr_offset=CONFIG["thr_offset"],
        auto_clip_white=CONFIG["auto_clip_white"],
        fg_ratio_max=CONFIG["fg_ratio_max"],
        fg_ratio_min=CONFIG["fg_ratio_min"],
        auto_step=CONFIG["auto_step"],
        auto_max_iter=CONFIG["auto_max_iter"],
        min_cc_area=CONFIG["min_cc_area"],
        close_kernel=CONFIG["close_kernel"],
        close_iter=CONFIG["close_iter"],
        open_kernel=CONFIG["open_kernel"],
        open_iter=CONFIG["open_iter"],
        fill_holes=CONFIG["fill_holes"],
        ws_min_distance=CONFIG["ws_min_distance"],
        ws_threshold_rel=CONFIG["ws_threshold_rel"],
        bridge_min_area=CONFIG["bridge_min_area"],
        bridge_kernel=CONFIG["bridge_kernel"],
        bridge_iter=CONFIG["bridge_iter"],
        # 其他参数
        include_border=CONFIG["include_border"],
    )
    print("saved:", out)
