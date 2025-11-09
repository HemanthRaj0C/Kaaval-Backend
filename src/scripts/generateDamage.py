#!/usr/bin/env python3
import os, csv, math, random, argparse
import numpy as np
import cv2

# -----------------------------
# Basic utilities
# -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def clamp01(x):
    return np.clip(x, 0, 1)

def to_float(img):
    return img.astype(np.float32) / 255.0

def to_u8(imgf):
    return (clamp01(imgf) * 255.0).round().astype(np.uint8)

def resize_112(img):
    return cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)

# -----------------------------
# Degradations (building blocks)
# -----------------------------
def gaussian_blur(img, ksize):
    k = max(1, int(ksize))
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(img, (k, k), 0)

def motion_blur(img, ksize, angle_deg):
    k = max(3, int(ksize))
    if k % 2 == 0: k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    cv2.line(kernel, (k//2, 0), (k//2, k-1), 1, 1)
    # rotate kernel
    M = cv2.getRotationMatrix2D((k/2-0.5, k/2-0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= kernel.sum() if kernel.sum() != 0 else 1
    return cv2.filter2D(img, -1, kernel)

def downscale_upscale(img, scale=0.5):
    h, w = img.shape[:2]
    nh, nw = max(8, int(h*scale)), max(8, int(w*scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def add_gaussian_noise(img, sigma=0.05):
    f = to_float(img)
    noise = np.random.normal(0, sigma, f.shape).astype(np.float32)
    return to_u8(f + noise)

def add_poisson_noise(img, scale=30):
    vals = 2 ** np.ceil(np.log2(scale))
    noisy = np.random.poisson(img.astype(np.float32) * vals) / float(vals)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def change_contrast_brightness(img, alpha=1.0, beta=0.0):
    # alpha: contrast, beta: brightness (0-255 scale)
    out = cv2.convertScaleAbs(img, alpha=float(alpha), beta=float(beta))
    return out

def desaturate(img, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= float(np.clip(factor, 0.0, 1.0))
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def color_cast(img, b_mult=1.0, g_mult=1.0, r_mult=1.0):
    f = to_float(img)
    f[...,0] *= b_mult
    f[...,1] *= g_mult
    f[...,2] *= r_mult
    return to_u8(f)

def gamma_correct(img, gamma=1.0):
    inv = 1.0 / max(1e-6, gamma)
    table = np.array([(i/255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def jpeg_compress(img, quality=40):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode('.jpg', img, encode_param)
    if not ok: return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img

def vignette(img, strength=0.6):
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols*strength)
    kernel_y = cv2.getGaussianKernel(rows, rows*strength)
    kernel = kernel_y @ kernel_x.T
    mask = kernel / kernel.max()
    v = np.empty_like(img, dtype=np.float32)
    for c in range(3):
        v[..., c] = to_float(img)[..., c] * mask
    return to_u8(v)

def erode_edges(img, ksize=3):
    k = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, k, iterations=1)

def random_occlusion(img, max_frac=0.15):
    h, w = img.shape[:2]
    oh = random.randint(int(h*0.05), int(h*max_frac))
    ow = random.randint(int(w*0.05), int(w*max_frac))
    y = random.randint(0, h-oh)
    x = random.randint(0, w-ow)
    out = img.copy()
    # dark patch
    color = int(random.uniform(10, 50))
    cv2.rectangle(out, (x,y), (x+ow, y+oh), (color, color, color), -1)
    return out

# -----------------------------
# Recipes (light / severe / cctv)
# -----------------------------
def make_light(img):
    img = resize_112(img)
    img = gaussian_blur(img, ksize=random.choice([3,5,7]))
    img = change_contrast_brightness(img, alpha=random.uniform(0.80, 0.95), beta=random.uniform(-15, 5))
    if random.random() < 0.5:
        img = desaturate(img, factor=random.uniform(0.6, 0.9))
    if random.random() < 0.4:
        img = gamma_correct(img, gamma=random.uniform(0.9, 1.2))
    if random.random() < 0.4:
        img = vignette(img, strength=random.uniform(0.3, 0.6))
    return img

def make_severe(img):
    img = resize_112(img)
    img = gaussian_blur(img, ksize=random.choice([9,11,13,15]))
    img = motion_blur(img, ksize=random.choice([7,9,11]), angle_deg=random.uniform(-45, 45))
    img = downscale_upscale(img, scale=random.uniform(0.25, 0.6))
    img = add_gaussian_noise(img, sigma=random.uniform(0.03, 0.10))
    img = add_poisson_noise(img, scale=random.randint(10, 60))
    img = change_contrast_brightness(img, alpha=random.uniform(0.65, 0.85), beta=random.uniform(-25, 10))
    img = color_cast(img,
                     b_mult=random.uniform(0.9, 1.1),
                     g_mult=random.uniform(0.9, 1.1),
                     r_mult=random.uniform(0.7, 0.95))  # slight pallor
    img = jpeg_compress(img, quality=random.randint(25, 45))
    if random.random() < 0.6:
        img = random_occlusion(img, max_frac=0.20)
    if random.random() < 0.4:
        img = erode_edges(img, ksize=random.choice([3,5]))
    return img

def make_cctv(img):
    img = resize_112(img)
    img = downscale_upscale(img, scale=random.uniform(0.2, 0.5))
    img = motion_blur(img, ksize=random.choice([7,9,11]), angle_deg=random.uniform(-10, 10))
    img = add_gaussian_noise(img, sigma=random.uniform(0.02, 0.06))
    img = change_contrast_brightness(img, alpha=random.uniform(0.75, 0.9), beta=random.uniform(-20, 10))
    img = jpeg_compress(img, quality=random.randint(20, 40))
    return img

RECIPES = {
    "light": make_light,
    "severe": make_severe,
    "cctv": make_cctv,
}

# -----------------------------
# Main generator
# -----------------------------
def process_folder(src, dst, modes, per_image=2, seed=42):
    random.seed(seed)
    ensure_dir(dst)
    manifest_path = os.path.join(dst, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orig_file", "out_file", "mode", "params"])

        files = [x for x in os.listdir(src) if x.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        files.sort()
        for i, fname in enumerate(files):
            in_path = os.path.join(src, fname)
            img = cv2.imread(in_path)
            if img is None:
                print(f"[SKIP] failed to read {in_path}")
                continue

            base, _ = os.path.splitext(fname)
            for m in modes:
                make = RECIPES[m]
                for k in range(per_image):
                    out_img = make(img)
                    out_name = f"{base}_{m}_{k+1}.jpg"
                    out_path = os.path.join(dst, out_name)
                    cv2.imwrite(out_path, out_img)
                    w.writerow([fname, out_name, m, "auto"])
            if (i+1) % 200 == 0:
                print(f"[{i+1}] processed...")

    print(f"✅ Done. Damaged images saved to: {dst}")
    print(f"📝 Manifest: {manifest_path}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate artificial decomposed/damaged faces.")
    ap.add_argument("--src", default="backend/data/aligned", help="Folder with aligned faces (112x112 recommended)")
    ap.add_argument("--dst", default="backend/data/damaged", help="Output folder for damaged variants")
    ap.add_argument("--modes", default="light,severe,cctv", help="Comma-separated: light,severe,cctv")
    ap.add_argument("--per-image", type=int, default=2, help="How many variants per mode per image")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in RECIPES]
    if not modes:
        raise SystemExit("No valid modes selected. Use any of: light,severe,cctv")

    process_folder(args.src, args.dst, modes, per_image=args.per_image, seed=args.seed)
