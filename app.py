import streamlit as st
import numpy as np
import pickle
import os
import hashlib
from PIL import Image, ImageDraw, ImageFont
from mtcnn import MTCNN
from deepface import DeepFace
import cv2

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Team Face ID",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Space Mono', monospace; }
  .stApp { background: #0d0d0f; color: #e8e6e0; }
  section[data-testid="stSidebar"] { background: #111114; border-right: 1px solid #222228; }
  .card { background: #16161a; border: 1px solid #2a2a32; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
  .badge-human   { background:#1a3a2a; color:#4ade80; border:1px solid #2d6b47; border-radius:8px; padding:.4rem 1rem; display:inline-block; font-family:'Space Mono',monospace; font-size:.85rem; }
  .badge-unknown { background:#3a2a1a; color:#fb923c; border:1px solid #6b4a2d; border-radius:8px; padding:.4rem 1rem; display:inline-block; font-family:'Space Mono',monospace; font-size:.85rem; }
  .badge-error   { background:#3a1a1a; color:#f87171; border:1px solid #6b2d2d; border-radius:8px; padding:.4rem 1rem; display:inline-block; font-family:'Space Mono',monospace; font-size:.85rem; }
  .name-tag { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #60a5fa; letter-spacing: .05em; }
  hr { border-color: #222228; }
  .stButton > button { background: #1e1e24; color: #e8e6e0; border: 1px solid #3a3a46; border-radius: 8px; font-family: 'Space Mono', monospace; font-size: .8rem; padding: .5rem 1.2rem; transition: all .2s; }
  .stButton > button:hover { background: #2a2a36; border-color: #60a5fa; color: #60a5fa; }
  .section-label { font-family: 'Space Mono', monospace; font-size: .7rem; letter-spacing: .15em; color: #555568; text-transform: uppercase; margin-bottom: .5rem; }
  .member-chip { display: inline-block; background: #1a2a3a; color: #93c5fd; border: 1px solid #2d4f6b; border-radius: 20px; padding: .2rem .8rem; font-family: 'Space Mono', monospace; font-size: .75rem; margin: .2rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(_BASE_DIR, "teammate_images")
DB_PATH      = os.path.join(_BASE_DIR, "team_encodings.pkl")
MODEL_NAME   = "Facenet"          # pure-python, no dlib needed
THRESHOLD    = 10.0               # Euclidean distance threshold for Facenet
MTCNN_CONF      = 0.90          # MTCNN base confidence — 0.90 catches distant/small faces
MIN_FACE_RATIO  = 0.001         # face >= 0.1% of image area (handles faces in group/street shots)
MIN_EYE_SEP     = 0.08          # eye separation >= 8% of face width (kills texture FPs)
CACHE_VERSION   = "v2-rotation" # bump this to invalidate old pkl automatically
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Load MTCNN once ────────────────────────────────────────────────────────────
@st.cache_resource
def load_mtcnn():
    return MTCNN()

detector = load_mtcnn()

# ── Utilities ──────────────────────────────────────────────────────────────────

def pil_to_rgb(img: Image.Image) -> np.ndarray:
    # Auto-correct EXIF orientation (fixes rotated phone/camera photos)
    from PIL import ImageOps
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return np.array(img.convert("RGB"))


def folder_fingerprint(root: str) -> str:
    h = hashlib.sha1()
    for dirpath, _, files in sorted(os.walk(root)):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                fp = os.path.join(dirpath, fname)
                h.update(fp.encode())
                h.update(str(os.path.getmtime(fp)).encode())
    return h.hexdigest()


def save_db_atomic(db: dict):
    tmp = DB_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(db, f)
    os.replace(tmp, DB_PATH)


def load_db_from_disk() -> dict:
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

# ── Face helpers ───────────────────────────────────────────────────────────────

MTCNN_MIN_IMG_SIZE = 80   # MTCNN ONet needs patches >= 48px; skip tiny images

def detect_faces_mtcnn(rgb: np.ndarray) -> list[dict]:
    """
    MTCNN detection with guards:
    1. Image too small → skip entirely (avoids Conv2D crash on tiny images/scans)
    2. Confidence >= 0.90
    3. Minimum face area >= 0.1% of image
    4. Eye separation >= 8% of face width (kills texture/scan false positives)
    """
    img_h, img_w = rgb.shape[:2]

    # Guard 0: image must be large enough for MTCNN's ONet (needs 48px patches)
    if img_h < MTCNN_MIN_IMG_SIZE or img_w < MTCNN_MIN_IMG_SIZE:
        return []

    img_area = img_h * img_w

    try:
        raw = detector.detect_faces(rgb)
    except Exception:
        # Catches Conv2D empty-output errors on unusual images (CT scans, etc.)
        return []

    valid = []
    for r in raw:
        # Guard 1: confidence
        if r["confidence"] < MTCNN_CONF:
            continue
        # Guard 2: minimum face size
        x, y, w, h = r["box"]
        if (w * h) / img_area < MIN_FACE_RATIO:
            continue
        # Guard 3: eye separation — kills texture/medical-image false positives
        kp = r.get("keypoints", {})
        le = kp.get("left_eye")
        re = kp.get("right_eye")
        if le and re:
            eye_dist = abs(le[0] - re[0])
            if eye_dist < w * MIN_EYE_SEP:
                continue
        valid.append(r)
    return valid


def rotate_rgb(rgb: np.ndarray, angle: int) -> np.ndarray:
    """Rotate numpy RGB array by angle (90/180/270) with expand."""
    if angle == 0:
        return rgb
    return np.array(Image.fromarray(rgb).rotate(angle, expand=True))


def transform_box_to_original(box, angle, orig_w, orig_h):
    """
    Convert MTCNN box from rotated-image space back to original-image space.
    PIL.rotate(90)  = CCW 90  → rotated size: (orig_h, orig_w) i.e. new_W=orig_H, new_H=orig_W
    PIL.rotate(270) = CW  90  → rotated size: (orig_h, orig_w)
    PIL.rotate(180)           → rotated size: (orig_w, orig_h)

    For a top-left corner point (px, py) in rotated space:
      rotate(90)  CCW: orig_px = orig_w - py - h,  orig_py = px          (w,h swap)
      rotate(270) CW:  orig_px = py,               orig_py = orig_h - px - w  (w,h swap)
      rotate(180):     orig_px = orig_w - px - w,  orig_py = orig_h - py - h
    """
    x, y, w, h = [int(v) for v in box]
    if angle == 0:
        return [x, y, w, h]
    elif angle == 90:   # PIL CCW 90 — rotated dims are (orig_h wide, orig_w tall)
        return [orig_w - y - h, x, h, w]
    elif angle == 270:  # PIL CW  90 — rotated dims are (orig_h wide, orig_w tall)
        return [y, orig_h - x - w, h, w]
    elif angle == 180:
        return [orig_w - x - w, orig_h - y - h, w, h]
    return [x, y, w, h]


def is_human(rgb: np.ndarray):
    """
    Try detecting faces at 0°, 90°, 270°, 180°.
    Returns (human, faces_in_ORIGINAL_coords, rotated_rgb_for_cropping, angle).
    - faces box coords are transformed back to original image space (for correct drawing)
    - rotated_rgb is kept for accurate face cropping (crop must use rotated space)
    """
    orig_h, orig_w = rgb.shape[:2]
    for angle in [0, 90, 270, 180]:
        rotated = rotate_rgb(rgb, angle)
        faces   = detect_faces_mtcnn(rotated)
        if faces:
            # Transform each face box back to original image coordinates
            corrected = []
            for face in faces:
                f = dict(face)
                f["box"] = transform_box_to_original(face["box"], angle, orig_w, orig_h)
                f["_rotated_box"] = face["box"]   # keep original rotated box for cropping
                corrected.append(f)
            return True, corrected, rotated, angle
    return False, [], rgb, 0


def crop_face(rgb: np.ndarray, box: tuple) -> np.ndarray:
    """Crop + clamp face region from image."""
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    return rgb[y:y+h, x:x+w]


def get_embedding(face_crop: np.ndarray) -> np.ndarray | None:
    """
    Get a face embedding via DeepFace (Facenet model).
    Pure Python/TensorFlow — no dlib, no CMake, no C++ required.
    """
    try:
        # DeepFace expects BGR (OpenCV format)
        bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
        result = DeepFace.represent(
            img_path=bgr,
            model_name=MODEL_NAME,
            enforce_detection=False,   # we already detected with MTCNN
            detector_backend="skip",
        )
        return np.array(result[0]["embedding"])
    except Exception:
        return None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance — works better than Euclidean for Facenet."""
    a, b = a / (np.linalg.norm(a) + 1e-10), b / (np.linalg.norm(b) + 1e-10)
    return float(1 - np.dot(a, b))

# ── Dataset encoding ───────────────────────────────────────────────────────────

def build_db_from_folder(root: str, status_el) -> tuple[dict, list[str]]:
    db:  dict[str, list] = {}
    log: list[str]       = []

    if not os.path.isdir(root):
        msg = f"❌ Folder '{root}/' not found next to app.py"
        status_el.error(msg)
        return db, [msg]

    member_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    if not member_dirs:
        status_el.error(f"No sub-folders found inside '{root}/'")
        return db, [f"❌ No sub-folders in '{root}/'"]

    for member in member_dirs:
        member_path = os.path.join(root, member)
        images = sorted([
            f for f in os.listdir(member_path)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])

        embeddings  = []
        member_log  = []

        for i, fname in enumerate(images):
            fpath = os.path.join(member_path, fname)
            status_el.info(f"⏳ Encoding **{member}** — {i+1}/{len(images)}: `{fname}`")
            try:
                img  = Image.open(fpath)
                rgb  = pil_to_rgb(img)
                human, faces, rgb_rotated, angle = is_human(rgb)

                if not human:
                    member_log.append(f"  ✗ {fname} — no face detected")
                    continue

                # Largest face — crop from rotated image using _rotated_box
                face     = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                crop_box = face.get("_rotated_box", face["box"])
                crop     = crop_face(rgb_rotated, crop_box)
                emb      = get_embedding(crop)

                if emb is not None:
                    embeddings.append(emb)
                    member_log.append(f"  ✓ {fname}")
                else:
                    member_log.append(f"  ✗ {fname} — embedding failed")

            except Exception as e:
                member_log.append(f"  ✗ {fname} — {e}")

        if embeddings:
            db[member] = embeddings
            log.append(f"✅ {member}: {len(embeddings)}/{len(images)} images encoded")
        else:
            log.append(f"❌ {member}: no usable images — skipped")
        log.extend(member_log)

    return db, log

# ── Startup strategy ───────────────────────────────────────────────────────────

def init_db_strategy() -> str:
    cached = load_db_from_disk()
    if not cached:
        return "encode"
    # Invalidate if cache version changed (e.g. after rotation fix)
    if cached.get("__version__") != CACHE_VERSION:
        return "encode"
    current_fp = folder_fingerprint(DATASET_DIR) if os.path.isdir(DATASET_DIR) else "none"
    return "cache" if cached.get("__fingerprint__") == current_fp else "encode"

# ── Draw bounding boxes ────────────────────────────────────────────────────────

def draw_results(pil_img: Image.Image, face_infos: list[dict]) -> Image.Image:
    img     = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for info in face_infos:
        x, y, w, h = info["box"]
        x, y  = max(0, x), max(0, y)
        label = info.get("label", "?")
        color = info.get("color", (255, 255, 255, 200))

        draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
        text_h = 22
        draw.rectangle([x, max(0, y-text_h), x+w, y], fill=(*color[:3], 160))

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x+4, max(0, y-text_h+3)), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")

# ── Recognition ───────────────────────────────────────────────────────────────

COSINE_THRESHOLD = 0.30   # lower = stricter. Facenet cosine: ~0.20–0.40 is a good range

def recognize(rgb_rotated: np.ndarray, db: dict, faces: list[dict]) -> list[dict]:
    """rgb_rotated: the rotation-corrected image used for embedding (not original)."""
    all_embeddings, all_names = [], []
    for name, emb_list in db.items():
        if name.startswith("__"):
            continue
        for emb in emb_list:
            all_embeddings.append(emb)
            all_names.append(name)

    results = []
    for face in faces:
        # Use _rotated_box + rotated image for accurate cropping
        crop_box = face.get("_rotated_box", face["box"])
        crop = crop_face(rgb_rotated, crop_box)
        emb  = get_embedding(crop)
        info = dict(face)

        if emb is None:
            info.update(label="encode-fail", status="error", color=(150, 150, 150, 200))
            results.append(info)
            continue

        if not all_embeddings:
            info.update(label="No DB", status="no_db", color=(200, 200, 100, 200))
            results.append(info)
            continue

        distances = [cosine_distance(emb, e) for e in all_embeddings]
        best_idx  = int(np.argmin(distances))
        best_dist = distances[best_idx]

        if best_dist <= COSINE_THRESHOLD:
            info.update(
                label=all_names[best_idx],
                status="team",
                matched_name=all_names[best_idx],
                distance=best_dist,
                color=(96, 165, 250, 220),
            )
        else:
            info.update(
                label="Unknown",
                status="unknown",
                distance=best_dist,
                color=(251, 146, 60, 220),
            )
        results.append(info)
    return results

# ══════════════════════════════════════════════════════════════════════════════
# SESSION INIT
# ══════════════════════════════════════════════════════════════════════════════

if "db" not in st.session_state:
    strategy = init_db_strategy()
    if strategy == "cache":
        raw = load_db_from_disk()
        st.session_state.db         = {k: v for k, v in raw.items() if not k.startswith("__")}
        st.session_state.encode_log = ["✅ Loaded from cache (folder unchanged)"]
    else:
        st.session_state.db            = {}
        st.session_state.encode_log    = []
        st.session_state._needs_encode = True

db: dict = st.session_state.db

if st.session_state.get("_needs_encode"):
    banner = st.info("⏳ **First run** — encoding images from `teammate_images/`. Cached after this, won't repeat.")
    status = st.empty()

    new_db, log = build_db_from_folder(DATASET_DIR, status)

    current_fp       = folder_fingerprint(DATASET_DIR) if os.path.isdir(DATASET_DIR) else "none"
    to_save          = dict(new_db)
    to_save["__fingerprint__"] = current_fp
    to_save["__version__"]     = CACHE_VERSION
    save_db_atomic(to_save)

    st.session_state.db            = new_db
    st.session_state.encode_log    = log
    st.session_state._needs_encode = False
    db = new_db

    status.empty()
    banner.empty()
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2 style='font-family:Space Mono,monospace;font-size:1rem;color:#60a5fa;'>◈ TEAM ROSTER</h2>", unsafe_allow_html=True)

    enrolled = [k for k in db if not k.startswith("__")]

    if enrolled:
        for m in enrolled:
            n = len(db[m])
            st.markdown(f"<span class='member-chip'>👤 {m} &nbsp;·&nbsp; {n} encoding{'s' if n>1 else ''}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#f87171;font-size:.85rem;'>No members loaded — check folder path.</span>", unsafe_allow_html=True)

    if st.session_state.get("encode_log"):
        with st.expander("📋 Encoding log"):
            for line in st.session_state.encode_log:
                color = "#4ade80" if line.startswith("✅") else "#f87171" if line.startswith("❌") else "#888"
                st.markdown(f"<div style='font-size:.72rem;font-family:Space Mono,monospace;color:{color};'>{line}</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("🔄  Re-scan teammate_images/", use_container_width=True):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    with st.expander("📁 Expected folder layout"):
        st.code(
            "teammate_images/\n"
            "├── nidhishree/\n"
            "│   ├── 01.jpg … 10.jpg\n"
            "├── sparsha/\n"
            "│   └── 01.jpg … 10.jpg\n"
            "└── sanjana/\n"
            "    └── 01.jpg … 10.jpg",
            language="",
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Recognition sensitivity</div>", unsafe_allow_html=True)
    COSINE_THRESHOLD = st.slider("Threshold (lower = stricter)", 0.10, 0.60, COSINE_THRESHOLD, 0.01, label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='font-family:Space Mono,monospace;font-size:1.6rem;letter-spacing:.05em;margin-bottom:0;'>"
    "TEAM&nbsp;<span style='color:#60a5fa;'>FACE&nbsp;ID</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='color:#555568;font-size:.85rem;margin-bottom:1.5rem;'>"
    "MTCNN detector · Facenet embeddings · zero native dependencies</div>",
    unsafe_allow_html=True,
)

# ── Shared result renderer ─────────────────────────────────────────────────────
def render_results(pil_img: Image.Image, source_label: str):
    """Run detection + recognition on pil_img and render results."""
    rgb = pil_to_rgb(pil_img)
    col_img, col_res = st.columns([1.1, 1], gap="large")

    with st.spinner("Analysing…"):
        human_found, faces, rgb_rotated, angle = is_human(rgb)

    with col_res:
        st.markdown("<div class='section-label'>Detection results</div>", unsafe_allow_html=True)

        if not human_found:
            st.markdown("<div class='badge-error'>⚠ NOT A HUMAN</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='card' style='margin-top:.8rem;'>
              <p style='color:#888;font-size:.9rem;margin:0;'>
                No human face detected.<br>
                Could be an animal, object, illustration, or face
                too small / occluded for MTCNN.
              </p>
            </div>""", unsafe_allow_html=True)
            annotated = pil_img

        else:
            face_infos = recognize(rgb_rotated, db, faces)
            annotated  = draw_results(pil_img, face_infos)

            team_hits = [f for f in face_infos if f.get("status") == "team"]
            unknowns  = [f for f in face_infos if f.get("status") == "unknown"]
            errors    = [f for f in face_infos if f.get("status") in ("error", "no_db")]

            total = len(face_infos)
            st.markdown(
                f"<div class='badge-human'>👤 {total} human face{'s' if total>1 else ''} detected</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<hr>", unsafe_allow_html=True)

            if not enrolled:
                st.warning("No team members loaded. Check `teammate_images/` exists beside app.py.")
            else:
                for f in team_hits:
                    conf = round((1 - f["distance"]) * 100, 1)
                    st.markdown(f"""
                    <div class='card'>
                      <div class='section-label'>✅ Team member found!</div>
                      <div class='name-tag'>✦ {f['matched_name']}</div>
                      <div style='margin-top:.4rem;color:#555568;font-size:.8rem;font-family:Space Mono,monospace;'>
                        match confidence &nbsp;{conf}%
                      </div>
                    </div>""", unsafe_allow_html=True)

                for _ in unknowns:
                    st.markdown("""
                    <div class='card'>
                      <div class='badge-unknown'>🔶 Human — not a team member</div>
                      <p style='color:#888;font-size:.85rem;margin:.6rem 0 0;'>
                        Face detected but does not match Nidhishree, Sparsha, or Sanjana.
                      </p>
                    </div>""", unsafe_allow_html=True)

                if errors:
                    st.markdown(
                        "<div class='badge-error'>⚠ Encoding error on one or more faces</div>",
                        unsafe_allow_html=True,
                    )

    with col_img:
        st.markdown(f"<div class='section-label'>{source_label}</div>", unsafe_allow_html=True)
        max_display = 600
        dw, dh = annotated.size
        if dw > max_display or dh > max_display:
            scale    = min(max_display / dw, max_display / dh)
            disp_img = annotated.resize((int(dw * scale), int(dh * scale)), Image.LANCZOS)
        else:
            disp_img = annotated
        st.image(disp_img, use_container_width=False)


# ── Input tabs ────────────────────────────────────────────────────────────────
tab_upload, tab_webcam = st.tabs(["📁  Upload Image", "📷  Webcam"])

# ── Tab 1: Upload ──────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Drop an image to identify",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        pil_img = Image.open(uploaded)
        render_results(pil_img, "Annotated image")
    else:
        st.markdown("""
        <div class='card' style='text-align:center;padding:2.5rem 2rem;border-style:dashed;margin-top:1rem;'>
          <div style='font-size:2.5rem;margin-bottom:.8rem;'>📁</div>
          <div style='font-family:Space Mono,monospace;color:#60a5fa;font-size:.95rem;margin-bottom:.4rem;'>
            Drop or browse an image above
          </div>
          <div style='color:#555568;font-size:.82rem;'>
            JPG · PNG · WEBP supported
          </div>
        </div>""", unsafe_allow_html=True)

# ── Tab 2: Webcam ──────────────────────────────────────────────────────────────
with tab_webcam:
    st.markdown(
        "<div class='section-label' style='margin-bottom:.8rem;'>Live camera capture</div>",
        unsafe_allow_html=True,
    )

    # streamlit-webrtc is not always available; use st.camera_input (built-in since 1.18)
    cam_img = st.camera_input(
        "Point camera at a face and take a snapshot",
        label_visibility="collapsed",
    )

    if cam_img:
        pil_img = Image.open(cam_img)
        render_results(pil_img, "Webcam snapshot")
    else:
        st.markdown("""
        <div class='card' style='text-align:center;padding:2rem;border-style:dashed;margin-top:1rem;'>
          <div style='font-size:2.5rem;margin-bottom:.8rem;'>📷</div>
          <div style='font-family:Space Mono,monospace;color:#60a5fa;font-size:.95rem;margin-bottom:.4rem;'>
            Allow camera access and take a snapshot
          </div>
          <div style='color:#555568;font-size:.82rem;'>
            Click the capture button above · Browser will ask for camera permission
          </div>
        </div>""", unsafe_allow_html=True)