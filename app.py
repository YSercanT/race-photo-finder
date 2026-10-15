"""Streamlit arayuzu. Calistirmak icin: streamlit run app.py"""

import os
import time
from collections import Counter, deque
from io import BytesIO

import numpy as np
import streamlit as st

from photo_finder import config, database, download, faces, folders, indexer, search
from photo_finder.images import load_image

RESULTS_PER_PAGE = 40

st.set_page_config(page_title="Photo Finder", page_icon="🏃", layout="wide")
st.title("🏃 Photo Finder")
st.caption("Yaris fotograflari arasinda seni bulur: gogus numarasi + yuz tanima.")


def get_api_key() -> str:
    """Oncelik: ortam degiskeni -> .streamlit/secrets.toml -> kenar cubugundan elle giris."""
    key = os.environ.get(config.GOOGLE_API_KEY_ENV)
    if not key:
        try:
            key = st.secrets.get(config.GOOGLE_API_KEY_ENV)
        except Exception:  # noqa: BLE001 - secrets.toml hic yoksa Streamlit hata firlatir
            key = None
    return key or st.sidebar.text_input("Google API anahtari", type="password")


@st.cache_data(ttl=3600, show_spinner="Drive klasoru listeleniyor (buyuk klasorlerde 1 dakikayi bulabilir)...")
def list_folder(folder_id: str, api_key: str, with_prefix: bool) -> list[download.DriveFile]:
    # Birden fazla link varsa her linkin fotograflari kendi klasor adiyla gruplansin
    prefix = download.get_folder_name(folder_id, api_key) if with_prefix else ""
    return download.list_drive_images(folder_id, api_key, prefix=prefix)


@st.cache_data(max_entries=5, show_spinner="Yuzun analiz ediliyor...")
def reference_from_selfies(selfie_bytes: tuple[bytes, ...]) -> tuple[np.ndarray, int]:
    # Sadece bellekte cache'lenir (diske yazilmaz); ayni selfie'ler icin modeli tekrar calistirmaz
    return faces.build_reference([load_image(BytesIO(b)) for b in selfie_bytes])


@st.cache_data(max_entries=200, show_spinner=False)
def preview(file: download.DriveFile, api_key: str) -> np.ndarray:
    return load_image(BytesIO(download.fetch_image_bytes(file, api_key, max_side=800)), max_side=800)


def folder_key(folder: str) -> str:
    return f"folder::{folder}"


def set_folders(paths: list[str], value: bool) -> None:
    """Buton callback'i: widget'lar cizilmeden ONCE calisir, bu yuzden onay kutularini degistirebilir."""
    for path in paths:
        st.session_state[folder_key(path)] = value


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours} sa {minutes} dk" if hours else f"{minutes} dk {secs} sn"


api_key = get_api_key()
if not api_key:
    st.info("Baslamak icin soldaki kutuya Google API anahtarini gir ya da GOOGLE_API_KEY ortam degiskenini ayarla.")
    st.stop()

# --- 1. Yaris linkleri ---
links_text = st.text_area(
    "Google Drive klasor linkleri (her satira bir tane; ayni yarisin birden fazla linki olabilir)",
    placeholder="https://drive.google.com/drive/folders/...",
    height=100,
)
try:
    folder_ids = list(dict.fromkeys(download.parse_folder_id(line) for line in links_text.splitlines() if line.strip()))
    if not folder_ids:
        st.stop()
    files_by_id: dict[str, download.DriveFile] = {}
    for folder_id in folder_ids:
        for f in list_folder(folder_id, api_key, with_prefix=len(folder_ids) > 1):
            files_by_id[f.id] = f  # ayni fotograf iki linkte de varsa bir kez say
except download.DriveError as exc:
    st.error(str(exc))
    st.stop()

files = list(files_by_id.values())
if not files:
    st.warning("Bu klasorlerde fotograf bulunamadi.")
    st.stop()

# --- 2. Klasor secimi ---
counts = Counter(f.folder for f in files)
all_folders = sorted(counts)
groups = folders.group_folders(all_folders)

st.subheader("Hangi klasorler taransin?")
st.caption(f"{len(files)} fotograf, {len(all_folders)} klasor. Expo, odul toreni ya da toplu fotograflarda "
           "da olabilirsin; hangilerinin onemli olduguna sen karar ver.")

col_kw, col_add, col_remove, col_all, col_none = st.columns([4, 1.3, 1.3, 1, 1], vertical_alignment="bottom")
keywords = folders.parse_keywords(col_kw.text_input(
    "Anahtar kelimeyle toplu secim (virgulle ayir)", placeholder="21k, finish, pazar"))
matching = [f for f in all_folders if folders.matches_any(f, keywords)] if keywords else []
col_add.button(f"Eslesenleri sec ({len(matching)})", on_click=set_folders, args=(matching, True),
               disabled=not matching, use_container_width=True)
col_remove.button("Eslesenleri kaldir", on_click=set_folders, args=(matching, False),
                  disabled=not matching, use_container_width=True)
col_all.button("Tumu", on_click=set_folders, args=(all_folders, True), use_container_width=True)
col_none.button("Hicbiri", on_click=set_folders, args=(all_folders, False), use_container_width=True)

selected_folders = []
for top, paths in groups.items():
    n_selected = sum(st.session_state.get(folder_key(p), False) for p in paths)
    n_photos = sum(counts[p] for p in paths)
    # Baslik SABIT olmali: Streamlit paneli basligindan tanir; baslik degisirse (or. secili sayisi)
    # paneli yeni sanip kapali cizer ve her tiklamada kapanir. Degisen bilgi panelin icinde.
    with st.expander(f"📁 {top} — {len(paths)} klasor, {n_photos} foto"):
        if len(paths) > 1:
            c1, c2, c3 = st.columns([1, 1, 4], vertical_alignment="center")
            c1.button("Hepsini sec", key=f"all::{top}", on_click=set_folders, args=(paths, True))
            c2.button("Hicbirini", key=f"none::{top}", on_click=set_folders, args=(paths, False))
            c3.caption(f"{n_selected}/{len(paths)} klasor secili")
        for path in paths:
            if st.checkbox(folders.label(path, top if top != folders.ROOT_LABEL else "", counts),
                           key=folder_key(path)):
                selected_folders.append(path)

if selected_folders:
    per_group = Counter(path.split("/", 1)[0] or folders.ROOT_LABEL for path in selected_folders)
    st.caption("Secili: " + " · ".join(f"{top} ({n})" for top, n in per_group.items()))

selected_files = [f for f in files if f.folder in set(selected_folders)]
selected_ids = {f.id for f in selected_files}

# --- 3. Tarama (indeksleme) ---
conn = database.connect()
remaining = len(selected_ids - database.processed_ids(conn, selected_ids))
st.write(f"**{len(selected_files)}** fotograf secili, **{len(selected_files) - remaining}** tanesi daha once tarandi.")

if remaining and st.button(f"Tara ({remaining} fotograf)", type="primary"):
    # Modelleri sayactan ONCE yukle: yoksa ilk fotografa ~1 dk eklenir ve sure tahmini saatler cikar
    with st.spinner("Modeller yukleniyor (ilk seferde ~1 dk)..."):
        indexer.load_models()
    bar = st.progress(0.0)
    error_box = st.empty()
    errors = []
    start = time.time()
    recent = deque(maxlen=30)  # son 30 fotografin bitis zamani: tahmin guncel hiza gore
    for p in indexer.index_drive_files(conn, selected_files, api_key):
        recent.append(time.time())
        per_photo = (recent[-1] - recent[0]) / (len(recent) - 1) if len(recent) > 1 else 0
        eta = per_photo * (p.total - p.done)
        bar.progress(p.done / p.total,
                     text=f"{p.done}/{p.total} · kalan ~{format_duration(eta)} · {p.file.folder}/{p.file.name}")
        if p.error:
            errors.append(f"{p.file.folder}/{p.file.name}: {p.error}")
            # Hatayi aninda goster: sessizce birikirse saatlerce fark edilmez
            error_box.warning(f"{len(errors)} fotograf islenemedi. Son hata: {p.error}")
    bar.empty()
    st.success(f"Tarama bitti: {format_duration(time.time() - start)}")
    if errors:
        with st.expander(f"{len(errors)} fotograf islenemedi"):
            st.code("\n".join(errors))
    st.rerun()

if not selected_ids:
    st.stop()

# --- 4. Arama ---
st.divider()
st.subheader("Seni nasil bulalim?")
st.caption("Ikisini birlikte kullanmak en iyisi: numarasi gorunen VEYA yuzu eslesen her fotograf gelir. "
           "Numaran yagmurluk altinda kaldiysa ya da hic yoksa sadece selfie yuklemen yeterli.")
col_bib, col_selfie = st.columns(2)
bib = col_bib.text_input("Gogus numaran (istege bagli)", placeholder="1234")
selfies = col_selfie.file_uploader(
    "Kendi fotograflarin (istege bagli; 1-3 tane, yuzun net gorunsun)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)
col_threshold, col_background = st.columns([3, 2], vertical_alignment="bottom")
threshold = col_threshold.slider("Yuz benzerlik esigi", 0.20, 0.80, config.FACE_SIMILARITY_THRESHOLD, 0.05,
                                 help="Dusuk: daha cok fotograf, daha cok yanlis eslesme. Yuksek: tersi.")
include_background = col_background.checkbox(
    "Arka plandaki kucuk yuzleri de ara", value=True,
    help="Baskasinin fotografinda arkada kalmis olabilirsin (start, expo, finis). Kucuk yuzler daha az "
         "net oldugu icin yanlis eslesme biraz artabilir; sonuclarda 'arka plan' diye isaretlenir.")
min_face_size = config.FACE_MIN_SIZE_INDEX if include_background else config.FACE_MIN_SIZE

if not (bib or selfies):
    st.info("Numarani yaz, selfie yukle ya da ikisini birden yap.")
    st.stop()

reference = None
if selfies:
    try:
        reference, n_used = reference_from_selfies(tuple(s.getvalue() for s in selfies))
        if n_used < len(selfies):
            st.warning(f"{len(selfies)} fotograftan {n_used} tanesinde yuz bulundu; aramada onlar kullaniliyor.")
    except ValueError as exc:
        # Selfie ise yaramadiysa numarayla aramaya devam et; ikisi de yoksa dur
        (st.warning if bib else st.error)(str(exc) + (" Sadece numarayla araniyor." if bib else ""))
        if not bib:
            st.stop()

matches = search.search(conn, selected_ids, bib=bib, reference=reference, face_threshold=threshold,
                        min_face_size=min_face_size)

# --- 5. Sonuclar ---
st.subheader(f"{len(matches)} fotograf bulundu")
if len(matches) > RESULTS_PER_PAGE:
    st.caption(f"Ilk {RESULTS_PER_PAGE} tanesi gosteriliyor.")

columns = st.columns(4)
for i, match in enumerate(matches[:RESULTS_PER_PAGE]):
    photo = files_by_id[match.file_id]
    reasons = []
    if match.bib_match:
        reasons.append("🔢 numara")
    if match.face_score is not None and match.face_score >= threshold:
        background = " (arka plan)" if match.face_size < config.FACE_MIN_SIZE else ""
        reasons.append(f"🙂 yuz {match.face_score:.2f}{background}")
    with columns[i % 4]:
        st.image(preview(photo, api_key), channels="BGR")
        st.caption(f"{' · '.join(reasons)}  \n{photo.folder}/{photo.name}  \n[Drive'da ac]({photo.view_url})")
