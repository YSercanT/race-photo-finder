"""Klasor yollari ve esik degerleri; tum ayarlar tek yerde."""

from pathlib import Path

# --- Yollar ---
# Bu dosya photo_finder/config.py -> iki ust klasor proje koku
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RACES_DIR = DATA_DIR / "races"
REFERENCE_DIR = DATA_DIR / "reference"
LABELS_DIR = DATA_DIR / "labels"
SPLITS_FILE = DATA_DIR / "splits.json"
INDEX_DIR = ROOT_DIR / "index"
INDEX_DB = INDEX_DIR / "photos.db"

# --- Google Drive ---
# Anahtar koda yazilmaz: ortam degiskeninden veya .streamlit/secrets.toml'dan okunur
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
DOWNLOAD_WORKERS = 8  # ayni anda indirilen fotograf sayisi

# --- Goruntu ---
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_IMAGE_SIDE = 2048

# --- Modeller ---
FACE_MODEL = "buffalo_l"
# Tespit modeli resmi birden fazla olcekte tarar (insightface 2.x):
#   128  -> kameraya cok yakin, ekrani dolduran yuzler (selfie, podyum yakin cekimi)
#   1024 -> uzaktaki / arka plandaki kucuk yuzler
# Tek olcek (sadece 1024) selfie'lerdeki buyuk yuzleri KACIRIYOR: model o boyutu taniyamiyor.
FACE_DET_SIZES = [(128, 128), (1024, 1024)]
FACE_MIN_DET_SCORE = 0.5  # "bu bir yuz" guveni

# Yuz boyutu (kisa kenar, 2048px'lik resimde piksel):
FACE_MIN_SIZE_INDEX = 20  # taramada bundan buyukler kaydedilir (ham veri; filtre aramada)
FACE_MIN_SIZE = 40        # aramada varsayilan: kucuk yuzlerin embedding'i daha az guvenilir
# Arayuzde "arka plandaki yuzler" acikken FACE_MIN_SIZE_INDEX'e kadar inilir

# --- Esikler (evaluate.py ile dev setinde ayarlanacak) ---
# cosine similarity; ustu "ayni kisi". Ilk gercek gozlem (Ephesus, 3 selfie): kendi fotograflarin
# 0.35-0.60 (sapkali / basi egik olanlar ~0.35), en benzer yabancilar 0.22-0.28. Selfie ile yaris
# fotografi arasindaki aci/isik farki skorlari dusuruyor. evaluate.py ile kesinlestirilecek.
FACE_SIMILARITY_THRESHOLD = 0.32
OCR_MIN_CONFIDENCE = 0.50         # EasyOCR guveni; alti yok sayilir

# --- Gogus numarasi ---
# Tek haneliler cok gurultu getirir (reklam, saat, kronometre); 6+ hane genelde numara degil.
BIB_MIN_DIGITS = 2
BIB_MAX_DIGITS = 5
