"""Google Drive klasorlerini (alt klasorler dahil) listeleme ve fotograflari bellege indirme.

Diske hicbir sey yazilmaz: her fotograf bayt olarak gelir, islenir, atilir.
"""

import re
from dataclasses import dataclass

import requests

from photo_finder import config

API_URL = "https://www.googleapis.com/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"


class DriveError(RuntimeError):
    """Kullaniciya gosterilebilecek, anlasilir Drive hatasi."""


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str
    folder: str                         # ana klasore gore yol: "CP1/Ahmet"; ana klasordeyse ""
    thumbnail_link: str | None = None   # Drive'in urettigi kucuk surum (kisa omurlu link)

    @property
    def view_url(self) -> str:
        return f"https://drive.google.com/file/d/{self.id}/view"


def parse_folder_id(link: str) -> str:
    """'https://drive.google.com/drive/folders/<ID>?usp=sharing' -> '<ID>'"""
    link = link.strip()
    for pattern in (r"/folders/([\w-]+)", r"[?&]id=([\w-]+)"):
        if match := re.search(pattern, link):
            return match.group(1)
    if re.fullmatch(r"[\w-]{10,}", link):  # sadece ID yapistirilmis
        return link
    raise DriveError("Gecerli bir Google Drive klasor linki degil.")


def _get(session: requests.Session, url: str, params: dict) -> requests.Response:
    response = session.get(url, params=params, timeout=60)
    if response.status_code in (403, 404):
        raise DriveError(
            "Drive erisimi reddedildi. Klasor 'Baglantiya sahip herkes' ile paylasilmis mi, "
            "API anahtari dogru ve Drive API acik mi?"
        )
    response.raise_for_status()
    return response


def _list_children(session: requests.Session, folder_id: str, api_key: str):
    """Bir klasorun dogrudan icindekileri dondurur; Drive sayfa sayfa (1000'er) verir."""
    params = {
        "q": f"'{folder_id}' in parents and trashed = false",
        "fields": "nextPageToken, files(id, name, mimeType, thumbnailLink)",
        "pageSize": 1000,
        "key": api_key,
        "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true",
    }
    while True:
        data = _get(session, API_URL, params).json()
        yield from data.get("files", [])
        if "nextPageToken" not in data:
            return
        params["pageToken"] = data["nextPageToken"]


def get_folder_name(folder_id: str, api_key: str, session: requests.Session | None = None) -> str:
    session = session or requests.Session()
    params = {"fields": "name", "key": api_key, "supportsAllDrives": "true"}
    return _get(session, f"{API_URL}/{folder_id}", params).json()["name"]


def list_drive_images(folder_id: str, api_key: str, session: requests.Session | None = None,
                      prefix: str = "") -> list[DriveFile]:
    """Klasor agacini gezip tum fotograflari, bulunduklari alt klasor yoluyla dondurur.

    `prefix`: birden fazla link birlikte taranirken her linkin yollari kendi adiyla baslasin diye.
    """
    session = session or requests.Session()
    images = []
    stack = [(folder_id, prefix)]  # (klasor ID, yol): ozyineleme yerine yigin, derin agaclarda da guvenli
    while stack:
        current_id, path = stack.pop()
        for item in _list_children(session, current_id, api_key):
            if item["mimeType"] == FOLDER_MIME:
                stack.append((item["id"], f"{path}/{item['name']}".lstrip("/")))
            elif item["mimeType"].startswith("image/"):
                images.append(DriveFile(item["id"], item["name"], path, item.get("thumbnailLink")))
    return sorted(images, key=lambda f: (f.folder, f.name))


def fetch_image_bytes(file: DriveFile, api_key: str, session: requests.Session | None = None,
                      max_side: int = config.MAX_IMAGE_SIDE) -> bytes:
    """Fotografi bellege indirir.

    Once Drive'in kucultulmus surumunu dener (~0.5 MB): modellere zaten en fazla `max_side`
    piksel veriyoruz, 10 MB'lik orijinali indirmek 20 kat bant genisligi demek.
    Olmazsa orijinale geri duser.
    """
    session = session or requests.Session()
    if file.thumbnail_link:
        # Link "...=s220" ile biter; boyut parametresini degistirerek buyuk surumu isteriz
        url = re.sub(r"=s\d+$", f"=s{max_side}", file.thumbnail_link)
        try:
            response = session.get(url, timeout=60)
            if response.ok and response.headers.get("content-type", "").startswith("image/"):
                return response.content
        except requests.RequestException:
            pass
    params = {"alt": "media", "key": api_key, "supportsAllDrives": "true"}
    return _get(session, f"{API_URL}/{file.id}", params).content
