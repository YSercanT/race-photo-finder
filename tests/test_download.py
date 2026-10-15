import pytest

from photo_finder.download import FOLDER_MIME, DriveError, DriveFile, list_drive_images, parse_folder_id

ID = "1AbC-dEf_GhIjKlMnOp"


@pytest.mark.parametrize("link", [
    f"https://drive.google.com/drive/folders/{ID}?usp=sharing",
    f"https://drive.google.com/drive/u/0/folders/{ID}",
    f"https://drive.google.com/drive/mobile/folders/{ID}?usp=sharing&fbclid=PAZXh0bgNhZW0CMTEAAa_aem_dDD",
    f"https://drive.google.com/open?id={ID}",
    f"  {ID}  ",
])
def test_parse_folder_id(link):
    assert parse_folder_id(link) == ID


def test_parse_folder_id_rejects_garbage():
    with pytest.raises(DriveError):
        parse_folder_id("https://example.com/foto")


class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


class FakeSession:
    """Drive API'yi taklit eder: {klasor_id: [sayfa1, sayfa2, ...]}"""

    def __init__(self, tree, status=200):
        self.tree, self.status = tree, status

    def get(self, url, params, timeout):
        folder_id = params["q"].split("'")[1]
        pages = self.tree[folder_id]
        page = int(params.get("pageToken", 0))
        data = {"files": pages[page]}
        if page + 1 < len(pages):
            data["nextPageToken"] = str(page + 1)
        return FakeResponse(data, self.status)


def img(id_, name):
    return {"id": id_, "name": name, "mimeType": "image/jpeg"}


def test_lists_subfolders_and_pages():
    tree = {
        "root": [[img("a", "a.jpg"), {"id": "cp1", "name": "CP1", "mimeType": FOLDER_MIME}],
                 [{"id": "notes", "name": "notes.txt", "mimeType": "text/plain"}]],
        "cp1": [[img("b", "b.jpg"), {"id": "ali", "name": "Ali", "mimeType": FOLDER_MIME}]],
        "ali": [[img("c", "c.jpg")]],
    }
    files = list_drive_images("root", "KEY", session=FakeSession(tree))
    assert files == [DriveFile("a", "a.jpg", ""), DriveFile("b", "b.jpg", "CP1"), DriveFile("c", "c.jpg", "CP1/Ali")]


def test_forbidden_gives_friendly_error():
    with pytest.raises(DriveError, match="paylasilmis"):
        list_drive_images("root", "KEY", session=FakeSession({"root": [[]]}, status=403))
