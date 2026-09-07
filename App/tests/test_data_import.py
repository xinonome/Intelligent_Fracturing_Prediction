from pathlib import Path
import csv
import hashlib
import io
import json
from types import SimpleNamespace
import zipfile

import pytest

from App.services import data_import_service as service
from App.services.data_import_service import import_tables, inspect_table
from App.services.txt_parser_service import TXT_FIELD_MAPPING, TXT_HEADERS, TxtValidationError, parse_txt


ROOT = Path(__file__).resolve().parents[2]


def test_standard_construction_table_is_detected():
    result = inspect_table(ROOT / "Data" / "raw_frac" / "FDBH26.xlsx")

    assert result.can_import is True
    assert result.field_profile == "标准施工表"
    assert "FDBH" in result.headers
    assert "SGSJ" in result.headers
    assert "工况识别与风险预测" in result.suggested_scenarios


def test_composite_table_is_not_registered_as_a_single_stage():
    result = inspect_table(ROOT / "Data" / "raw_frac" / "便签数据1(已自动还原).xlsx")

    assert result.can_import is False
    assert result.field_profile == "综合数据"
    assert "综合文件" in result.reason


VALUES = ["1234.50", "12.25", "67.80", "3.5", "2", "2026-09-05 10:00:00.125", "65.75", "8.5", "4"]


def write_txt(path, rows=None, *, delimiter=",", header=False, encoding="utf-8-sig"):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter)
    if header:
        writer.writerow(TXT_HEADERS)
    writer.writerows(rows if rows is not None else [VALUES])
    path.write_bytes(stream.getvalue().encode(encoding))
    return path


@pytest.mark.parametrize("encoding", ["utf-8-sig", "gb18030", "utf-16"])
@pytest.mark.parametrize("delimiter", [",", "\t"])
@pytest.mark.parametrize("header", [False, True])
def test_txt_original_layout_encodings_and_headers(tmp_path, encoding, delimiter, header):
    source = write_txt(tmp_path / "source.TXT", encoding=encoding, delimiter=delimiter, header=header)
    parsed = parse_txt(source)
    assert parsed.rows[0].values == tuple(VALUES)
    assert parsed.rows[0].timestamp.microsecond == 125000
    assert parsed.rows[0].line_number == (2 if header else 1)
    assert parsed.has_header is header
    assert parsed.source_bytes == source.read_bytes()
    assert inspect_table(source).can_import


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 6, 7, 8])
@pytest.mark.parametrize("invalid", ["", "bad", "NaN", "inf", "-1", "1e400", "1e-400"])
def test_txt_rejects_bad_numeric_fields_without_writing(tmp_path, index, invalid):
    row = VALUES.copy()
    row[index] = invalid
    source = write_txt(tmp_path / "bad.txt", [VALUES, row], header=True)
    target = tmp_path / "raw_frac"
    inspection = inspect_table(source)
    assert not inspection.can_import
    assert "第 3 行" in inspection.reason
    assert TXT_HEADERS[index] in inspection.reason
    result = import_tables([source], target, well_name="真实井", stage_id="12")
    assert not result.imported
    assert len(result.rejected) == 1
    assert not target.exists()


@pytest.mark.parametrize("invalid", ["", "10:00:00", "2026-02-30 10:00:00", "2026-09-05", "2026-09-05 24:00:00", "garbage", "2026-09-05 10:00:00+08:00"])
def test_txt_rejects_invalid_or_ambiguous_time(tmp_path, invalid):
    row = VALUES.copy()
    row[5] = invalid
    source = write_txt(tmp_path / "bad.txt", [row])
    with pytest.raises(TxtValidationError, match="第 1 行「时间」"):
        parse_txt(source)


@pytest.mark.parametrize("time", [VALUES[5], "2026-09-05 09:59:59"])
def test_txt_rejects_duplicate_or_reversed_time(tmp_path, time):
    row = VALUES.copy()
    row[5] = time
    with pytest.raises(TxtValidationError, match="严格递增"):
        parse_txt(write_txt(tmp_path / "bad.txt", [VALUES, row]))


@pytest.mark.parametrize("rows", [[], [VALUES[:-1]], [VALUES + ["extra"]], [list(TXT_HEADERS)]])
def test_txt_rejects_empty_header_only_or_wrong_column_count(tmp_path, rows):
    source = write_txt(tmp_path / "bad.txt", rows)
    assert not inspect_table(source).can_import


@pytest.mark.parametrize("payload", [b"\xff", b"1,2\x00", b'"unclosed,2,3', "1,2,�".encode()])
def test_txt_rejects_corrupt_encoding_and_csv(tmp_path, payload):
    source = tmp_path / "bad.txt"
    source.write_bytes(payload)
    assert not inspect_table(source).can_import


def test_txt_import_preserves_all_nine_fields_identity_and_source(tmp_path, monkeypatch):
    source = write_txt(tmp_path / "not-a-real-well.txt", header=True)
    original = source.read_bytes()
    target = tmp_path / "Data" / "raw_frac"
    monkeypatch.setattr(service, "PATHS", SimpleNamespace(data=tmp_path / "Data"))
    invalidations = []
    monkeypatch.setattr(service, "clear_dataset_catalog_cache", lambda: invalidations.append(True))
    result = import_tables([source], well_name=" 用户填写井名 ", stage_id=" 012 ")
    assert not result.rejected
    assert invalidations == [True]
    csv_path, = result.imported
    assert csv_path.parent == target
    assert csv_path.suffix == ".csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert [row[name] for name in TXT_HEADERS] == VALUES
    for name, value in zip(TXT_HEADERS, VALUES):
        assert row[TXT_FIELD_MAPPING[name]] == value
    assert row["LJYL"] == "1234.50"  # Liquid and sand must never be swapped.
    assert row["LJSL"] == "67.80"
    assert row["JTBH"] == "用户填写井名"
    assert row["FDBH"] == "012"
    assert row["WORKING_TYPE"] == row["PROBABILITY"] == row["BZJD"] == ""
    assert row["SOURCE_LINE"] == "2"
    assert row["SOURCE_FILE"] == source.name
    provenance = json.loads(csv_path.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert provenance["source_path"] == str(source.resolve())
    assert provenance["source_sha256"] == row["SOURCE_SHA256"] == hashlib.sha256(original).hexdigest()
    assert provenance["field_mapping"] == TXT_FIELD_MAPPING
    assert provenance["identity_source"] == "user_input"
    assert provenance["well_name"] == "用户填写井名"
    assert provenance["stage_id"] == "012"
    assert provenance["row_count"] == 1
    assert (target / provenance["original_file"]).read_bytes() == source.read_bytes() == original
    assert inspect_table(csv_path).field_profile == "标准施工表"
    assert not list(target.glob(".txt_import_*"))


@pytest.mark.parametrize("well_name,stage_id", [(None, None), ("", "12"), ("井", " "), ("null", "12"), ("井", "a\nb")])
def test_txt_requires_explicit_valid_identity(tmp_path, well_name, stage_id):
    source = write_txt(tmp_path / "fake-well-stage12.txt")
    result = import_tables([source], tmp_path / "raw_frac", well_name=well_name, stage_id=stage_id)
    assert not result.imported
    assert len(result.rejected) == 1
    assert not (tmp_path / "raw_frac").exists()


def test_batch_metadata_and_legacy_csv_are_independent(tmp_path):
    first = write_txt(tmp_path / "one.txt")
    second = write_txt(tmp_path / "two.txt")
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("JTBH,FDBH,SGSJ,PL,SB\n原井,8,2026-09-05 10:00:00,12,3\n", encoding="utf-8-sig")
    result = import_tables([first, second, legacy], tmp_path / "raw_frac", metadata_by_path={
        first: {"well_name": "甲井", "stage_id": "1"},
        str(second): {"well_name": "乙井", "stage_id": "2"},
    })
    assert not result.rejected
    assert len(result.imported) == 3
    for path, expected in zip(result.imported[:2], [("甲井", "1"), ("乙井", "2")]):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        assert (row["JTBH"], row["FDBH"]) == expected
    assert result.imported[2].read_bytes() == legacy.read_bytes()


def test_txt_is_revalidated_after_preflight(tmp_path, monkeypatch):
    source = write_txt(tmp_path / "source.txt")
    inspection = inspect_table(source)
    monkeypatch.setattr(service, "inspect_table", lambda _: inspection)
    row = VALUES.copy()
    row[6] = "broken"
    write_txt(source, [row])
    result = import_tables([source], tmp_path / "raw_frac", well_name="井", stage_id="1")
    assert not result.imported
    assert "泵压" in result.rejected[0][1]
    assert not (tmp_path / "raw_frac").exists()


def test_txt_publish_failure_rolls_back_and_batch_continues(tmp_path, monkeypatch):
    source = write_txt(tmp_path / "source.txt")
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("FDBH,SGSJ,PL,SB,SGBY\n1,2026-09-05 10:00:00,1,2,3", encoding="utf-8")
    original_rename = Path.rename

    def fail_csv_publish(path, target):
        if path.name == "table.csv":
            raise PermissionError("simulated publish failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_csv_publish)
    target = tmp_path / "raw_frac"
    result = import_tables([source, legacy], target, well_name="井", stage_id="1")
    assert len(result.imported) == len(result.rejected) == 1
    assert result.imported[0].read_bytes() == legacy.read_bytes()
    assert list((target / "originals").iterdir()) == []
    assert not list(target.glob("*.source.json"))
    assert not list(target.glob(".txt_import_*"))


def test_xlsx_inline_sparse_headers_and_xls_copy_remain_compatible(tmp_path):
    xlsx = tmp_path / "input.xlsx"
    with zipfile.ZipFile(xlsx, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", '''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>FDBH</t></is></c><c r="C1" t="inlineStr"><is><t>SGSJ</t></is></c></row></sheetData></worksheet>''')
    xls = tmp_path / "legacy.xls"
    xls.write_bytes(b"opaque legacy bytes")
    assert inspect_table(xlsx).field_profile == "标准施工表"
    result = import_tables([xlsx, xls], tmp_path / "raw_frac")
    assert not result.rejected
    assert [path.read_bytes() for path in result.imported] == [xlsx.read_bytes(), xls.read_bytes()]


def test_imported_txt_is_readable_by_actual_timeline_loader(tmp_path):
    from App.data.fsl_timeline_loader import FSLTimelineLoader

    second = VALUES.copy()
    second[5] = "2026-09-05 10:00:01.125"
    second[6] = "66.5"
    source = write_txt(tmp_path / "source.txt", [VALUES, second])
    result = import_tables([source], tmp_path / "raw_frac", well_name="真实井", stage_id="12")
    loader = FSLTimelineLoader.__new__(FSLTimelineLoader)
    loader._stages = {}
    records = loader._read_source_records(result.imported[0], list(service.STANDARD_HEADERS))
    loader._build_stages(records)
    stage = loader.stage(loader.stage_ids()[0])
    assert stage["well_id"] == "真实井"
    assert stage["source_stage_id"] == "12"
    assert stage["sample_count"] == 2
    assert stage["duration_s"] == 1
    assert stage["pressure_mpa"] == [65.75, 66.5]
    assert stage["flow_m3_min"] == [12.25, 12.25]
    assert stage["main_label"] == "未标注"


def test_page_identity_controls_refresh_and_real_import(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QLineEdit, QPushButton, QTableWidget
    from App.ui.pages.data_import_page import build_data_import_page

    app = QApplication.instance() or QApplication([])
    source = write_txt(tmp_path / "filename-is-not-a-well.txt")
    monkeypatch.setattr(service, "PATHS", SimpleNamespace(data=tmp_path / "Data"))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args: ([str(source)], ""))
    callbacks = []
    page = build_data_import_page(SimpleNamespace(path=lambda _: tmp_path), on_imported=lambda: callbacks.append(True))
    try:
        next(button for button in page.findChildren(QPushButton) if button.text() == "选择表格").click()
        button = page.findChild(QPushButton, "importTablesButton")
        assert not button.isEnabled()
        well = page.findChild(QLineEdit, "txt_well_name_0")
        stage = page.findChild(QLineEdit, "txt_stage_id_0")
        assert well.text() == stage.text() == ""
        well.setText("界面真实井")
        stage.setText("08")
        assert button.isEnabled()
        page.set_global_dataset("raw_unrelated_file")
        page.refresh_catalog()
        assert page.property("globalDatasetId") == "raw_unrelated_file"
        assert page.findChild(QLineEdit, "txt_well_name_0").text() == "界面真实井"
        assert page.findChild(QLineEdit, "txt_stage_id_0").text() == "08"
        button.click()
        app.processEvents()
        assert callbacks == [True]
        assert "已导入 1 个文件" in page.findChild(QLabel, "notice").text()
        assert page.findChild(QTableWidget, "importFilesTable").rowCount() == 0
        csv_path, = (tmp_path / "Data" / "raw_frac").glob("*.csv")
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        assert row["JTBH"] == "界面真实井"
        assert row["FDBH"] == "08"
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()
