from ml.detect._contract_fallback import LABELS
from ml.detect.labels import (COCO_TO_LABEL, build_index_map, is_proxy, map_label, normalize_name,
                              source_tag)


def test_all_mapped_labels_in_contract():
    assert set(COCO_TO_LABEL.values()) <= set(LABELS)


def test_coco_mapping_from_task():
    expect = {"person": "person", "cell phone": "cell_phone", "cup": "cup", "wine glass": "cup", "bowl": "cup",
              "bottle": "bottle", "banana": "food", "apple": "food", "sandwich": "food", "orange": "food",
              "donut": "food", "cake": "food", "pizza": "food", "laptop": "laptop", "handbag": "bag",
              "backpack": "bag", "suitcase": "bag", "book": "document"}
    for k, v in expect.items():
        assert map_label(k) == v, k


def test_unmapped_and_variants():
    assert map_label("car") is None
    assert map_label("tv") is None
    assert map_label("Cell_Phone") == "cell_phone"
    assert map_label("cash_bag") == "cash_bag"
    assert normalize_name("  Cell-Phone ") == "cell phone"


def test_bag_is_not_cash_bag():
    # COCO handbag/backpack не доказывает «инкассаторская сумка»
    for name in ("handbag", "backpack", "suitcase"):
        assert map_label(name) == "bag"
        assert map_label(name) != "cash_bag"


def test_proxy_semantics_kept_in_source():
    assert is_proxy("book") and is_proxy("bowl")
    assert not is_proxy("person") and not is_proxy("handbag")
    assert source_tag("rtdetr_v2_r18vd", "book") == "rtdetr_v2_r18vd:book~proxy"
    assert source_tag("rtdetr_v2_r18vd", "cell phone") == "rtdetr_v2_r18vd:cell phone"


def test_build_index_map_with_and_without_proxy():
    id2label = {0: "person", 1: "bicycle", 67: "cell phone", 73: "book", 45: "bowl"}
    m = build_index_map(id2label)
    assert m == {0: "person", 67: "cell_phone", 73: "document", 45: "cup"}
    m2 = build_index_map(id2label, allow_proxy=False)
    assert m2 == {0: "person", 67: "cell_phone"}
    assert build_index_map(["cash", "cash_bag", "zebra"]) == {0: "cash", 1: "cash_bag"}
