from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from labelme._automation import Detection
from labelme._automation._suppression import suppress_detections_greedy


@pytest.fixture
def make_full_detection() -> Callable[..., Detection]:
    def _make(*, label: str | None = None) -> Detection:
        return Detection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            mask=np.ones((11, 11), dtype=bool),
            label=label,
        )

    return _make


def test_redundant_drops_duplicate_mask_keeps_first(
    make_full_detection: Callable[..., Detection],
) -> None:
    first = make_full_detection()
    second = make_full_detection()

    kept = suppress_detections_greedy(detections=[first, second], iou_threshold=0.5)

    assert kept == [first]


def test_redundant_keeps_first_when_descriptions_differ() -> None:
    mask = np.ones((11, 11), dtype=bool)
    first = Detection(bbox=(0.0, 0.0, 10.0, 10.0), mask=mask, description="first")
    second = Detection(
        bbox=(0.0, 0.0, 10.0, 10.0), mask=mask.copy(), description="second"
    )

    kept = suppress_detections_greedy(detections=[first, second], iou_threshold=0.5)

    assert len(kept) == 1
    assert kept[0].description == "first"


def test_redundant_keeps_only_first_among_three_duplicates(
    make_full_detection: Callable[..., Detection],
) -> None:
    first = make_full_detection()
    second = make_full_detection()
    third = make_full_detection()

    kept = suppress_detections_greedy(
        detections=[first, second, third], iou_threshold=0.5
    )

    assert kept == [first]


def test_redundant_keeps_disjoint_masks() -> None:
    first = Detection(
        bbox=(0.0, 0.0, 5.0, 5.0),
        mask=np.ones((6, 6), dtype=bool),
    )
    second = Detection(
        bbox=(20.0, 20.0, 25.0, 25.0),
        mask=np.ones((6, 6), dtype=bool),
    )

    kept = suppress_detections_greedy(detections=[first, second], iou_threshold=0.5)

    assert kept == [first, second]


def test_redundant_drops_mask_contained_in_larger_mask() -> None:
    # Small mask is fully inside the large one — low IoU, high containment.
    # Tree-cluster vs single-tree scenario.
    large = Detection(
        bbox=(0.0, 0.0, 20.0, 20.0),
        mask=np.ones((21, 21), dtype=bool),
    )
    small = Detection(
        bbox=(5.0, 5.0, 9.0, 9.0),
        mask=np.ones((5, 5), dtype=bool),
    )

    kept = suppress_detections_greedy(detections=[large, small], iou_threshold=0.5)

    assert kept == [large]


def test_redundant_keeps_partial_overlap_below_containment_threshold() -> None:
    # big: 10x10 full mask, area 100. small: 5x5 full mask, area 25, shifted
    # one row past big's lower edge so only a 4x5 strip (20 pixels) overlaps.
    # IoU = 20/(100+25-20) = 0.19 (below 0.5, IoU check misses).
    # containment = 20/min(100,25) = 0.8 (below 0.85, containment check misses).
    big = Detection(
        bbox=(0.0, 0.0, 9.0, 9.0),
        mask=np.ones((10, 10), dtype=bool),
    )
    small = Detection(
        bbox=(5.0, 6.0, 9.0, 10.0),
        mask=np.ones((5, 5), dtype=bool),
    )

    kept = suppress_detections_greedy(detections=[big, small], iou_threshold=0.5)

    assert kept == [big, small]


def test_redundant_keeps_triangles_sharing_bbox() -> None:
    rows, cols = np.indices((11, 11))
    lower_left = Detection(
        bbox=(0.0, 0.0, 10.0, 10.0),
        mask=rows + cols <= 10,
    )
    upper_right = Detection(
        bbox=(0.0, 0.0, 10.0, 10.0),
        mask=rows + cols > 10,
    )

    kept = suppress_detections_greedy(
        detections=[lower_left, upper_right], iou_threshold=0.5
    )

    assert kept == [lower_left, upper_right]


def test_redundant_does_not_suppress_across_labels(
    make_full_detection: Callable[..., Detection],
) -> None:
    tree = make_full_detection(label="tree")
    grass = make_full_detection(label="grass")

    kept = suppress_detections_greedy(detections=[tree, grass], iou_threshold=0.5)

    assert kept == [tree, grass]


def test_redundant_empty_detections() -> None:
    assert suppress_detections_greedy(detections=[], iou_threshold=0.5) == []


def test_redundant_drops_duplicate_bbox_only_detections() -> None:
    first = Detection(bbox=(0.0, 0.0, 10.0, 10.0))
    second = Detection(bbox=(0.0, 0.0, 10.0, 10.0))

    kept = suppress_detections_greedy(detections=[first, second], iou_threshold=0.5)

    assert kept == [first]


def test_redundant_passes_through_detections_without_bbox() -> None:
    no_bbox = Detection(mask=np.ones((11, 11), dtype=bool))

    kept = suppress_detections_greedy(detections=[no_bbox, no_bbox], iou_threshold=0.5)

    assert kept == [no_bbox, no_bbox]


def test_redundant_raises_on_mask_shape_mismatch() -> None:
    bad = Detection(
        bbox=(0.0, 0.0, 9.0, 9.0),
        mask=np.ones((11, 11), dtype=bool),
    )

    with pytest.raises(ValueError, match="mask shape"):
        suppress_detections_greedy(detections=[bad], iou_threshold=0.5)


def test_redundant_rejects_mixed_mask_and_bbox_only() -> None:
    with_mask = Detection(
        bbox=(0.0, 0.0, 10.0, 10.0),
        mask=np.ones((11, 11), dtype=bool),
    )
    bbox_only = Detection(bbox=(0.0, 0.0, 10.0, 10.0))

    with pytest.raises(ValueError, match="homogeneous"):
        suppress_detections_greedy(detections=[with_mask, bbox_only], iou_threshold=0.5)
