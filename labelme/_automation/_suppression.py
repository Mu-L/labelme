from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ._shape_builders import Detection


# eq=False: numpy arrays don't reduce to a scalar bool, so the auto-generated
# dataclass __eq__ (which calls bool() on the result) would raise ValueError.
@dataclass(frozen=True, eq=False)
class _LocalMask:
    mask: NDArray[np.bool_]
    origin_xy: tuple[int, int]
    area: int


def suppress_detections_greedy(
    *,
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    """Callers must pass detections in priority order (highest first); the first
    detection in each redundant cluster is kept and later ones are dropped.

    Redundancy combines IoU with intersection-over-smaller (containment), so
    nested masks (e.g. tree-cluster containing a single tree) deduplicate even
    when their IoU is low.
    """
    if not detections:
        return []

    # Mixing bbox-only and mask detections would silently let bbox-only peers
    # (treated as fully-opaque rectangles) suppress overlapping mask detections
    # via containment. Require homogeneous input until a caller needs otherwise.
    mask_presence = {d.mask is not None for d in detections if d.bbox is not None}
    if len(mask_presence) > 1:
        raise ValueError(
            "detections must be homogeneous: either all have masks or none do"
        )

    kept: list[Detection] = []
    kept_masks_by_label: dict[str | None, list[_LocalMask]] = {}
    for detection in detections:
        if detection.bbox is None:
            kept.append(detection)
            continue
        new_local = _make_local_mask(detection=detection)
        peers = kept_masks_by_label.setdefault(detection.label, [])
        if any(
            _is_redundant_pair(
                new=new_local,
                peer=peer,
                iou_threshold=iou_threshold,
            )
            for peer in peers
        ):
            continue
        kept.append(detection)
        peers.append(new_local)
    return kept


def _is_redundant_pair(
    *,
    new: _LocalMask,
    peer: _LocalMask,
    iou_threshold: float,
) -> bool:
    # Containment (intersection-over-smaller) catches nested masks whose IoU
    # is too low for the IoU check (e.g. tree-cluster swallowing a single tree).
    CONTAINMENT_THRESHOLD: Final[float] = 0.85

    intersection = _compute_mask_intersection_area(a=new, b=peer)
    if intersection == 0:
        return False
    iou = intersection / (new.area + peer.area - intersection)
    if iou >= iou_threshold:
        return True
    containment = intersection / min(new.area, peer.area)
    return containment >= CONTAINMENT_THRESHOLD


def _make_local_mask(*, detection: Detection) -> _LocalMask:
    xmin, ymin, xmax, ymax = np.array(detection.bbox).round().astype(int).tolist()
    if detection.mask is None:
        h, w = ymax - ymin + 1, xmax - xmin + 1
        mask = np.ones((h, w), dtype=np.bool_)
        return _LocalMask(mask=mask, origin_xy=(xmin, ymin), area=h * w)
    # Mask geometry below assumes mask covers exactly the bbox-derived extent
    # (matching the OSAM Annotation contract). Reject inconsistent shapes loudly
    # so a future non-OSAM caller doesn't silently produce wrong IoU values.
    expected_shape = (ymax - ymin + 1, xmax - xmin + 1)
    if detection.mask.shape != expected_shape:
        raise ValueError(
            f"mask shape {detection.mask.shape} does not match "
            f"bbox-derived extent {expected_shape}"
        )
    return _LocalMask(
        mask=detection.mask,
        origin_xy=(xmin, ymin),
        area=int(np.count_nonzero(detection.mask)),
    )


def _compute_mask_intersection_area(*, a: _LocalMask, b: _LocalMask) -> int:
    # bbox endpoints are inclusive pixel coords (mask width = xmax - xmin + 1),
    # so xmin + w is the exclusive x-upper-bound used for clipping.
    a_xmin, a_ymin = a.origin_xy
    b_xmin, b_ymin = b.origin_xy
    a_h, a_w = a.mask.shape
    b_h, b_w = b.mask.shape

    inter_xmin = max(a_xmin, b_xmin)
    inter_ymin = max(a_ymin, b_ymin)
    inter_xmax = min(a_xmin + a_w, b_xmin + b_w)
    inter_ymax = min(a_ymin + a_h, b_ymin + b_h)
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0

    sub_a = a.mask[
        inter_ymin - a_ymin : inter_ymax - a_ymin,
        inter_xmin - a_xmin : inter_xmax - a_xmin,
    ]
    sub_b = b.mask[
        inter_ymin - b_ymin : inter_ymax - b_ymin,
        inter_xmin - b_xmin : inter_xmax - b_xmin,
    ]
    return int(np.count_nonzero(sub_a & sub_b))
