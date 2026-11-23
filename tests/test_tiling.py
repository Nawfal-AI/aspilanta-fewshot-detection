"""Tests for the tile grid and cross-tile NMS."""

import numpy as np
import pytest

from aspilanta.pipeline.tiling import cross_tile_nms, make_tiles


class TestMakeTiles:
    def test_covers_the_full_image(self):
        """Every pixel must fall inside at least one tile.

        A gap in coverage is a blind spot the detector can never see into, and
        it would show up only as unexplained missed detections.
        """
        width, height, tile = 1500, 1100, 640
        tiles = make_tiles(width, height, tile, overlap=0.25)
        covered = np.zeros((height, width), dtype=bool)
        for x1, y1, x2, y2 in tiles:
            covered[y1:y2, x1:x2] = True
        assert covered.all()

    def test_every_tile_is_full_size(self):
        """Partial edge tiles would be rescaled and detected at a different scale."""
        tiles = make_tiles(1500, 1100, 640, overlap=0.25)
        for x1, y1, x2, y2 in tiles:
            assert (x2 - x1, y2 - y1) == (640, 640)

    def test_image_smaller_than_tile_gives_one_tile(self):
        assert len(make_tiles(400, 300, 640, 0.25)) == 1

    def test_tiles_are_unique(self):
        tiles = make_tiles(1300, 900, 640, 0.25)
        assert len(tiles) == len(set(tiles))

    def test_more_overlap_gives_more_tiles(self):
        assert len(make_tiles(2000, 2000, 640, 0.5)) \
            > len(make_tiles(2000, 2000, 640, 0.1))


class TestCrossTileNms:
    def test_empty_input(self):
        boxes, scores, classes = cross_tile_nms(
            np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int))
        assert len(boxes) == 0

    def test_keeps_the_highest_scoring_duplicate(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        scores = np.array([0.6, 0.9])
        classes = np.array([3, 3])
        kept_boxes, kept_scores, _ = cross_tile_nms(boxes, scores, classes)
        assert len(kept_boxes) == 1
        assert kept_scores[0] == pytest.approx(0.9)

    def test_centre_distance_catches_low_iou_duplicates(self):
        """The addition plain IoU-based NMS misses.

        Adjacent tiles often each catch a different partial view of an insect
        near the seam. The two boxes genuinely have low IoU but are the same
        physical insect, and keeping both inflates the count.
        """
        boxes = np.array([[0, 0, 10, 10], [6, 0, 16, 10]], dtype=float)
        scores = np.array([0.9, 0.8])
        classes = np.array([3, 3])

        iou_only, _, _ = cross_tile_nms(boxes, scores, classes,
                                        iou_thresh=0.55, center_dist_frac=0.0)
        with_centres, _, _ = cross_tile_nms(boxes, scores, classes,
                                            iou_thresh=0.55, center_dist_frac=0.7)
        assert len(iou_only) == 2
        assert len(with_centres) == 1

    def test_different_classes_are_never_merged(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
        kept, _, _ = cross_tile_nms(boxes, np.array([0.9, 0.8]),
                                    np.array([3, 0]))
        assert len(kept) == 2

    def test_distant_boxes_both_survive(self):
        boxes = np.array([[0, 0, 10, 10], [500, 500, 510, 510]], dtype=float)
        kept, _, _ = cross_tile_nms(boxes, np.array([0.9, 0.8]), np.array([3, 3]))
        assert len(kept) == 2
