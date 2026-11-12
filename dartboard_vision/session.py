"""Continuous detection and round state for already rectified camera frames.

The empty reference and calibration are retained across throws and clear-board
resets. Capture is the caller's responsibility. Scores require an explicitly
supplied scoring function with verified board orientation.
"""

from copy import deepcopy

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contacts import intersections, shaft_lines
from .geometry import BOARD
from .masks import dart_mask, evidence_score


def associate(points, observations, radius):
    if not len(points) or not len(observations):
        return []
    distances = np.linalg.norm(
        np.array(points)[:, None] - np.array(observations)[None], axis=2
    )
    # Prefer a feasible match to any infeasible edge, then minimize distance.
    cost = np.where(distances <= radius, distances, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if distances[i, j] <= radius]


class RoundState:
    def __init__(
        self,
        match_radius_px=12.0,
        confirm_frames=2,
        clear_frames=3,
        clear_fraction=0.001,
        score_point=None,
    ):
        self.match_radius = match_radius_px
        self.confirm_frames = confirm_frames
        self.clear_frames = clear_frames
        self.clear_fraction = clear_fraction
        self.score_point = score_point
        self.hits = []
        self.pending = []
        self.history = []
        self.round_id = 1
        self.clear_streak = 0

    def snapshot(self, events=()):
        total = (
            0
            if not self.hits
            else (
                sum(h["score"]["score"] for h in self.hits)
                if self.score_point
                else None
            )
        )
        return deepcopy(
            dict(
                round_id=self.round_id,
                hits=self.hits,
                pending_count=len(self.pending),
                round_total=total,
                score_verified=self.score_point is not None,
                events=list(events),
                completed_rounds=len(self.history),
            )
        )

    def observe(self, contacts, foreground_fractions, t_s, alignment_valid=True):
        events = []
        if not alignment_valid:
            self.clear_streak = 0
            self.pending = []
            return self.snapshot([dict(type="alignment_hold", t_s=t_s)])
        clear = (
            not contacts
            and len(foreground_fractions) >= 2
            and all(f <= self.clear_fraction for f in foreground_fractions)
        )
        self.clear_streak = self.clear_streak + 1 if clear else 0
        if self.clear_streak >= self.clear_frames:
            if self.hits:
                archived = self.snapshot()
                archived["cleared_t_s"] = t_s
                self.history.append(archived)
                self.hits = []
                self.round_id += 1
                events.append(dict(type="round_reset", t_s=t_s))
            self.pending = []
            return self.snapshot(events)
        points = [c["point_pixels"] for c in contacts]
        matched = associate(
            [h["point_pixels"] for h in self.hits], points, self.match_radius
        )
        used = {j for _, j in matched}
        for i, j in matched:
            self.hits[i].update(
                point_pixels=list(points[j]),
                last_seen_t_s=t_s,
                support=contacts[j].get("support"),
            )
            if self.score_point:
                self.hits[i]["score"] = self.score_point(points[j])
        remaining = [j for j in range(len(contacts)) if j not in used]
        pmatches = associate(
            [p["point_pixels"] for p in self.pending],
            [points[j] for j in remaining],
            self.match_radius,
        )
        next_pending = []
        paired = set()
        for i, k in pmatches:
            j = remaining[k]
            p = dict(self.pending[i])
            p.update(point_pixels=list(points[j]), streak=p["streak"] + 1)
            paired.add(j)
            if p["streak"] >= self.confirm_frames:
                hit = dict(
                    id=len(self.hits) + 1,
                    point_pixels=list(points[j]),
                    first_seen_t_s=p["first_seen_t_s"],
                    confirmed_t_s=t_s,
                    last_seen_t_s=t_s,
                    support=contacts[j].get("support"),
                    score=self.score_point(points[j]) if self.score_point else None,
                )
                self.hits.append(hit)
                events.append(dict(type="new_hit", id=hit["id"], t_s=t_s))
            else:
                next_pending.append(p)
        for j in remaining:
            if j not in paired:
                next_pending.append(
                    dict(point_pixels=list(points[j]), streak=1, first_seen_t_s=t_s)
                )
        self.pending = next_pending
        # Missing detections never remove a stored hit. Only confirmed image
        # evidence of an empty board clears the round.
        return self.snapshot(events)


class DartboardSession:
    def __init__(
        self, empty_images, empty_valid, settings, board=BOARD, score_point=None
    ):
        if len(empty_images) < 2 or not len(empty_images) == len(empty_valid) == len(
            settings
        ):
            raise ValueError(
                "Need matching references, visibility and settings for at least two cameras"
            )
        self.empty = [im.copy() for im in empty_images]
        self.valid = [np.asarray(v, bool).copy() for v in empty_valid]
        self.settings = list(settings)
        self.board = board
        y, x = np.mgrid[: board.size, : board.size]
        self.roi = (
            np.hypot(x - board.center[0], y - board.center[1])
            <= board.radii[-1] * board.ppm
        )
        self.state = RoundState(match_radius_px=6 * board.ppm, score_point=score_point)

    def process(self, images, valid, t_s, alignment_valid=True):
        if not len(images) == len(self.empty) == len(valid):
            raise ValueError("Camera count must match the empty reference")
        masks = []
        axes = []
        fractions = []
        display = []
        for old, new, seen, initial, s in zip(
            self.empty, images, valid, self.valid, self.settings
        ):
            support = np.asarray(seen, bool) & initial & self.roi
            if support.sum() < 0.5 * self.roi.sum():
                alignment_valid = False
            mask, info = dart_mask(old, new, support, s)
            masks.append(mask)
            strength = evidence_score(info["fields"], s)
            display.append(strength * np.float32(mask > 0))
            axes.append(shaft_lines(mask, strength)[0])
            fractions.append(float(np.count_nonzero(mask) / max(1, support.sum())))
        self.display_strength = np.maximum.reduce(display)
        contacts, _ = intersections(
            axes, self.board.center, self.board.radii[-1] * self.board.ppm
        )
        current = self.state.observe(contacts, fractions, t_s, alignment_valid)
        return dict(
            t_s=t_s,
            contacts=contacts,
            foreground_fractions=fractions,
            alignment_valid=alignment_valid,
            round=current,
        ), masks
