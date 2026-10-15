"""Specimen photo grouping."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .config import GroupingConfig
from .descriptors import color_affinity, sift_match, texture_affinity
from .photo import PhotoRecord


@dataclass
class Member:
    rec: PhotoRecord
    confidence: float
    margin: float | None
    how: str                                # Membership source: anchor, assigned, or pile_link.
    flags: list[str] = field(default_factory=list)


@dataclass
class Group:
    kind: str                               # Group type: single or collection.
    members: list[Member]
    flags: list[str] = field(default_factory=list)
    id: str = ""


def _seq_aff(a: PhotoRecord, b: PhotoRecord, cfg: GroupingConfig) -> float | None:
    if a.meta.capture_time and b.meta.capture_time:
        dt = abs((a.meta.capture_time - b.meta.capture_time).total_seconds())
        return float(np.exp(-dt / 180.0))
    if a.meta.seq_number is not None and b.meta.seq_number is not None:
        return float(np.exp(-abs(a.meta.seq_number - b.meta.seq_number) / cfg.sequence_window))
    return None


def _weights_conflict(a: PhotoRecord, b: PhotoRecord) -> bool:
    if not a.balance or not b.balance:
        return False
    if min(a.balance.confidence, b.balance.confidence) < 0.65:
        return False
    x, y = a.balance.weight_g, b.balance.weight_g
    return x is not None and y is not None and abs(x - y) > max(1.0, 0.015 * max(x, y))


def pair_affinity(a: PhotoRecord, b: PhotoRecord, cfg: GroupingConfig, cache: dict) -> tuple[float, dict]:
    key = tuple(sorted((str(a.path), str(b.path))))
    if key in cache:
        return cache[key]
    parts: dict[str, float] = {}
    if _weights_conflict(a, b):
        res = (0.0, {"veto": "confirmed weights differ"})
        cache[key] = res
        return res
    da, db = a.descriptor, b.descriptor
    if da is None or db is None:
        res = (0.0, {"no_descriptor": 1.0})
        cache[key] = res
        return res
    weak = 0.5 if (da.source == "scene" or db.source == "scene") else 1.0     # Reduce scene-descriptor weights.
    num, den = 0.0, 0.0
    c = color_affinity(da, db); parts["color"] = c
    num += cfg.w_color * weak * c; den += cfg.w_color * weak
    t = texture_affinity(da, db); parts["texture"] = t
    num += cfg.w_texture * weak * t; den += cfg.w_texture * weak
    s = _seq_aff(a, b, cfg)
    if s is not None:
        parts["sequence"] = s
        num += cfg.w_sequence * s; den += cfg.w_sequence
    if da.embedding is not None and db.embedding is not None:
        e = float(np.clip((da.embedding @ db.embedding - 0.3) / 0.6, 0, 1)); parts["embed"] = e
        num += cfg.w_embed * e; den += cfg.w_embed
    base = num / den if den else 0.0
    inl, ratio = sift_match(da, db, cfg.sift_min_inliers)
    sift = float(1 - np.exp(-cfg.w_sift / 0.40 * inl / 25.0)) if inl >= cfg.sift_min_inliers else 0.0
    parts["sift_inliers"] = float(inl)
    parts["sift"] = sift
    parts["base"] = base
    score = float(1 - (1 - base) * (1 - sift))
    if weak < 1:
        score *= 0.50
        parts["scene_features_capped"] = 1.0
    res = (score, parts)
    cache[key] = res
    return res


def group_photos(recs: list[PhotoRecord], cfg: GroupingConfig) -> tuple[list[Group], list[PhotoRecord]]:
    recs = sorted(recs, key=lambda r: (r.path.name.casefold(), str(r.path)))
    cache: dict = {}
    piles = [r for r in recs if r.scene == "pile"]
    rest = [r for r in recs if r.scene != "pile"]
    groups: list[Group] = []

    # Link collection photos by shared local features.
    if piles:
        idx = list(range(len(piles)))
        parent = list(idx)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        links: dict[int, list[float]] = {i: [] for i in idx}
        for i in idx:
            for j in idx[i + 1:]:
                da, db = piles[i].descriptor, piles[j].descriptor
                inl, _ = sift_match(da, db, cfg.sift_min_inliers) if da and db else (0, 0.0)
                s = _seq_aff(piles[i], piles[j], cfg) or 0.0
                close = bool(da and db and s > 0.8 and color_affinity(da, db) > 0.8)
                if inl >= cfg.sift_min_inliers or close:
                    parent[find(i)] = find(j)
                    v = 1 - np.exp(-inl / 25.0) if inl >= cfg.sift_min_inliers else 0.5
                    links[i].append(v); links[j].append(v)
        clusters: dict[int, list[int]] = {}
        for i in idx:
            clusters.setdefault(find(i), []).append(i)
        for members in clusters.values():
            ms = []
            for i in members:
                conf = float(np.mean(links[i])) if links[i] else 0.35
                fl = [] if links[i] else ["no_feature_link_to_other_pile_photos"]
                ms.append(Member(piles[i], conf, None, "pile_link", fl))
            groups.append(Group("collection", ms, ["per_piece_measurement_not_implemented"]))

    # Cluster measured-photo anchors.
    anchors = [r for r in rest if r.measured]
    deps = [r for r in rest if not r.measured]
    singles: list[list[PhotoRecord]] = []
    if len(anchors) == 1:
        singles = [anchors]
    elif len(anchors) > 1:
        n = len(anchors)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                s, _ = pair_affinity(anchors[i], anchors[j], cfg, cache)
                D[i, j] = D[j, i] = 1.0 - s
        lab = fcluster(linkage(squareform(D, checks=False), "complete"), t=1.0 - cfg.cluster_threshold, criterion="distance")
        for k in sorted(set(lab)):
            singles.append([anchors[i] for i in range(n) if lab[i] == k])
    sgroups = [Group("single", [Member(r, 1.0, None, "anchor") for r in cl]) for cl in singles]
    anchor_members = [tuple(g.members) for g in sgroups]

    leftovers: list[PhotoRecord] = []
    for r in deps:
        if not sgroups:
            leftovers.append(r)
            continue
        scores = []
        for g, anchors_in_group in zip(sgroups, anchor_members):
            if any(_weights_conflict(r, m.rec) for m in g.members):
                scores.append(0.0)
                continue
            aff = sorted([pair_affinity(r, m.rec, cfg, cache)[0] for m in anchors_in_group], reverse=True)
            scores.append(float(np.mean(aff[:2])))
        order = np.argsort(scores)[::-1]
        best = scores[order[0]]
        second = scores[order[1]] if len(scores) > 1 else 0.0
        margin = best - second
        if best <= 0 or best < cfg.assign_min_affinity:
            leftovers.append(r)
            continue
        fl = []
        if margin < cfg.assign_min_margin:
            fl.append("ambiguous_between_specimens")
        if best < 0.55:
            fl.append("low_affinity")
        sgroups[order[0]].members.append(Member(r, best, float(margin), "assigned", fl))

    # Score each anchor by its strongest group link.
    for g in sgroups:
        for m in g.members:
            if m.how == "anchor" and len(g.members) > 1:
                m.confidence = float(max(pair_affinity(m.rec, o.rec, cfg, cache)[0] for o in g.members if o is not m))
                if m.confidence < 0.45:
                    m.flags.append("weak_agreement_with_group")

    used: set[int] = set()
    for i, r in enumerate(leftovers):
        if i in used:
            continue
        mates = [(j, pair_affinity(r, leftovers[j], cfg, cache)[0]) for j in range(i + 1, len(leftovers)) if j not in used]
        mates = [(j, s) for j, s in mates if s >= cfg.cluster_threshold]
        compatible = []
        for j, s in mates:
            if all(not _weights_conflict(leftovers[j], leftovers[k]) and
                   pair_affinity(leftovers[j], leftovers[k], cfg, cache)[0] >= cfg.cluster_threshold
                   for k, _ in compatible):
                compatible.append((j, s))
        mates = compatible
        if mates:
            g = Group("single", [Member(r, float(np.mean([s for _, s in mates])), None, "assigned", ["no_measured_photo_in_group"])],
                      ["no_ruler_measurement"])
            for j, s in mates:
                used.add(j)
                g.members.append(Member(leftovers[j], s, None, "assigned", ["no_measured_photo_in_group"]))
            used.add(i)
            sgroups.append(g)
    unassigned = [r for i, r in enumerate(leftovers) if i not in used]
    groups.extend(sgroups)
    return groups, unassigned
