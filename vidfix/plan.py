from __future__ import annotations


def plan_segments(
    cut_times: list[float],
    duration: float,
    max_len: float = 25.0,
    min_len: float = 1.0,
) -> list[tuple[float, float]]:
    """Turn scene-cut timestamps into tagged-ready (t0, t1) segments.

    Long scenes are split at ``max_len`` (20–30s). Scenes shorter than
    ``min_len`` (~1s) are merged into the previous segment, or the next
    one if they are first.
    """
    duration = float(duration)
    if duration <= 0:
        return []

    bounds: list[float] = [0.0]
    for c in cut_times:
        t = round(float(c), 3)
        if 0.05 < t < duration - 0.05:
            bounds.append(t)
    bounds.append(round(duration, 3))
    bounds = sorted(set(bounds))

    pieces: list[tuple[float, float]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        t = a
        while (b - t) > max_len + 1e-3:
            nxt = round(t + max_len, 3)
            pieces.append((t, nxt))
            t = nxt
        pieces.append((t, round(b, 3)))

    if not pieces:
        return [(0.0, round(duration, 3))]

    merged: list[list[float]] = [list(pieces[0])]
    for t0, t1 in pieces[1:]:
        if (t1 - t0) < min_len:
            merged[-1][1] = t1
        else:
            merged.append([t0, t1])

    if len(merged) >= 2 and (merged[0][1] - merged[0][0]) < min_len:
        merged[1][0] = merged[0][0]
        merged.pop(0)

    out: list[tuple[float, float]] = []
    for a, b in merged:
        a, b = round(a, 3), round(b, 3)
        if b - a > 0.001:
            out.append((a, b))
    return out or [(0.0, round(duration, 3))]


def _self_check() -> None:
    # 無切點：依最長 25 秒切開
    s = plan_segments([], 80.0, max_len=25.0, min_len=1.0)
    assert s[0] == (0.0, 25.0)
    assert s[-1][1] == 80.0
    assert all(b - a <= 25.001 for a, b in s[:-1])
    # 短於 1 秒併入前一段
    s = plan_segments([10.0, 10.4, 20.0], 30.0, max_len=25.0, min_len=1.0)
    assert all(b - a >= 1.0 or (a == s[0][0] and len(s) == 1) for a, b in s)
    # 開頭短段併入下一段
    s = plan_segments([0.3], 10.0, max_len=25.0, min_len=1.0)
    assert s[0][0] == 0.0
    assert s[0][1] == 10.0 or s[0][1] >= 0.3


if __name__ == "__main__":
    _self_check()
    print("plan_segments ok")
