"""Exhaustive distinct-score policy accelerated by a Fenwick active-row index."""
import numpy as np
from numba import njit


@njit
def _prefix(tree, pos):
    total = 0
    while pos:
        total += tree[pos]
        pos -= pos & -pos
    return total


@njit
def _kth(tree, k):
    pos = 0
    bit = 1
    while bit * 2 < len(tree):
        bit *= 2
    while bit:
        nxt = pos + bit
        if nxt < len(tree) and tree[nxt] < k:
            pos = nxt
            k -= tree[nxt]
        bit //= 2
    return pos


@njit
def _add(tree, index, delta):
    pos = index + 1
    while pos < len(tree):
        tree[pos] += delta
        pos += pos & -pos


@njit
def exhaustive_counts(scores, times, coin_ends, targets, horizon_ns):
    """All thresholds; recompute only displaced cooldown alarms until convergence."""
    n = len(scores)
    order = np.argsort(scores)[::-1]
    active = np.zeros(n + 1, np.int64)
    selected = np.zeros(n + 1, np.int64)
    is_alarm = np.zeros(n, np.bool_)
    target_counts = np.zeros(n + 1, np.int64)
    begins = np.empty(n, np.int64)
    ends = np.empty(n, np.int64)
    begin = 0
    for end in coin_ends:
        begins[begin:end] = begin
        ends[begin:end] = end
        begin = end
    cutoffs = np.empty(len(scores), np.float64)
    caught = np.empty(len(scores), np.int64)
    false = np.empty(len(scores), np.int64)
    alarms = np.empty(len(scores), np.int64)
    cursor = 0
    epoch = 0
    nc, nf, na = 0, 0, 0
    while cursor < len(order):
        threshold = scores[order[cursor]]
        while cursor < len(order) and scores[order[cursor]] == threshold:
            pos = order[cursor]
            _add(active, pos, 1)
            cursor += 1
            prior_rank = _prefix(selected, pos)
            if prior_rank > _prefix(selected, begins[pos]):
                prior = _kth(selected, prior_rank)
                if times[pos] <= times[prior] + horizon_ns:
                    continue
            end = ends[pos]
            # Insert the newly enabled unsuppressed flag; only its suffix can change.
            while pos < end and not is_alarm[pos]:
                is_alarm[pos] = True
                _add(selected, pos, 1)
                na += 1
                target = targets[pos]
                if target < 0:
                    nf += 1
                else:
                    if target_counts[target] == 0:
                        nc += 1
                    target_counts[target] += 1
                lo, hi = pos + 1, end
                bound = times[pos] + horizon_ns
                while lo < hi:
                    mid = (lo + hi) // 2
                    if times[mid] <= bound:
                        lo = mid + 1
                    else:
                        hi = mid
                # Remove previously selected alarms now suppressed by this origin.
                prior_rank = _prefix(selected, pos + 1)
                limit = _prefix(selected, lo)
                while prior_rank < limit:
                    old = _kth(selected, prior_rank + 1)
                    is_alarm[old] = False
                    _add(selected, old, -1)
                    na -= 1
                    target = targets[old]
                    if target < 0:
                        nf -= 1
                    else:
                        target_counts[target] -= 1
                        if target_counts[target] == 0:
                            nc -= 1
                    limit -= 1
                rank = _prefix(active, lo)
                if rank == _prefix(active, end):
                    break
                pos = _kth(active, rank + 1)
                # An existing alarm at the new successor is a fixed point:
                # the remainder of that coin's greedy sequence is unchanged.
        epoch += 1
        cutoffs[epoch - 1] = threshold
        caught[epoch - 1] = nc
        false[epoch - 1] = nf
        alarms[epoch - 1] = na
    return cutoffs[:epoch], caught[:epoch], false[:epoch], alarms[:epoch]
