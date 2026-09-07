# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""CPU ownership plan shared by compact placement and SM-local static execution."""

from dataclasses import dataclass

import torch


@dataclass
class OwnerSchedule:
    task_work: torch.Tensor
    task_q_subtile: torch.Tensor
    owner_indptr: torch.Tensor
    page_owners: torch.Tensor
    task_owners: torch.Tensor
    task_cost: torch.Tensor
    owner_cost: list[int]
    range_owners: list[dict]


def build_owner_schedule(
    plan, int_workspace, kv_indices, num_pages, num_heads, page_size, sm_counts
):
    """Keep the original split/output ABI; only regroup the CTA tasks.

    First version: noncausal caller, identity page table, page-aligned KV ranges.
    Repeated Q tiles of a KV range share one owner. Weighted greedy placement
    balances estimated attention work per SM, rather than the number of splits.
    """
    if num_heads != 128 or page_size != 64:
        raise ValueError("First-stage schedule requires 128 heads and 64-token pages")
    indices = kv_indices.cpu()
    if not torch.equal(indices, torch.arange(num_pages, dtype=torch.int32)):
        raise ValueError("First-stage schedule requires an identity page table")
    counts = list(map(int, sm_counts))
    if len(counts) != 2 or min(counts) <= 0:
        raise ValueError("Both partitions must have SMs")
    p = list(map(int, plan))
    if p[0] * p[1] != sum(counts):
        raise ValueError("Plan grid and runtime SM count differ")
    host = int_workspace.cpu()

    def read(offset, count):
        return host[offset : offset + 4 * count].view(torch.int32).tolist()

    work_count = read(p[15], p[1] + 1)[-1]
    kv_ptr = read(p[3], work_count)
    kv_start, kv_end = read(p[13], work_count), read(p[14], work_count)
    groups = {}
    for work in range(work_count):
        start, end = kv_start[work], kv_end[work]
        if start % page_size or end % page_size or end <= start:
            raise ValueError(
                "First-stage schedule requires nonempty page-aligned KV ranges"
            )
        key = (kv_ptr[work], start, end)
        groups.setdefault(key, []).extend((work, x) for x in range(p[0]))
    page_owners = torch.full((num_pages,), -1, dtype=torch.int32)
    costs = [0, 0]
    queues = [[], []]
    ranges = []
    weighted = sorted(
        groups.items(), key=lambda item: -len(item[1]) * (item[0][2] - item[0][1])
    )
    for (ptr, start, end), tasks in weighted:
        cost = len(tasks) * (end - start)
        owner = min(range(2), key=lambda o: (costs[o] / counts[o], o))
        costs[owner] += cost
        begin_page, end_page = ptr + start // page_size, ptr + end // page_size
        if begin_page < 0 or end_page > num_pages:
            raise ValueError("Planner range exceeds KV page table")
        previous = page_owners[begin_page:end_page]
        if ((previous != -1) & (previous != owner)).any():
            raise ValueError("Overlapping KV ranges received conflicting owners")
        page_owners[begin_page:end_page] = owner
        queues[owner].extend((work, x, end - start) for work, x in tasks)
        ranges.append(
            dict(
                kv_indptr=ptr,
                kv_start=start,
                kv_end=end,
                owner=owner,
                cta_tasks=len(tasks),
                cost=cost,
            )
        )
    if (page_owners < 0).any():
        raise ValueError("Some physical KV pages are not covered by the schedule")
    records = queues[0] + queues[1]
    if len(set((w, x) for w, x, _ in records)) != work_count * p[0]:
        raise RuntimeError("CTA task enumeration lost or duplicated work")
    return OwnerSchedule(
        task_work=torch.tensor([r[0] for r in records], dtype=torch.int32),
        task_q_subtile=torch.tensor([r[1] for r in records], dtype=torch.int32),
        owner_indptr=torch.tensor([0, len(queues[0]), len(records)], dtype=torch.int32),
        page_owners=page_owners,
        task_owners=torch.tensor(
            [0] * len(queues[0]) + [1] * len(queues[1]), dtype=torch.int32
        ),
        task_cost=torch.tensor([r[2] for r in records], dtype=torch.int64),
        owner_cost=costs,
        range_owners=ranges,
    )
