from collections import defaultdict, deque


def topological_sort(dependencies: dict[str, list[str] | None]) -> list[str]:
    """
    Simple topological sort to return a list of nodes in topological order.
    """
    out_edges: dict[str, list[str]] = defaultdict(list)
    in_degrees: dict[str, int] = defaultdict(int)

    for node, deps in dependencies.items():
        in_degrees.setdefault(node, 0)
        if deps is None:
            continue
        for dep in deps:
            out_edges[dep].append(node)
            in_degrees[node] += 1
            in_degrees.setdefault(dep, 0)

    queue: deque[str] = deque([n for n, d in in_degrees.items() if d == 0])
    sorted_nodes = []

    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for neighbor in out_edges[node]:
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_nodes) != len(dependencies):
        raise ValueError(f"Cycle detected in dependencies: {dependencies}")

    return sorted_nodes


def topological_levels(dependencies: dict[str, list[str] | None]) -> list[list[str]]:
    """
    topological sort of a DAG of dependencies to return a list of levels,
    such that every node's dependencies live in earlier levels.
    """
    deps: dict[str, list[str]] = {k: (v or []) for k, v in dependencies.items()}
    out_edges: dict[str, list[str]] = defaultdict(list)
    in_degrees: dict[str, int] = defaultdict(int)

    for n, pres in deps.items():
        in_degrees.setdefault(n, 0)
        for p in pres:
            out_edges[p].append(n)
            in_degrees[n] += 1
            in_degrees.setdefault(p, 0)

    queue: deque[str] = deque([n for n, d in in_degrees.items() if d == 0])
    levels: list[list[str]] = []

    while queue:
        this_lvl = list(queue)
        levels.append(this_lvl)
        for _ in range(len(this_lvl)):
            n = queue.popleft()
            for m in out_edges[n]:
                in_degrees[m] -= 1
                if in_degrees[m] == 0:
                    queue.append(m)

    if sum(len(level) for level in levels) != len(in_degrees):
        raise ValueError("Cycle detected in dependencies; levelization impossible")
    return levels
