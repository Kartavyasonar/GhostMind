"""GraphRAG: citation graph built with NetworkX."""
from typing import List, Dict, Optional, Set
import networkx as nx
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.models import Paper, Citation

log = structlog.get_logger()

_graph: Optional[nx.DiGraph] = None


async def build_graph(db: AsyncSession) -> nx.DiGraph:
    global _graph
    G = nx.DiGraph()

    papers = (await db.execute(select(Paper))).scalars().all()
    citations = (await db.execute(select(Citation))).scalars().all()

    for p in papers:
        G.add_node(p.arxiv_id, title=p.title, abstract=p.abstract[:200])

    for c in citations:
        G.add_edge(c.paper_id, c.cites_arxiv_id, weight=c.weight)

    _graph = G
    log.info("Citation graph built", nodes=G.number_of_nodes(), edges=G.number_of_edges())
    return G


def get_graph() -> nx.DiGraph:
    if _graph is None:
        return nx.DiGraph()
    return _graph


def graph_expand(seed_ids: List[str], hops: int = 2, max_nodes: int = 20) -> List[str]:
    G = get_graph()
    if G.number_of_nodes() == 0:
        return seed_ids

    visited: Set[str] = set(seed_ids)
    frontier = set(seed_ids)

    for _ in range(hops):
        next_frontier: Set[str] = set()
        for node in frontier:
            if node in G:
                for neighbor in list(G.successors(node)) + list(G.predecessors(node)):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
        frontier = next_frontier
        if len(visited) >= max_nodes:
            break

    return list(visited)[:max_nodes]


def compute_graph_stats() -> Dict:
    G = get_graph()
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "is_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
        "avg_degree": (
            sum(d for _, d in G.degree()) / G.number_of_nodes()
            if G.number_of_nodes() > 0 else 0.0
        ),
    }
