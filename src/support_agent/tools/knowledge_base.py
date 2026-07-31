"""`search_knowledge_base` — retrieval over past resolved tickets.

Example trigger from the problem statement: "How do I reset my password?"

The docstring on the tool function is not documentation, it is prompt: it is what
the model reads when deciding whether to call this. It has to say when the tool
is the right choice and when it is not, because a knowledge-base search on
"what plan am I on?" wastes a round trip and returns nothing useful.

Retrieval quality feeds escalation. When the best fused score falls below
`escalation_min_retrieval_score`, that is the signal that the knowledge base has
no real answer — worth handing to a human rather than paraphrasing a poor match
confidently. The score is returned in `data` so the orchestrator can act on it.
"""

from psycopg_pool import AsyncConnectionPool

from support_agent.db.repositories import knowledge_base
from support_agent.tools.base import ToolResult, ToolStatus, tool_errors_to_result


@tool_errors_to_result
async def search_knowledge_base(
    query: str,
    *,
    category: str | None = None,
    limit: int = 3,
    pool: AsyncConnectionPool,
) -> ToolResult:
    """Search past resolved support tickets for an answer to this question.

    This docstring becomes the tool description the model sees. It must
    convey: use it when the customer asks something support has likely answered
    before (how-to, troubleshooting, policy); do not use it for account-specific
    facts, which need `lookup_customer`.

    Embeds the query, runs hybrid retrieval, and returns the top matches with
    their scores. NOT_FOUND when nothing clears the relevance floor — which is a
    real answer, not a failure, and the agent should say so rather than invent one.
    """
    hits = await knowledge_base.search(
        pool,
        query,
        query_embedding=None,
        category=category,
        limit=limit,
    )
    if not hits:
        return ToolResult(
            ToolStatus.NOT_FOUND,
            "No relevant knowledge-base article was found. Do not invent an answer.",
        )
    return ToolResult(
        ToolStatus.SUCCESS,
        f"Found {len(hits)} relevant support article(s).",
        data={
            "hits": [
                {
                    "ticket_id": hit.ticket_id,
                    "answer": hit.support_agent_response,
                    "category": hit.category,
                    "resolution_type": hit.resolution_type,
                    "score": round(hit.score, 4),
                    "keyword_rank": hit.keyword_rank,
                    "semantic_rank": hit.semantic_rank,
                }
                for hit in hits
            ],
            "best_score": round(hits[0].score, 4),
        },
    )
