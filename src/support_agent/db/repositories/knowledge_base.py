"""Knowledge-base retrieval over the 25 resolved support tickets.

Hybrid search: Postgres full-text for keyword matching, pgvector cosine distance
for semantic matching, fused with Reciprocal Rank Fusion.

Both halves earn their place on this dataset. "403 forbidden error" is a keyword
match — the literal token appears in TK-014. "I can't get into my account" is a
semantic one — it shares no content words with the article that answers it.
Running only one of the two loses a class of question.

RRF is used rather than a weighted score blend because tsvector ranks and cosine
distances are not on comparable scales; fusing by *rank* sidesteps having to
normalise them against each other.

Degradation: if `embedding` is NULL (the seed script has not run, or the
embeddings call failed), search falls back to keyword-only rather than erroring.
"""

import re
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool


@dataclass(frozen=True, slots=True)
class KBHit:
    """One retrieved article plus the scores that put it there.

    The per-signal scores are kept, not just the fused one, because the agent's
    escalation policy needs to know *how weak* the best match was, and because a
    reviewer debugging a bad retrieval wants to see which half of the hybrid
    fired.
    """

    ticket_id: str
    customer_question: str
    support_agent_response: str
    tags: list[str]
    resolution_type: str
    category: str
    score: float  # fused RRF score, the ranking key
    keyword_rank: int | None  # position in the keyword result set, None if absent
    semantic_rank: int | None  # position in the vector result set, None if absent


def _expand_support_query(query: str) -> str:
    """Add local support-domain synonyms without sending the corpus off-machine."""
    lowered = query.casefold()
    expansions = {
        "cannot get into": "login sign in password account access",
        "can't get into": "login sign in password account access",
        "cant get into": "login sign in password account access",
        "locked out": "login sign in password reset account access",
        "passwrod": "password",
        "rest my": "reset my",
        "won't connect": "integration connection error",
        "cant connect": "integration connection error",
    }
    additions = [replacement for phrase, replacement in expansions.items() if phrase in lowered]
    return " ".join((query, *additions))


_GENERIC_QUERY_WORDS = {
    "a",
    "after",
    "an",
    "and",
    "are",
    "before",
    "behaves",
    "called",
    "can",
    "cannot",
    "different",
    "differently",
    "do",
    "does",
    "error",
    "feature",
    "for",
    "from",
    "get",
    "have",
    "help",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "produces",
    "ravenna",
    "support",
    "the",
    "this",
    "to",
    "undocumented",
    "what",
    "when",
    "with",
}


def _meaningful_terms(text: str) -> set[str]:
    return {
        term.casefold()
        for term in re.findall(r"[A-Za-z0-9]{2,}", text)
        if term.casefold() not in _GENERIC_QUERY_WORDS
    }


def _lexical_coverage(query: str, row: dict) -> float:
    query_terms = _meaningful_terms(query)
    if not query_terms:
        return 0.0
    candidate = " ".join(
        (
            row["customer_question"],
            " ".join(row.get("tags") or []),
            row.get("category") or "",
        )
    )
    return len(query_terms & _meaningful_terms(candidate)) / len(query_terms)


async def search(
    pool: AsyncConnectionPool,
    query: str,
    *,
    query_embedding: list[float] | None = None,
    category: str | None = None,
    limit: int = 3,
) -> list[KBHit]:
    """Hybrid search. Returns at most `limit` articles, best first.

    `query_embedding` is optional; without it this is keyword-only. `category`
    narrows the candidate set when the agent already knows the topic area.
    """
    category_clause = "AND category = %s" if category else ""
    expanded_query = _expand_support_query(query)
    terms = re.findall(r"[A-Za-z0-9]{2,}", expanded_query)
    keyword_query = " OR ".join(dict.fromkeys(terms)) or query
    keyword_params: list = [keyword_query]
    if category:
        keyword_params.append(category)
    keyword_params.append(max(limit * 5, 20))

    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT ticket_id, customer_question, support_agent_response, tags,
                   resolution_type, category,
                   ts_rank(
                       search_vector,
                       websearch_to_tsquery('english', %s)
                   ) AS keyword_score,
                   row_number() OVER (
                       ORDER BY ts_rank(
                           search_vector,
                           websearch_to_tsquery('english', %s)
                       ) DESC
                   )::integer AS rank
            FROM kb_articles
            WHERE search_vector @@ websearch_to_tsquery('english', %s)
              {category_clause}
            LIMIT %s
            """,
            [keyword_query, keyword_query, *keyword_params],
        )
        keyword_rows = await cursor.fetchall()

        fuzzy_params: list = [expanded_query]
        if category:
            fuzzy_params.append(category)
        fuzzy_params.append(max(limit * 5, 20))
        cursor = await conn.execute(
            f"""
            SELECT ticket_id, customer_question, support_agent_response, tags,
                   resolution_type, category,
                   similarity(lower(customer_question), lower(%s)) AS fuzzy_similarity,
                   row_number() OVER (
                       ORDER BY similarity(lower(customer_question), lower(%s)) DESC
                   )::integer AS rank
            FROM kb_articles
            WHERE similarity(lower(customer_question), lower(%s)) >= 0.18
              {category_clause}
            ORDER BY fuzzy_similarity DESC
            LIMIT %s
            """,
            [expanded_query, expanded_query, *fuzzy_params],
        )
        fuzzy_rows = await cursor.fetchall()

        semantic_rows = []
        if query_embedding:
            vector = "[" + ",".join(str(value) for value in query_embedding) + "]"
            semantic_params: list = [vector]
            if category:
                semantic_params.append(category)
            semantic_params.append(max(limit * 5, 20))
            cursor = await conn.execute(
                f"""
                SELECT ticket_id, customer_question, support_agent_response, tags,
                       resolution_type, category,
                       1 - (embedding <=> %s::vector) AS semantic_similarity,
                       row_number() OVER (
                           ORDER BY embedding <=> %s::vector
                       )::integer AS rank
                FROM kb_articles
                WHERE embedding IS NOT NULL
                  {category_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [vector, vector, *semantic_params],
            )
            semantic_rows = await cursor.fetchall()

    combined: dict[str, dict] = {}
    for row in keyword_rows:
        coverage = _lexical_coverage(expanded_query, row)
        if float(row["keyword_score"]) < 0.05 or coverage < 0.3:
            continue
        combined[row["ticket_id"]] = {
            **row,
            "keyword_rank": row["rank"],
            "semantic_rank": None,
            "score": 60 / (60 + row["rank"]),
        }
    for row in semantic_rows:
        coverage = _lexical_coverage(expanded_query, row)
        # A semantic neighbour still needs meaningful lexical grounding. This
        # prevents a query containing an unknown workflow/error identifier from
        # inheriting a confident but unrelated historical resolution.
        if float(row["semantic_similarity"]) < 0.55 or coverage < 0.35:
            continue
        existing = combined.setdefault(
            row["ticket_id"],
            {
                **row,
                "keyword_rank": None,
                "semantic_rank": None,
                "score": 0.0,
            },
        )
        existing["semantic_rank"] = row["rank"]
        existing["score"] = min(1.0, existing["score"] + 60 / (60 + row["rank"]))
    for row in fuzzy_rows:
        similarity = float(row["fuzzy_similarity"])
        coverage = _lexical_coverage(expanded_query, row)
        if similarity < 0.24 or coverage < 0.2:
            continue
        existing = combined.setdefault(
            row["ticket_id"],
            {
                **row,
                "keyword_rank": None,
                "semantic_rank": None,
                "score": similarity,
            },
        )
        existing["score"] = min(1.0, max(existing["score"], similarity))

    ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
    return [
        KBHit(
            ticket_id=row["ticket_id"],
            customer_question=row["customer_question"],
            support_agent_response=row["support_agent_response"],
            tags=row["tags"],
            resolution_type=row["resolution_type"],
            category=row["category"],
            score=float(row["score"]),
            keyword_rank=row["keyword_rank"],
            semantic_rank=row["semantic_rank"],
        )
        for row in ranked[:limit]
    ]


async def get_by_id(pool: AsyncConnectionPool, ticket_id: str) -> KBHit | None:
    """Fetch a single article by its `TK-0NN` identifier."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT ticket_id, customer_question, support_agent_response, tags,
                   resolution_type, category
            FROM kb_articles
            WHERE upper(ticket_id) = upper(%s)
            """,
            (ticket_id,),
        )
        row = await cursor.fetchone()
    if not row:
        return None
    return KBHit(**row, score=1.0, keyword_rank=1, semantic_rank=None)


async def upsert_article(
    pool: AsyncConnectionPool,
    article: dict,
    embedding: list[float] | None = None,
) -> None:
    """Insert or update one article. Used by `scripts/seed_db.py`."""
    vector = (
        "[" + ",".join(str(value) for value in embedding) + "]" if embedding is not None else None
    )
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO kb_articles (
                ticket_id, customer_question, support_agent_response, tags,
                resolution_type, category, embedding
            )
            VALUES (
                %(ticket_id)s, %(customer_question)s, %(support_agent_response)s,
                %(tags)s, %(resolution_type)s, %(category)s, %(embedding)s::vector
            )
            ON CONFLICT (ticket_id) DO UPDATE SET
                customer_question = EXCLUDED.customer_question,
                support_agent_response = EXCLUDED.support_agent_response,
                tags = EXCLUDED.tags,
                resolution_type = EXCLUDED.resolution_type,
                category = EXCLUDED.category,
                embedding = EXCLUDED.embedding
            """,
            {**article, "embedding": vector},
        )
