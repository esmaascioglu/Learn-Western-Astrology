-- ============================================================================
-- Learn-Western-Astrology — vector store schema (Supabase Postgres + pgvector)
-- Dense embeddings: OpenAI text-embedding-3-small (1024-d).
-- Lexical leg 1: Postgres full-text search (BM25-like, exact + heading-weighted).
-- Lexical leg 2: SPLADE learned sparse vectors (30522-d sparsevec, term expansion).
-- Hybrid retrieval = RRF of all three legs (dense + FTS + SPLADE).
-- ============================================================================

create extension if not exists vector;

-- ---- Source documents (one row per book) --------------------------------
create table if not exists documents (
    id          bigint generated always as identity primary key,
    slug        text unique not null,
    title       text not null,
    author      text,
    source_path text,
    num_pages   int,
    created_at  timestamptz not null default now()
);

-- ---- Chunks --------------------------------------------------------------
create table if not exists chunks (
    id          bigint generated always as identity primary key,
    document_id bigint not null references documents(id) on delete cascade,
    chunk_index int    not null,
    content     text   not null,
    page_start  int,
    page_end    int,
    -- structured tags: {planet, sign, aspect, topic, section, ...}
    metadata    jsonb  not null default '{}'::jsonb,
    token_count int,
    embedding   vector(1024),
    -- SPLADE learned-sparse vector over the DistilBERT vocabulary (term expansion).
    sparse_embedding sparsevec(30522),
    -- Section heading weighted 'A', body weight 'D' (default). A query matching
    -- the heading (e.g. "Sun Conjunct Sun") then ranks far above body-only hits.
    fts         tsvector generated always as (
        setweight(to_tsvector('english', coalesce(metadata->>'section', '')), 'A') ||
        to_tsvector('english', content)
    ) stored,
    created_at  timestamptz not null default now(),
    unique (document_id, chunk_index)
);

-- ---- Indexes -------------------------------------------------------------
create index if not exists chunks_embedding_hnsw
    on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_fts_gin
    on chunks using gin (fts);
create index if not exists chunks_metadata_gin
    on chunks using gin (metadata);
-- No ANN index on sparse_embedding: at this corpus size (~2k rows) an exact
-- inner-product scan is sub-millisecond. Add an hnsw sparsevec_ip_ops index here
-- if the corpus grows large.

-- ---- Hybrid search (RRF of dense cosine + FTS rank + SPLADE dot product) -
-- Three complementary legs, fused by Reciprocal Rank Fusion:
--   * semantic  — dense paraphrase match (OpenAI embedding, cosine).
--   * fts        — exact lexical, heading-weighted (BM25-like). High precision.
--   * sparse     — SPLADE learned-sparse, term-expanded. Closes the vocab gap.
-- Lexical legs are down-weighted slightly (2 of 3 legs are lexical) so the dense
-- semantic signal is not out-voted.
drop function if exists hybrid_search(text, vector, int, int, float, float, jsonb);

create or replace function hybrid_search(
    query_text       text,
    query_embedding  vector(1024),
    query_sparse     sparsevec(30522) default null,
    match_count      int   default 20,
    rrf_k            int   default 50,
    full_text_weight float default 0.7,
    semantic_weight  float default 1.0,
    sparse_weight    float default 1.0,
    filter           jsonb default '{}'::jsonb
)
returns table (
    id          bigint,
    document_id bigint,
    content     text,
    page_start  int,
    page_end    int,
    metadata    jsonb,
    score       double precision
)
language sql stable
as $$
with q as (
    select to_tsquery(
        'english',
        replace(plainto_tsquery('english', query_text)::text, ' & ', ' | ')
    ) as tsq
),
fts as (
    select c.id,
           row_number() over (order by ts_rank_cd(c.fts, q.tsq) desc) as rank
    from chunks c, q
    where c.metadata @> filter
      and c.fts @@ q.tsq
    limit match_count * 2
),
semantic as (
    select c.id,
           row_number() over (order by c.embedding <=> query_embedding) as rank
    from chunks c
    where c.metadata @> filter
    order by c.embedding <=> query_embedding
    limit match_count * 2
),
sparse as (
    select c.id,
           row_number() over (order by c.sparse_embedding <#> query_sparse) as rank
    from chunks c
    where query_sparse is not null
      and c.sparse_embedding is not null
      and c.metadata @> filter
    order by c.sparse_embedding <#> query_sparse
    limit match_count * 2
),
ids as (
    select id from fts
    union select id from semantic
    union select id from sparse
),
combined as (
    select i.id,
           coalesce(full_text_weight / (rrf_k + fts.rank), 0.0) +
           coalesce(semantic_weight  / (rrf_k + semantic.rank), 0.0) +
           coalesce(sparse_weight    / (rrf_k + sparse.rank), 0.0) as score
    from ids i
    left join fts      on fts.id = i.id
    left join semantic on semantic.id = i.id
    left join sparse   on sparse.id = i.id
)
select c.id, c.document_id, c.content, c.page_start, c.page_end, c.metadata, combined.score
from combined
join chunks c on c.id = combined.id
order by combined.score desc
limit match_count;
$$;
