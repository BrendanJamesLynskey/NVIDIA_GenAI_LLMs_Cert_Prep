# RAG Systems

Retrieval-Augmented Generation accounts for 22% of the NCA-GENL Associate exam (Experimentation domain) and 9% of the NCP-GENL Professional exam (Data Preparation domain). Questions span the full pipeline — from embedding choice and index construction through to evaluation metrics — so understanding each component in isolation and how they interact is necessary. The portfolio repos below contain the technical depth; this file maps the cert-relevant distinctions and trade-offs.

---

## Parametric vs Non-Parametric Memory

A pretrained language model stores knowledge in its weights — this is parametric memory. It is fixed at inference time, cannot be updated without retraining, and cannot cite sources. The model may also confabulate: generating plausible-sounding but incorrect facts.

The original RAG paper (Lewis et al., NeurIPS 2020) framed the core argument: combining a seq2seq model with a dense retrieval index over Wikipedia achieves state-of-the-art on open-domain QA while producing "more specific, diverse and factual" language than parametric-only baselines. The retrieval index is non-parametric memory — it can be updated by modifying the index without retraining the model.

**RAG-Sequence vs RAG-Token.** Lewis et al. proposed two variants: RAG-Sequence uses the same retrieved passages for the entire generated output; RAG-Token allows different passages to inform each generated token. RAG-Sequence is simpler and is the standard in most production implementations; RAG-Token is rarely used outside of research.

**When RAG is the right choice.** RAG is appropriate when: the knowledge base is large and frequently updated; precise source attribution is required; retraining is too slow or expensive for the update cadence. RAG is the wrong choice when: the corpus is small enough to fit in context (just use context stuffing); the data is structured and SQL is sufficient; the update frequency is very low and fine-tuning is viable.

---

## Embedding Models

The retriever is the most consequential component in a RAG pipeline. A weak retriever cannot be compensated for by a better LLM.

**Dense embeddings (bi-encoder).** Both the query and document are encoded independently to fixed-dimension vectors. Retrieval is a nearest-neighbour search over pre-computed document embeddings. The cosine similarity (or dot product) between query and document vectors is the relevance score. Dense embeddings capture semantic similarity and generalise across paraphrases but can miss exact-match signals.

**Sparse embeddings.** BM25 is the classical sparse retrieval function: term-frequency × inverse-document-frequency with saturation and length normalisation. SPLADE and similar learned sparse models produce token-level sparse vectors that combine lexical precision with some semantic generalisation. Sparse retrieval is excellent at exact-match, abbreviations, and rare proper nouns.

**ColBERT-style late interaction.** Rather than encoding a document to a single vector, ColBERT retains per-token vectors for both query and document and computes relevance as the sum of maximum inner products (MaxSim) across query tokens. This is more accurate than single-vector bi-encoders but more expensive to store and retrieve. It sits between bi-encoder speed and cross-encoder accuracy.

**Cross-encoders.** The query and document are concatenated and processed jointly through a transformer. This produces very accurate relevance scores — the model sees the interaction between query and document — but is too slow for first-stage retrieval over large corpora (O(N) forward passes per query). Cross-encoders are used exclusively as **rerankers** applied to a small candidate set from the first stage.

Full technical depth: [RAG\_01\_Embedding\_Models](https://github.com/BrendanJamesLynskey/RAG_01_Embedding_Models).

---

## Vector Databases

After embedding, documents are stored in an index that supports approximate nearest-neighbour (ANN) search.

**HNSW (Hierarchical Navigable Small World).** A graph-based index: each node maintains edges to near neighbours at multiple hierarchy levels. Query time is $O(\log N)$ with high recall at low ef\_search values. HNSW is memory-intensive (stores the graph in RAM) but is the standard default for latency-sensitive applications.

**IVF (Inverted File Index).** Clusters documents at index time; query searches only a subset of clusters (nprobe). Faster to build than HNSW and more memory-efficient. Recall degrades if the query cluster assignment is wrong; combining with PQ improves compression at the cost of some recall.

**PQ (Product Quantisation).** Compresses high-dimensional vectors by splitting them into sub-vectors and quantising each sub-vector independently. Reduces storage by 8–32× but introduces approximation error. Typically combined with IVF or HNSW rather than used alone.

**Database options.**

| System | Storage backend | Strengths | Weaknesses |
| --- | --- | --- | --- |
| pgvector | PostgreSQL | SQL joins, ACID, familiar ops tooling | Slower ANN than dedicated systems at scale |
| Qdrant | Custom Rust engine | Fast HNSW, payload filtering, sparse+dense | Newer ecosystem |
| Weaviate | Custom Go engine | Hybrid search built-in, GraphQL, modules | Resource-hungry default config |
| Pinecone | Managed cloud | Fully managed, simple API | Proprietary, cloud-only, cost |
| Chroma | DuckDB / Parquet | Simple, embeddable, good for dev | Not production-scale |

Full technical depth: [RAG\_02\_Vector\_Databases](https://github.com/BrendanJamesLynskey/RAG_02_Vector_Databases).

---

## Hybrid Search and Reranking

Neither dense nor sparse retrieval alone is optimal. Dense retrieval misses exact lexical matches; sparse retrieval misses paraphrase and semantic similarity. Hybrid search combines both.

**BM25 + dense fusion.** Run BM25 and dense ANN retrieval independently, producing two ranked lists. Merge them. Reciprocal Rank Fusion (RRF) is the standard merging function:

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + r(d)}$$

where $r(d)$ is the rank of document $d$ in ranker $r$ and $k = 60$ is a smoothing constant. RRF requires no trained weights, is robust across retrieval quality distributions, and consistently outperforms individual retrievers on mixed-query workloads.

**Cross-encoder reranking.** After first-stage retrieval (typically top 50–200 documents), a cross-encoder reranker scores each candidate by processing the (query, document) pair jointly. The top-k reranked documents are then passed to the generator. The reranker substantially improves precision at the cost of latency proportional to the number of candidates scored.

The typical hybrid RAG retrieval stack is: BM25 || dense ANN → RRF merge → cross-encoder rerank → top-k to LLM context.

Full technical depth: [RAG\_03\_Hybrid\_Search\_and\_Reranking](https://github.com/BrendanJamesLynskey/RAG_03_Hybrid_Search_and_Reranking).

---

## Chunking and Ingestion

The retrieval unit must be sized correctly: too large and context gets swamped; too small and each chunk lacks sufficient context for the LLM to generate a coherent answer.

**Fixed-size chunking.** Split by token count with a sliding overlap (e.g., 512 tokens, 64-token overlap). Simple and predictable. Breaks mid-sentence; loses discourse structure. Suitable for homogeneous prose.

**Semantic / sentence chunking.** Segment at natural sentence or paragraph boundaries. Produces more coherent chunks at the cost of variable sizes. Libraries such as LangChain's `SemanticChunker` embed adjacent sentences and split where cosine similarity drops.

**Layout-aware chunking.** Document structure (headers, tables, code blocks, captions) is preserved during parsing. Tools such as Unstructured or LlamaParse extract layout-annotated elements before chunking. Critical for PDFs, HTML, and mixed-modality documents where a naive text split would mix table cells with prose.

**Metadata enrichment.** Attaching metadata — source URL, document date, section heading, page number — enables filtered retrieval (e.g., "only retrieve from documents dated after 2024-01-01") and improves citation generation. Metadata filtering runs before or inside the ANN search depending on the vector database.

Full technical depth: [RAG\_04\_Chunking\_and\_Ingestion](https://github.com/BrendanJamesLynskey/RAG_04_Chunking_and_Ingestion).

---

## Agentic RAG

Standard single-pass RAG (retrieve → generate) is brittle for multi-step questions. Agentic RAG patterns introduce reasoning loops over retrieval.

**HyDE (Hypothetical Document Embeddings).** The LLM generates a hypothetical answer to the query, then the embedding of that hypothetical answer is used as the retrieval query rather than the original question. Useful when the original query is short and ambiguous; the hypothetical answer is closer in embedding space to relevant documents.

**Self-RAG.** The model is trained to emit reflection tokens — `[Retrieve]`, `[Relevant]`, `[Supported]`, `[IsUseful]` — interspersed in its output. It decides when to retrieve, evaluates retrieved passages for relevance and grounding, and can abort retrieval if the passage is unhelpful. Reduces unnecessary retrieval calls compared to retrieve-always approaches.

**Corrective RAG (CRAG).** Adds an evaluator that scores retrieved documents; if none score above a threshold, the system triggers a web search fallback or rewrites the query. Addresses the failure mode where first-stage retrieval returns nothing useful.

**Query routing and decomposition.** A router classifies the incoming query and directs it to the appropriate retrieval source (vector store, SQL database, web search, API). Query decomposition breaks a complex multi-part question into sub-queries, executes them in parallel or sequence, and merges results before final generation.

Full technical depth: [RAG\_05\_Agentic\_RAG\_Patterns](https://github.com/BrendanJamesLynskey/RAG_05_Agentic_RAG_Patterns).

---

## GraphRAG

Standard vector RAG retrieves chunks based on local semantic similarity. It does not capture explicit entity relationships or multi-hop reasoning paths.

**Microsoft GraphRAG** (Edge et al., 2024) constructs a knowledge graph from the corpus: entities and relationships are extracted by an LLM, then clustered into communities, and community summaries are generated at multiple granularities. At query time, community summaries are retrieved alongside raw chunks, giving the LLM access to global context that dense vector search cannot surface.

GraphRAG improves performance on questions requiring synthesis across many documents — "what are the main themes across the entire corpus?" — at substantially higher construction cost. It is not a replacement for vector RAG on document-level factual lookup.

**Hybrid graph+vector.** Production systems increasingly combine both: a vector index for semantic chunk retrieval and a knowledge graph for entity-centric navigation. The two can complement each other when the query routing layer directs accordingly.

Full technical depth: [RAG\_06\_GraphRAG\_and\_KGs](https://github.com/BrendanJamesLynskey/RAG_06_GraphRAG_and_KGs).

---

## Evaluation

RAG evaluation must assess both the retrieval stage and the generation stage independently, because failures in each have different remediation paths.

**RAGAS** (Retrieval Augmented Generation Assessment; Es et al., 2023) is a framework for reference-free evaluation of RAG pipelines. It uses an LLM to score the following dimensions:

- **Faithfulness** — does the generated answer contain only claims that are supported by the retrieved context? (Measures hallucination relative to context, not relative to ground truth.)
- **Answer relevancy** — is the generated answer responsive to the original question?
- **Context precision** — of the retrieved chunks, what proportion are actually relevant to the question?
- **Context recall** — does the retrieved context contain the information needed to answer the question?

The reference-free design is practically important: it allows iterative evaluation without annotating ground-truth answers for every test question.

**Citation / grounding metrics.** Beyond RAGAS, production systems measure attribution: does each factual claim in the response map to a specific retrieved source? This is distinct from faithfulness (which measures consistency with retrieved context) and is the metric end-users and auditors most care about.

**Retrieval recall@k.** On a labelled test set, the fraction of questions for which at least one relevant document appears in the top-k retrieved results. The standard diagnostic for retrieval failures independent of generation quality.

Full coverage of production RAG, monitoring, and evaluation: [RAG\_07\_Production\_RAG](https://github.com/BrendanJamesLynskey/RAG_07_Production_RAG). Evaluation breadth: [LLM\_Hub\_Evaluations](https://github.com/BrendanJamesLynskey/LLM_Hub_Evaluations).

---

## Likely Exam Angles

- **Bi-encoder vs cross-encoder use cases.** Bi-encoders are used for first-stage retrieval at scale; cross-encoders are used for reranking a small candidate set. A question asking "which is appropriate for ranking 10 million documents in real time?" points to bi-encoder. Distractors may suggest cross-encoders for their accuracy advantage without acknowledging the latency cost.
- **RRF vs learned fusion.** RRF requires no training, uses only rank positions (not scores), and is robust to scale mismatch between BM25 and dense scores. Distractors may describe score normalisation as the standard fusion approach; RRF is the default.
- **RAGAS faithfulness vs answer relevancy.** Faithfulness is about whether the answer is grounded in the retrieved context; answer relevancy is about whether the answer addresses the question. A faithful but irrelevant answer is possible (the answer accurately reports something from context but doesn't answer the question asked).
- **When not to use RAG.** Examiners test this directly. RAG adds latency, complexity, and retrieval failure modes. For small, static corpora; structured data queryable by SQL; or tasks where the LLM's parametric knowledge is sufficient — RAG is not the right default.
- **HyDE mechanism.** HyDE encodes a *hypothetical answer* as the retrieval query, not the original question. This is counterintuitive and is a common distractor target.
- **GraphRAG vs vector RAG.** GraphRAG is better for global synthesis across documents; vector RAG is better for local fact lookup. GraphRAG has higher construction cost. They are not mutually exclusive.

---

## Further Reading

- RAG embedding models depth: [RAG\_01\_Embedding\_Models](https://github.com/BrendanJamesLynskey/RAG_01_Embedding_Models)
- Vector database depth: [RAG\_02\_Vector\_Databases](https://github.com/BrendanJamesLynskey/RAG_02_Vector_Databases)
- Hybrid search and reranking: [RAG\_03\_Hybrid\_Search\_and\_Reranking](https://github.com/BrendanJamesLynskey/RAG_03_Hybrid_Search_and_Reranking)
- Chunking and ingestion: [RAG\_04\_Chunking\_and\_Ingestion](https://github.com/BrendanJamesLynskey/RAG_04_Chunking_and_Ingestion)
- Agentic RAG patterns: [RAG\_05\_Agentic\_RAG\_Patterns](https://github.com/BrendanJamesLynskey/RAG_05_Agentic_RAG_Patterns)
- GraphRAG and knowledge graphs: [RAG\_06\_GraphRAG\_and\_KGs](https://github.com/BrendanJamesLynskey/RAG_06_GraphRAG_and_KGs)
- Production RAG, monitoring, evaluation: [RAG\_07\_Production\_RAG](https://github.com/BrendanJamesLynskey/RAG_07_Production_RAG)
- RAG & Retrieval hub: [LLM\_Hub\_RAG\_Retrieval](https://github.com/BrendanJamesLynskey/LLM_Hub_RAG_Retrieval)
- Lewis et al. (2020), original RAG paper: <https://arxiv.org/abs/2005.11401>
- Es et al. (2023), RAGAS paper: <https://arxiv.org/abs/2309.15217>
- Edge et al. (2024), Microsoft GraphRAG: <https://arxiv.org/abs/2404.16130>
- Self-RAG: <https://arxiv.org/abs/2310.11511>
