import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from google import genai
from dotenv import load_dotenv
import os

# -------------------------------------
# Configuration
# -------------------------------------

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3
RERANK_TOP_K = 4
FINAL_TOP_K = 2
GEMINI_MODEL = "gemini-3.1-flash-lite"
CHROMA_DB_PATH = "./rag_chroma_db"
COLLECTION_NAME = "rag_documents"


def normalize_vector(distances):
    """
    Convert ChromaDB distances into 0-1 relevance scores.
    Lower distance gets a higher relevance score.
    """
    min_distance = min(distances)
    max_distance = max(distances)

    normalized = [
        (max_distance - distance) /
        (max_distance - min_distance)
        for distance in distances
    ]

    return normalized

def normalize_bm25(scores):
    """
    Convert BM25 scores into 0-1 relevance scores.
    Higher BM25 scores receive higher normalized scores.
    """
    min_score = min(scores)
    max_score = max(scores)

    # Handle the case where all BM25 scores are identical.
    if max_score == min_score:
        return[0.0 for _ in scores ]

    normalized = [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]

    return normalized

def hybrid_score(vector_score, bm25_score, vector_weight=VECTOR_WEIGHT, bm25_weight=BM25_WEIGHT):
    vector_part = vector_score * vector_weight
    bm25_part = bm25_score * bm25_weight
    final_score= vector_part + bm25_part

    return final_score

# Initialize the BGE reranker model
reranker = CrossEncoder("BAAI/bge-reranker-base")

# Load environment variable from .env
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. Please add it to your .env file."
    )

# Initialize the Gemini client
client_gemini = genai.Client(
    api_key=gemini_api_key
)

# Connect to the existing persistent ChromaDB database
client =  chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Access the collection that contains our RAG chunks
collection =  client.get_collection(COLLECTION_NAME)


# -------------------------
# Knowledge Base Setup
# -------------------------

all_documents = collection.get()

documents = all_documents["documents"]
all_ids = all_documents["ids"]
all_metadatas = all_documents["metadatas"]

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

document_map = {}

for i in range(len(all_ids)):
    document_map[all_ids[i]] = documents[i]

print("\n--- STORED CHUNKS ---")

for doc_id,document in document_map.items():
    print(f"\nID: {doc_id}")
    print(f"Content: {document}")

metadata_map = {}

for i in range(len(all_ids)):
    metadata_map[all_ids[i]] = all_metadatas[i]

def run_rag(query):
    """
    Run the complete RAG pipeline for a user query
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    
    query = query.strip()

    # Retrieve the most relevant chunk from chromaDB
    results = collection.query(
        query_texts=[query],
        n_results=len(all_ids)
    )

    # Extract IDs adn distances from the first query result
    ids = results["ids"][0]
    distances = results["distances"][0]


    # Calculate BM25 score for all stored chunks
    tokenized_query = query.lower().split()
    bm25_raw_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores to a 0-1 range
    normalized_bm25_scores = normalize_bm25(bm25_raw_scores)

    # Map each chunk ID to its normalized BM25 score
    bm25_scores = {}

    for i in range(len(all_ids)):
        bm25_scores[all_ids[i]] = normalized_bm25_scores[i]

    # Normalize chromaDB distances into 0-1 vector relevance scores
    normalized_vector_scores = normalize_vector(distances)

    # Map each chunk ID tot its normalized vector score
    vector_scores = {}

    for i in range(len(ids)):
        vector_scores[ids[i]] =  normalized_vector_scores[i]

    # Combine candidate IDs from vector search and BM25
    candidate_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

    # Calculate hybrid scores for all candidate chunks
    hybrid_results = {}

    for doc_id in candidate_ids:
        vector_score = vector_scores.get(doc_id, 0.0)
        bm25_score = bm25_scores.get(doc_id, 0.0)

        score = hybrid_score(vector_score, bm25_score)

        hybrid_results[doc_id] = score

    # Rank candidate chunks by hybrid score from higher to lowest
    ranked_results = sorted(
        hybrid_results.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n--- HYBRID RANKING ---")

    for doc_id,score in ranked_results[:10]:
        print(
            doc_id,
            "Hybrid Score:",
            score,
            "BM25:",
            bm25_scores.get(doc_id,0.0),
            "Vector:",
            vector_scores.get(doc_id,0.0)
        )

    # Select the top hybrid-search candidates for reranking
    rerank_candidates = ranked_results[:RERANK_TOP_K]

    # Create query-document paris for the reranker
    rerank_pairs = [
        (query, document_map[doc_id])
        for doc_id, score in rerank_candidates
    ]

    # Generate relevance scores for each query-document pair
    rerank_scores = reranker.predict(rerank_pairs)
    print("Rerank Scores:", rerank_scores)

    # Pair each candidate chunk ID with its reranker score
    reranked_results = list(
        zip(
            [doc_id for doc_id,score in rerank_candidates],
            rerank_scores
        )
    )

    # Rank chunks by reranker score from highest to lowest
    final_reranked_results = sorted(
        reranked_results,
        key=lambda x: x[1],
        reverse=True
    )

    print("Final Reranked Results:", final_reranked_results)

    # Select the final top 2 chunks after reranking
    final_results =  final_reranked_results[:FINAL_TOP_K]

    # Retrieve the actual text for the final selected chunks
    final_chunks = []

    for doc_id, score in final_results:
        final_chunks.append(document_map[doc_id])

    # Combine the final chunks into a single context
    context = "\n\n".join(final_chunks)

    # Create a grounded prompt using the retrieved context
    prompt = f"""
    Answer the question using only the provided context.
    
    Rules:
    -Do not guess or invent information.
    -If the answer is not present in the context, say:
    "I couldn't find enough information in the knowledge base."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    try:

        # Generate an answer using the retrived context
        response =  client_gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

    except Exception as e:
        raise RuntimeError(
            f"Gemini API request failed: {e}"
        ) from e

    # return response.text
    return {
        "answer": response.text,
        "context": context,
        "final_results": final_results
    }

def evaluate_groundedness(question, context, answer):
    """
    Evaluate whether the generated answer is fully supported
    by the retrieved context.
    """

    prompt =  f"""
    You are a strict RAG evaluation judge.

    Your task is to determine whether the generated answer
    is fully supported by the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    {answer}

    Rules:
    - Use ONLY the provided context.
    - Do NOT use outside knowledge.
    - If all factual claims in the answer are supported
    by the context, return GROUNDED.
    - If the answer contains any unsupported factual claim,
    return NOT_GROUNDED.
    - Return ONLY one of these two labels:

    GROUNDED
    NOT_GROUNDED
    """

    try:
        response = client_gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

    except Exception as e:
        raise RuntimeError(
            f"Groundedness evaluation failed: {e}"
        ) from e

    return response.text.strip()

def evaluate_answer(question, expected_answer, generated_answer):
    """
    Evaluate whether the generated answer correctly answers
    the question based on the expected answer.
    """

    if expected_answer is None:
        return "NOT_APPLICABLE"

    prompt = f"""
    You are a strict RAG answer evaluation judge.

    Your task is to determine whether the generated answer
    correctly answers the question.

    Question:
    {question}

    Expected Answer:
    {expected_answer}

    Generated Answer:
    {generated_answer}

    Rules:
     - Compare the meaning, not exact wording.
    - The generated answer does not need to use the same words
      as the expected answer.
    - If the generated answer correctly conveys the key information
      from the expected answer, return CORRECT.
    - If the generated answer is incorrect, incomplete, or fails
      to answer the question, return INCORRECT.
    - Return ONLY one of these labels:

    CORRECT
    INCORRECT
    """

    try:

        response = client_gemini.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )

    except Exception as e:
        raise RuntimeError(
            f"Answer evaluation failed: {e}"
        )

    return response.text.strip()


# Evaluation questions for testing the RAG pipeline
evaluation_questions = [
    "What is RAG?",
    "How does RAG improve LLM responses?",
    "What does RAG retrieve?",
    "What is the role of a knowledge base in RAG?",
    "What is the capital of Japan?"
]

expected_chunks = {
    "What is RAG?": ["chunk_0"],
    "How does RAG improve LLM responses?": ["chunk_0", "chunk_1"],
    "What does RAG retrieve?": ["chunk_1"],
    "What is the role of a knowledge base in RAG?": ["chunk_1"],
    "What is the capital of Japan?": []
}

expected_answers = {
    "What is RAG?": "RAG is a technique that improves the responses generated by Large Language Models.",
    "How does RAG improve LLM responses?": "RAG improves LLM responses by retrieving relevant information from a knowledge base or vector database.",
    "What does RAG retrieve?": "RAG retrieves relevant information from a knowledge base or vector database.",
    "What is the role of a knowledge base in RAG?": "A knowledge base provides relevant information that RAG retrieves to improve LLM responses.",
    "What is the capital of Japan?": None
}

# Test each evaluation question

hit_count = 0

recall_total = 0
recall_count = 0

reciprocal_rank_total = 0
reciprocal_rank_count = 0

precision_total = 0
precision_count = 0

for query in evaluation_questions:
    print("\n" + "=" * 50)
    print("Testing Query:", query)
    print("=" * 50)
    
    result =  run_rag(query)

    print("Answer:", result["answer"])

    groundedness = evaluate_groundedness(
        query,
        result["context"],
        result["answer"]
    )

    print("Groudedness:", groundedness)

    answer_evaluation = evaluate_answer(
        query,
        expected_answers[query],
        result["answer"]
    )

    print("Answer Evaluation:", answer_evaluation)

    print("Final Results:", result["final_results"])

    print("Context:")
    print(result["context"])
    
    retrieved_ids = [
        doc_id for doc_id, score in result["final_results"]
    ]

    expected =  expected_chunks[query]

    # Anwer evaluation
    expected_answer = expected_answers[query]
    answer = result["answer"]

    if expected_answer:
        expected_words = expected_answer.lower().split()
        answer_lower = answer.lower()

        matched_words = [
            word
            for word in expected_words
            if word in answer_lower
        ]

        print("Matched Words:", matched_words)

    if expected:
        relevant_retrieved = [
            chunk_id
            for chunk_id in expected
            if chunk_id in retrieved_ids
        ]

        recall = len(relevant_retrieved) / len(expected)
    else:
        recall = None

    print("Recall@2:", recall)

    # Precision@2
    if expected:
        relevant_retrieved = [
            chunk_id
            for chunk_id in retrieved_ids
            if chunk_id in expected
        ]

        precision = len(relevant_retrieved) / len(retrieved_ids)
    else:
        precision = None

    print("Precision@2:", precision)

    if precision is not None:
        precision_total += precision
        precision_count += 1

    # Reciprocal Rank
    reciprocal_rank = 0

    for rank,chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected:
            reciprocal_rank = 1/rank
            break

    print("Reciprocal Rank:", reciprocal_rank)

    # Overall MRR
    if expected:
        reciprocal_rank_total += reciprocal_rank
        reciprocal_rank_count += 1

    # Counter for overall Recall@2  

    if recall is not None:
        recall_total += recall
        recall_count += 1

    hit = any(
        chunk_id in retrieved_ids
        for chunk_id in expected_chunks[query]
    )

    print("Retrieved IDs:", retrieved_ids)
    print("Expected IDs:", expected_chunks[query])
    print("Hit:", hit)

    if hit:
        hit_count += 1

hit_at_2 = hit_count / len(evaluation_questions)

print("\nOverall Hit@2:", hit_at_2)

overall_recall_at_2 = recall_total / recall_count

print("Overall Recall@2:", overall_recall_at_2)

overall_mrr = reciprocal_rank_total / reciprocal_rank_count

print("Overall MRR:", overall_mrr)

overall_precision_at_2 = precision_total / precision_count

print("Overall Precision@2:", overall_precision_at_2)
