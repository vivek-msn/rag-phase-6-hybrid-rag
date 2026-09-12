# RAG Phase 6 — Hybrid Search & Reranking

A Retrieval-Augmented Generation (RAG) project built with Python that combines vector search, BM25 keyword search, hybrid ranking, cross-encoder reranking, and Gemini for grounded answer generation.

## 🚀 Project Overview

This project demonstrates a complete RAG pipeline:

1. Document chunking
2. Embedding generation
3. Vector storage with ChromaDB
4. Semantic vector search
5. BM25 keyword search
6. Hybrid search
7. Cross-encoder reranking
8. Context construction
9. Gemini-based answer generation
10. Retrieval and answer evaluation

## 🏗️ RAG Pipeline

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
┌───────────────────────┐
│                       │
│  Vector Search        │
│  BM25 Search          │
│                       │
└───────────┬───────────┘
            ↓
      Hybrid Ranking
            ↓
    Cross-Encoder Reranking
            ↓
       Top Results
            ↓
      Context Building
            ↓
        Gemini LLM
            ↓
        Final Answer