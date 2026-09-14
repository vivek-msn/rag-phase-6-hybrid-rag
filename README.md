# RAG Phase 6 — Hybrid Search & Reranking

A production-style Retrieval-Augmented Generation (RAG) pipeline built with Python that combines dense vector search, BM25 keyword retrieval, hybrid ranking, cross-encoder reranking, and Gemini for grounded answer generation.

## 🚀 What I Built

This project implements an end-to-end RAG pipeline designed to improve retrieval quality before sending context to an LLM.

The system combines:

- Document chunking
- Sentence Transformer embeddings
- ChromaDB vector storage
- Semantic vector search
- BM25 keyword search
- Hybrid retrieval
- Cross-encoder reranking
- Context construction
- Gemini-based answer generation
- Retrieval evaluation
- Answer and groundedness evaluation

## 🏗️ Architecture

```text
                    Documents
                        │
                        ▼
                    Chunking
                        │
                        ▼
                   Embeddings
                        │
                        ▼
                    ChromaDB
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Vector Search           BM25 Search
             │                     │
             └──────────┬──────────┘
                        ▼
                  Hybrid Ranking
                        │
                        ▼
              Cross-Encoder Reranking
                        │
                        ▼
                  Top Relevant Chunks
                        │
                        ▼
                 Context Building
                        │
                        ▼
                    Gemini LLM
                        │
                        ▼
                   Final Answer