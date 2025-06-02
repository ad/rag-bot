package retrieval

import (
	"fmt"

	"github.com/ad/rag-bot/internal/llm"
	"github.com/ad/rag-bot/internal/types"
	"github.com/ad/rag-bot/internal/vectorstore"
)

type RetrievalEngine interface {
	FindRelevantDocuments(query string, limit int) ([]types.Document, error)
}

type VectorRetrieval struct {
	vectorStore *vectorstore.VectorStore
	llmEngine   llm.LLMEngine
}

func NewVectorRetrieval(vs *vectorstore.VectorStore, llm llm.LLMEngine) *VectorRetrieval {
	return &VectorRetrieval{
		vectorStore: vs,
		llmEngine:   llm,
	}
}

func (vr *VectorRetrieval) FindRelevantDocuments(query string, limit int) ([]types.Document, error) {
	// Генерируем эмбеддинг для запроса
	queryEmbedding, err := vr.llmEngine.GenerateEmbedding(query)
	if err != nil {
		return nil, fmt.Errorf("ошибка генерации эмбеддинга для запроса: %w", err)
	}

	// Ищем похожие документы
	results, err := vr.vectorStore.Search(queryEmbedding, limit)
	if err != nil {
		return nil, fmt.Errorf("ошибка векторного поиска: %w", err)
	}

	// Возвращаем документы
	var documents []types.Document
	for _, result := range results {
		documents = append(documents, result.Document)
	}

	return documents, nil
}
