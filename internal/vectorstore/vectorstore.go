package vectorstore

import (
	"fmt"
	"math"
	"sort"

	"github.com/ad/rag-bot/internal/types"
)

type VectorStore struct {
	documents []types.Document
}

type SearchResult struct {
	Document types.Document
	Score    float32
}

func NewVectorStore() *VectorStore {
	return &VectorStore{
		documents: make([]types.Document, 0),
	}
}

func (vs *VectorStore) AddDocument(doc types.Document) {
	vs.documents = append(vs.documents, doc)
}

func (vs *VectorStore) AddDocuments(docs []types.Document) {
	vs.documents = append(vs.documents, docs...)
}

func (vs *VectorStore) Search(queryEmbedding []float32, topK int) ([]SearchResult, error) {
	if len(vs.documents) == 0 {
		return nil, fmt.Errorf("векторное хранилище пустое")
	}

	if len(queryEmbedding) == 0 {
		return nil, fmt.Errorf("эмбеддинг запроса пустой")
	}

	if topK <= 0 {
		topK = 5
	}

	var results []SearchResult
	documentsWithEmbeddings := 0

	for _, doc := range vs.documents {
		if len(doc.Embedding) == 0 {
			continue
		}

		documentsWithEmbeddings++
		score := cosineSimilarity(queryEmbedding, doc.Embedding)

		// Фильтруем результаты с очень низким скором
		if score > 0.1 {
			results = append(results, SearchResult{
				Document: doc,
				Score:    score,
			})
		}
	}

	if documentsWithEmbeddings == 0 {
		return nil, fmt.Errorf("нет документов с эмбеддингами")
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("не найдено релевантных документов")
	}

	// Сортируем по убыванию схожести
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Возвращаем топ-K результатов
	if topK > len(results) {
		topK = len(results)
	}

	return results[:topK], nil
}

func (vs *VectorStore) GetDocumentCount() int {
	return len(vs.documents)
}

// cosineSimilarity вычисляет косинусное сходство между двумя векторами
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64

	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)))
}
