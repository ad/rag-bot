package main

import (
	"fmt"
	"log"
	"os"

	"github.com/ad/rag-bot/internal/llm"
	"github.com/ad/rag-bot/internal/parser"
	"github.com/ad/rag-bot/internal/vectorstore"
)

func main() {
	// Создаем папку если её нет
	if err := os.MkdirAll("data", 0755); err != nil {
		log.Fatal(err)
	}

	fmt.Println("=== Тест полной системы RAG ===")

	// 1. Инициализируем компоненты
	fmt.Println("1. Инициализация компонентов...")
	markdownParser := parser.NewMarkdownParser()
	llmClient := llm.NewOllamaClient()
	vectorStore := vectorstore.NewVectorStore()

	// 2. Парсим документы
	fmt.Println("2. Парсинг документов...")
	documents, err := markdownParser.ParseDirectory("data")
	if err != nil {
		log.Fatalf("Ошибка парсинга: %v", err)
	}
	fmt.Printf("Найдено документов: %d\n", len(documents))

	// 3. Генерируем эмбеддинги для каждого документа
	fmt.Println("3. Генерация эмбеддингов...")
	for i, doc := range documents {
		fmt.Printf("Обрабатываем документ %d/%d: %s\n", i+1, len(documents), doc.Title)

		// Комбинируем заголовок и содержимое для эмбеддинга
		text := doc.Title + "\n" + doc.Content

		embedding, err := llmClient.GenerateEmbedding(text)
		if err != nil {
			log.Printf("Ошибка генерации эмбеддинга для %s: %v", doc.ID, err)
			continue
		}

		documents[i].Embedding = embedding
		fmt.Printf("Эмбеддинг создан (размер: %d)\n", len(embedding))
	}

	// 4. Добавляем документы в векторное хранилище
	fmt.Println("4. Добавление документов в векторное хранилище...")
	vectorStore.AddDocuments(documents)
	fmt.Printf("Документов в хранилище: %d\n", vectorStore.GetDocumentCount())

	// 5. Тестируем поиск
	fmt.Println("\n5. Тестирование поиска...")
	testQueries := []string{
		"как создать сайт",
		"цели и задачи компании",
		"техническая поддержка",
		"разработка веб-приложений",
	}

	for _, query := range testQueries {
		fmt.Printf("\n--- Запрос: \"%s\" ---\n", query)

		// Генерируем эмбеддинг для запроса
		queryEmbedding, err := llmClient.GenerateEmbedding(query)
		if err != nil {
			log.Printf("Ошибка генерации эмбеддинга для запроса: %v", err)
			continue
		}

		// Ищем похожие документы
		results, err := vectorStore.Search(queryEmbedding, 3)
		if err != nil {
			log.Printf("Ошибка поиска: %v", err)
			continue
		}

		fmt.Printf("Найдено результатов: %d\n", len(results))
		for i, result := range results {
			fmt.Printf("%d. %s (Score: %.4f)\n", i+1, result.Document.Title, result.Score)
			fmt.Printf("   URL: %s\n", result.Document.URL)
			fmt.Printf("   Содержимое: %s...\n", truncateString(result.Document.Content, 100))
		}
	}

	fmt.Println("\n=== Тест завершен ===")
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
