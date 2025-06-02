package main

import (
	"fmt"
	"log"
	"os"

	"github.com/ad/rag-bot/internal/parser"
)

func main() {
	// Создаем папку для тестов если её нет
	if err := os.MkdirAll("test_data", 0755); err != nil {
		log.Fatal(err)
	}

	parser := parser.NewMarkdownParser()

	// Тестируем парсинг одного файла
	fmt.Println("=== Тест парсинга одного файла ===")
	doc, err := parser.ParseFile("data/avtomatizatsiya_v_onlayn_kazino.md")
	if err != nil {
		log.Printf("Ошибка: %v", err)
	} else {
		fmt.Printf("ID: %s\n", doc.ID)
		fmt.Printf("Title: %s\n", doc.Title)
		fmt.Printf("URL: %s\n", doc.URL)
		fmt.Printf("Content: %s\n", doc.Content[:100]+"...")
	}

	// Тестируем парсинг всей папки
	fmt.Println("\n=== Тест парсинга папки ===")
	docs, err := parser.ParseDirectory("data")
	if err != nil {
		log.Printf("Ошибка: %v", err)
	} else {
		fmt.Printf("Найдено документов: %d\n", len(docs))
		for _, doc := range docs {
			fmt.Printf("- %s (%s)\n", doc.Title, doc.ID)
		}
	}
}
