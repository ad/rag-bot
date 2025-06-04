package main

import (
	"fmt"
	"log"
	"os"

	"github.com/ad/rag-bot/internal/parser"
)

func main() {
	// Создаем папку если её нет
	if err := os.MkdirAll("data", 0755); err != nil {
		log.Fatal(err)
	}

	parser := parser.NewMarkdownParser()

	// парсинг всей папки
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
