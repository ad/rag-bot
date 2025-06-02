package main

import (
	"fmt"
	"log"

	"github.com/ad/rag-bot/internal/llm"
)

func main() {
	client := llm.NewOllamaClient("http://localhost:11434", "gemma3:1b")

	fmt.Println("Тестируем генерацию эмбеддингов...")

	embedding, err := client.GenerateEmbedding("Тестовый текст для эмбеддинга")
	if err != nil {
		log.Printf("Ошибка: %v", err)
		return
	}

	fmt.Printf("Размер эмбеддинга: %d\n", len(embedding))
	if len(embedding) == 0 {
		log.Println("Эмбеддинг пустой, проверьте работу LLM сервера.")
		return
	}

	fmt.Printf("Первые 5 значений: %v\n", embedding[:5])
}
