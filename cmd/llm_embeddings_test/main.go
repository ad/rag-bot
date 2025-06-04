package main

import (
	"fmt"
	"log"

	"github.com/ad/rag-bot/internal/llm"
)

func main() {
	client := llm.NewOllamaClient()

	fmt.Println("Тестируем генерацию эмбеддингов...")

	embedding, err := client.GenerateEmbedding("Тестовый текст для эмбеддинга")
	if err != nil {
		log.Printf("Ошибка: %v", err)
		return
	}

	count := len(embedding)

	if count == 0 {
		log.Println("Эмбеддинг пустой, проверьте работу LLM сервера.")
		return
	}

	fmt.Printf("Размер эмбеддинга: %d\n", count)

	testNum := min(3, count) // Ограничиваем вывод первыми 3 значениями для краткости

	fmt.Printf("Первые %d значения: %v\n", testNum, embedding[:testNum])
}
