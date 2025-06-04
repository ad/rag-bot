package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/ad/rag-bot/internal/cache"
	"github.com/ad/rag-bot/internal/llm"
	"github.com/ad/rag-bot/internal/parser"
	"github.com/ad/rag-bot/internal/retrieval"
	"github.com/ad/rag-bot/internal/vectorstore"

	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"

	_ "github.com/joho/godotenv/autoload"
)

func main() {
	rateLimiter := NewRateLimiter()

	// 1. Сначала инициализируем LLM
	llmEngine := llm.NewHTTPLLM(llm.GetApiURL())

	// 2. Инициализируем векторную систему и кэш
	fmt.Println("Инициализация векторной системы...")
	markdownParser := parser.NewMarkdownParser()
	vectorStore := vectorstore.NewVectorStore()
	embeddingCache := cache.NewEmbeddingCache("cache/embeddings.json")

	// 3. Загружаем и обрабатываем документы
	documents, err := markdownParser.ParseDirectory("data")
	if err != nil {
		log.Fatalf("Ошибка загрузки документов: %v", err)
	}

	fmt.Printf("Загружено документов: %d\n", len(documents))

	if len(documents) == 0 {
		log.Fatal("Не найдено документов для обработки в папке data/")
	}

	// Показываем статистику кэша
	cacheStats, err := embeddingCache.GetCacheStats()
	if err != nil {
		log.Printf("Ошибка получения статистики кэша: %v", err)
	} else {
		fmt.Printf("В кэше найдено эмбеддингов: %d\n", cacheStats)
	}

	fmt.Println("Генерация эмбеддингов...")

	// 4. Генерируем эмбеддинги для всех документов с использованием кэша
	successCount := 0
	cacheHits := 0
	cacheUpdates := 0

	for i, doc := range documents {
		if i%10 == 0 {
			fmt.Printf("Обработано %d/%d документов (кэш: %d попаданий, %d новых)\n",
				i, len(documents), cacheHits, cacheUpdates)

			embeddingCache.FlushCache() // Сбрасываем кэш каждые 10 документов
		}

		text := doc.Title + "\n" + doc.Content
		if strings.TrimSpace(text) == "" {
			log.Printf("Пропуск документа %s: пустое содержимое", doc.ID)
			continue
		}

		// Сначала пытаемся загрузить из кэша
		if cachedEmbedding, found := embeddingCache.GetEmbedding(doc); found {
			documents[i].Embedding = cachedEmbedding
			successCount++
			cacheHits++
			continue
		}

		// Если в кэше нет, генерируем новый эмбеддинг
		embedding, err := llmEngine.GenerateEmbedding(text)
		if err != nil {
			log.Printf("Ошибка генерации эмбеддинга для %s: %v", doc.ID, err)
			continue
		}

		if len(embedding) == 0 {
			log.Printf("Получен пустой эмбеддинг для документа %s", doc.ID)
			continue
		}

		// Сохраняем в документ
		documents[i].Embedding = embedding
		successCount++
		cacheUpdates++

		// Сохраняем в кэш
		if err := embeddingCache.SetEmbedding(doc, embedding); err != nil {
			log.Printf("Ошибка сохранения эмбеддинга в кэш для %s: %v", doc.ID, err)
		}
	}

	if successCount == 0 {
		log.Fatal("Не удалось сгенерировать эмбеддинги ни для одного документа")
	} else {
		embeddingCache.FlushCache() // Сбрасываем кэш каждые 10 документов
	}

	vectorStore.AddDocuments(documents)
	fmt.Printf("Инициализация завершена. Документов с эмбеддингами в хранилище: %d\n", successCount)
	fmt.Printf("Статистика кэша: %d попаданий, %d новых эмбеддингов\n", cacheHits, cacheUpdates)

	// ...existing code для телеграм бота...
	// 5. Создаем retrieval engine
	retrievalEngine := retrieval.NewVectorRetrieval(vectorStore, llmEngine)

	// 6. Запуск Telegram-бота
	tgToken := os.Getenv("TELEGRAM_BOT_TOKEN")
	if tgToken == "" {
		log.Fatal("TELEGRAM_BOT_TOKEN is not set")
	}

	opts := []bot.Option{
		bot.WithSkipGetMe(),
		bot.WithDefaultHandler(func(ctx context.Context, b *bot.Bot, update *models.Update) {
			if update.Message == nil {
				return
			}

			userID := update.Message.From.ID

			// Rate limiting
			if !rateLimiter.Allow(userID) {
				_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
					ChatID: update.Message.Chat.ID,
					Text:   "Слишком много запросов. Подождите ответа на предыдущий запрос.",
				})
				return
			}

			query := update.Message.Text
			log.Printf("Received message: %s from chat ID: %d", query, update.Message.Chat.ID)

			if query == "" || len(query) > 1000 {
				_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
					ChatID: update.Message.Chat.ID,
					Text:   "Пожалуйста, введите корректный запрос (до 1000 символов).",
				})
				return
			}

			if strings.TrimSpace(query) == "" {
				return
			}

			// Показываем индикатор печати
			_, _ = b.SendChatAction(ctx, &bot.SendChatActionParams{
				ChatID: update.Message.Chat.ID,
				Action: models.ChatActionTyping,
			})

			// Ищем документы
			docs, err := retrievalEngine.FindRelevantDocuments(query, 5)
			if err != nil {
				log.Printf("Ошибка поиска документов: %v", err)
				_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
					ChatID: update.Message.Chat.ID,
					Text:   "Ошибка при поиске документов.",
				})
				return
			}

			if len(docs) == 0 {
				_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
					ChatID: update.Message.Chat.ID,
					Text:   "Не найдено подходящих документов по вашему запросу.",
				})
				return
			}

			// Конвертируем в формат для llm.Answer()
			var llmDocs []llm.Document
			for _, doc := range docs {
				llmDoc := llm.Document{
					Header: doc.Title,
					Link:   doc.URL,
					Text:   doc.Content,
				}
				llmDocs = append(llmDocs, llmDoc)
			}

			log.Printf("Found %d documents for query: %s, %+v", len(llmDocs), query, llmDocs)

			// Генерируем ответ
			response, err := llmEngine.Answer(query, llmDocs)
			if err != nil {
				log.Printf("Ошибка генерации ответа: %v", err)
				response = "Ошибка при генерации ответа."
			}

			_, err = b.SendMessage(ctx, &bot.SendMessageParams{
				ChatID: update.Message.Chat.ID,
				Text:   truncateText(response, 4000),
			})

			log.Println("Ответ:", truncateText(response, 4000))

			if err != nil {
				log.Printf("Ошибка отправки сообщения: %v", err)
			} else {
				log.Printf("Ответ отправлен в чат ID: %d", update.Message.Chat.ID)
			}
		}),
	}

	b, err := bot.New(tgToken, opts...)
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	log.Println("Bot started...")
	if me, err := b.GetMe(ctx); err != nil {
		log.Fatalf("Failed to get bot info: %v", err)
	} else {
		log.Printf("Waiting for messages on @%s (ID: %d)", me.Username, me.ID)
	}

	b.Start(ctx)
}

// ...existing code для остальных функций...
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Безопасное обрезание текста
func truncateText(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen]
}
