// cmd/main.go
package main

import (
	"context"
	"encoding/csv"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/ad/rag-bot/internal/llm"
	"github.com/ad/rag-bot/internal/retrieval"

	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
)

func main() {
	rateLimiter := NewRateLimiter()

	docsFile := "data/docs.csv"

	// Загрузка документов
	file, err := os.Open(docsFile)
	if err != nil {
		log.Fatalf("failed to open docs file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("failed to read CSV: %v", err)
	}

	var docs []retrieval.Document
	// Пропускаем заголовок (первую строку)
	for i, record := range records {
		if i == 0 {
			continue // Пропускаем заголовок
		}
		if len(record) >= 3 {
			docs = append(docs, retrieval.Document{
				Header:   record[0],
				Link:     record[1],
				Keywords: strings.Join(strings.Split(record[2], ","), " "),
			})
		}
	}

	log.Printf("Loaded %d documents from CSV", len(docs))

	// Инициализация поисковика
	retriever := retrieval.NewRetriever(docs)

	// Получение URL API LLM из переменной окружения
	llmAPIURL := os.Getenv("LLM_API_URL")
	if llmAPIURL == "" {
		// log.Fatal("LLM_API_URL is not set")
		llmAPIURL = "http://localhost:11434"
	}

	// Инициализация LLM с HTTP API
	llmEngine := llm.NewHTTPLLM(llmAPIURL)

	// Запуск Telegram-бота
	tgToken := os.Getenv("TELEGRAM_BOT_TOKEN")
	if tgToken == "" {
		log.Fatal("TELEGRAM_BOT_TOKEN is not set")
	}

	opts := []bot.Option{
		bot.WithDefaultHandler(func(ctx context.Context, b *bot.Bot, update *models.Update) {
			// Проверяем, что сообщение не nil
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

				// answer with typing effect
				_, _ = b.SendChatAction(ctx, &bot.SendChatActionParams{
					ChatID: update.Message.Chat.ID,
					Action: models.ChatActionTyping,
				})

				return
			}

			// запись в лог запроса
			log.Printf("Received message: %s from chat ID: %d", update.Message.Text, update.Message.Chat.ID)
			if update.Message.Text == "" {
				_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
					ChatID: update.Message.Chat.ID,
					Text:   "Пожалуйста, введите ваш запрос.",
				})
				return
			}

			// Поиск документов по запросу
			query := update.Message.Text

			if len(query) > 1000 { // Ограничиваем длину
				_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
					ChatID: update.Message.Chat.ID,
					Text:   "Сообщение слишком длинное. Максимум 1000 символов.",
				})
				return
			}

			if strings.TrimSpace(query) == "" {
				return
			}

			docs := retriever.Search(query, 5)
			if len(docs) == 0 {
				_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
					ChatID: update.Message.Chat.ID,
					Text:   "Не найдено подходящих документов по вашему запросу.",
				})

				return
			}

			// answer with typing effect
			_, _ = b.SendChatAction(ctx, &bot.SendChatActionParams{
				ChatID: update.Message.Chat.ID,
				Action: models.ChatActionTyping,
			})

			// Запись в лог найденных документов
			// Генерация ответа с использованием LLM
			log.Printf("Generating response for query: %s", query)
			log.Printf("Found %d documents for query: %s", len(docs), query)
			log.Printf("Documents: %+v", docs)
			// Формирование ответа
			log.Printf("Sending request to LLM API at %s", llmAPIURL)
			log.Printf("Query: %s", query)
			response, err := llmEngine.Answer(query, docs)
			if err != nil {
				log.Printf("generation error: %v", err)
				response = "Ошибка при генерации ответа."
			}
			_, errSendMessage := b.SendMessage(ctx, &bot.SendMessageParams{
				ChatID: update.Message.Chat.ID,
				Text:   response,
				// ParseMode: models.ParseModeMarkdown,
			})

			if errSendMessage != nil {
				log.Printf("error sending message: %v", errSendMessage)
				_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
					ChatID: update.Message.Chat.ID,
					Text:   "Произошла ошибка при отправке ответа.",
				})
			} else {
				log.Printf("Response sent successfully to chat ID: %d", update.Message.Chat.ID)
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
	b.Start(ctx)
}
