package main

import (
	"encoding/xml"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	llm "github.com/ad/rag-bot/internal/llm"
	"github.com/gocolly/colly/v2"
)

// Структура для парсинга sitemap.xml
type URLSet struct {
	XMLName xml.Name `xml:"urlset"`
	URLs    []URL    `xml:"url"`
}

type URL struct {
	Loc string `xml:"loc"`
}

func main() {
	// Параметры конфигурации
	maxPages := 0                   // Максимальное количество страниц для скачивания
	requestDelay := 1 * time.Second // Задержка между запросами (1 секунда)

	// Создаем папку для результатов
	outputDir := "data"
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatal("Ошибка создания директории:", err)
	}

	// Получаем все URL из sitemap.xml
	urls, err := getSitemapURLs("https://nethouse.ru/sitemap.xml")
	if err != nil {
		log.Fatal("Ошибка получения sitemap:", err)
	}

	// Фильтруем URL, которые начинаются с нужного префикса
	targetPrefix := "https://nethouse.ru/about/instructions/"
	var filteredURLs []string
	for _, url := range urls {
		if strings.HasPrefix(url, targetPrefix) {
			filteredURLs = append(filteredURLs, url)
		}
	}

	fmt.Printf("Найдено %d страниц для скачивания (ограничение: %d)\n", len(filteredURLs), maxPages)

	// Создаем коллектор для парсинга страниц
	c := colly.NewCollector(
		colly.AllowedDomains("nethouse.ru"),
	)

	// Добавляем rate limiter для снижения нагрузки на сервер
	c.Limit(&colly.LimitRule{
		DomainGlob:  "nethouse.ru",
		Parallelism: 1,            // Только один одновременный запрос
		Delay:       requestDelay, // Задержка между запросами
	})

	// Настраиваем User-Agent
	c.UserAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

	// Парсим каждую страницу
	c.OnHTML("html", func(e *colly.HTMLElement) {
		// Получаем h1
		h1 := e.ChildText("h1")
		if h1 == "" {
			h1 = "Заголовок не найден"
		}

		// Извлекаем HTML из div.help-article__main
		var articleHTML string
		if e.DOM.Find("div.help-article__main").Length() > 0 {
			articleHTML, _ = e.DOM.Find("div.help-article__main").Html()
		} else {
			articleHTML = ""
		}

		if strings.TrimSpace(articleHTML) == "" {
			articleHTML = "Содержимое не найдено"
		}

		// Формируем промпт для Ollama
		ollamaPrompt := `Проанализируй следующий HTML-документ.
Извлеки только важный и содержательный текст: факты, определения, инструкции, ключевые выводы.
Не добавляй вступлений, объяснений или комментариев.
Результат представь в виде простого, чистого текста без форматирования и выделения заголовков.
Не используй markdown и HTML для разметки.

HTML:
` + articleHTML

		// Инициализируем LLM-клиент
		llmEngine := llm.NewHTTPLLM(llm.GetApiURL())

		params := map[string]interface{}{
			"temperature":    0,
			"repeat_penalty": 1.1,
		}

		ollamaResult, err := llmEngine.GenerateResponse(ollamaPrompt, params)
		if err != nil {
			log.Printf("Ошибка Ollama: %v", err)
			ollamaResult = "Ошибка генерации выжимки: " + err.Error()
		}

		// Создаем содержимое markdown файла
		markdownContent := fmt.Sprintf("# %s\n\n**URL:** %s\n\n%s\n", h1, e.Request.URL.String(), ollamaResult)

		// Создаем имя файла из URL
		filename := createFilename(e.Request.URL.String()) + ".md"
		filePath := filepath.Join(outputDir, filename)

		// Сохраняем файл
		err = ioutil.WriteFile(filePath, []byte(markdownContent), 0644)
		if err != nil {
			log.Printf("Ошибка сохранения файла %s: %v", filename, err)
		} else {
			fmt.Printf("Сохранено: %s\n", filename)
		}
	})

	// Обрабатываем все отфильтрованные URL с ограничением
	processedCount := 0

	c.OnRequest(func(r *colly.Request) {
		fmt.Printf("Обрабатывается (%d/%d): %s\n", processedCount+1, len(filteredURLs), r.URL.String())
	})

	c.OnError(func(r *colly.Response, err error) {
		log.Printf("Ошибка при обработке %s: %v", r.Request.URL, err)
	})

	// Начинаем обход всех отфильтрованных URL
	for _, url := range filteredURLs {
		if maxPages > 0 && processedCount >= maxPages {
			fmt.Printf("Достигнуто максимальное количество страниц (%d)\n", maxPages)
			break
		}
		c.Visit(url)
		processedCount++
	}

	fmt.Printf("Парсинг завершен. Обработано %d страниц. Файлы сохранены в папку: %s\n", processedCount, outputDir)
}

// Функция для получения всех URL из sitemap.xml
func getSitemapURLs(sitemapURL string) ([]string, error) {
	resp, err := http.Get(sitemapURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var urlset URLSet
	err = xml.Unmarshal(body, &urlset)
	if err != nil {
		return nil, err
	}

	var urls []string
	for _, url := range urlset.URLs {
		urls = append(urls, url.Loc)
	}

	return urls, nil
}

// Функция для создания валидного имени файла из URL
func createFilename(url string) string {
	// Убираем протокол и домен
	filename := strings.ReplaceAll(url, "https://nethouse.ru/about/instructions/", "")

	// Заменяем слеши на подчеркивания
	filename = strings.ReplaceAll(filename, "/", "_")

	// Убираем недопустимые символы для имени файла
	reg := regexp.MustCompile(`[<>:"/\\|?*]`)
	filename = reg.ReplaceAllString(filename, "_")

	// Если имя файла пустое, используем случайное
	if filename == "" || filename == "_" {
		filename = "page"
	}

	return filename
}
