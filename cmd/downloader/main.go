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
	outputDir := "parsed_pages"
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

		// Получаем содержимое из div.help-article__main с сохранением структуры
		var content string
		e.ForEach("div.help-article__main", func(i int, el *colly.HTMLElement) {
			content = extractTextWithStructure(el)
		})

		if content == "" {
			content = "Содержимое не найдено"
		}

		// Создаем содержимое markdown файла
		markdownContent := fmt.Sprintf("# %s\n\n**URL:** %s\n\n%s\n", h1, e.Request.URL.String(), content)

		// Создаем имя файла из URL
		filename := createFilename(e.Request.URL.String()) + ".md"
		filePath := filepath.Join(outputDir, filename)

		// Сохраняем файл
		err := ioutil.WriteFile(filePath, []byte(markdownContent), 0644)
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

// Функция для извлечения текста с сохранением структуры
func extractTextWithStructure(e *colly.HTMLElement) string {
	var result strings.Builder

	// Обрабатываем каждый прямой дочерний элемент
	e.ForEach("> *", func(i int, el *colly.HTMLElement) {
		processElement(el, &result, 0)
	})

	// Если ничего не извлекли, пробуем более простой подход
	if result.Len() == 0 {
		return extractSimpleText(e)
	}

	return cleanText(result.String())
}

// Рекурсивная функция для обработки элементов
func processElement(el *colly.HTMLElement, result *strings.Builder, depth int) {
	tagName := el.Name

	// Получаем текст только этого элемента (без дочерних)
	ownText := getOwnText(el)

	switch tagName {
	case "h1", "h2", "h3", "h4", "h5", "h6":
		text := strings.TrimSpace(el.Text)
		if text != "" {
			level := strings.Repeat("#", getHeaderLevel(tagName))
			result.WriteString(level + " " + text + "\n\n")
		}
	case "p":
		text := strings.TrimSpace(el.Text)
		if text != "" {
			result.WriteString(text + "\n\n")
		}
	case "ul", "ol":
		// Обрабатываем списки
		result.WriteString("\n")
		el.ForEach("li", func(i int, li *colly.HTMLElement) {
			text := strings.TrimSpace(li.Text)
			if text != "" {
				if tagName == "ul" {
					result.WriteString("- " + text + "\n")
				} else {
					result.WriteString(fmt.Sprintf("%d. %s\n", i+1, text))
				}
			}
		})
		result.WriteString("\n")
	case "li":
		// Пропускаем, обрабатываются в ul/ol
		return
	case "div", "section", "article":
		// Добавляем текст, если есть
		if ownText != "" {
			result.WriteString(ownText + "\n\n")
		}
		// Рекурсивно обрабатываем дочерние элементы
		el.ForEach("> *", func(i int, child *colly.HTMLElement) {
			processElement(child, result, depth+1)
		})
	case "br":
		result.WriteString("\n")
	case "strong", "b":
		text := strings.TrimSpace(el.Text)
		if text != "" {
			result.WriteString("**" + text + "**")
		}
	case "em", "i":
		text := strings.TrimSpace(el.Text)
		if text != "" {
			result.WriteString("*" + text + "*")
		}
	case "a":
		text := strings.TrimSpace(el.Text)
		href := el.Attr("href")
		if text != "" {
			if href != "" {
				result.WriteString(fmt.Sprintf("[%s](%s)", text, href))
			} else {
				result.WriteString(text)
			}
		}
	case "img":
		// Игнорируем изображения
	case "code":
		// Игнорируем изображения
	case "pre":
		// Игнорируем изображения
	default:
		// Для остальных элементов просто извлекаем текст
		text := strings.TrimSpace(el.Text)
		if text != "" && !hasTextInChildren(el) {
			result.WriteString(text + "\n\n")
		} else if ownText != "" {
			result.WriteString(ownText + " ")
		}

		// Обрабатываем дочерние элементы
		el.ForEach("> *", func(i int, child *colly.HTMLElement) {
			processElement(child, result, depth+1)
		})
	}
}

// Получить только собственный текст элемента (без дочерних)
func getOwnText(el *colly.HTMLElement) string {
	fullText := el.Text

	// Убираем текст всех дочерних элементов
	el.ForEach("*", func(i int, child *colly.HTMLElement) {
		childText := child.Text
		fullText = strings.ReplaceAll(fullText, childText, "")
	})

	return strings.TrimSpace(fullText)
}

// Проверить, есть ли текст в дочерних элементах
func hasTextInChildren(el *colly.HTMLElement) bool {
	hasText := false
	el.ForEach("*", func(i int, child *colly.HTMLElement) {
		if strings.TrimSpace(child.Text) != "" {
			hasText = true
		}
	})
	return hasText
}

// Получить уровень заголовка
func getHeaderLevel(tagName string) int {
	switch tagName {
	case "h1":
		return 1
	case "h2":
		return 2
	case "h3":
		return 3
	case "h4":
		return 4
	case "h5":
		return 5
	case "h6":
		return 6
	default:
		return 1
	}
}

// Простое извлечение текста как запасной вариант
func extractSimpleText(e *colly.HTMLElement) string {
	var result strings.Builder

	// Проходим по всем текстовым узлам
	e.ForEach("p, div, h1, h2, h3, h4, h5, h6, li, span", func(i int, el *colly.HTMLElement) {
		text := strings.TrimSpace(el.Text)
		if text != "" && !isChildOf(el, "p, div, h1, h2, h3, h4, h5, h6, li") {
			result.WriteString(text + "\n\n")
		}
	})

	// Если и это не помогло, берем весь текст
	if result.Len() == 0 {
		return strings.TrimSpace(e.Text)
	}

	return result.String()
}

// Проверить, является ли элемент дочерним для указанных селекторов
func isChildOf(el *colly.HTMLElement, parentSelectors string) bool {
	// Простая проверка - есть ли родители с такими тегами
	parent := el.DOM.Parent()
	for parent.Length() > 0 {
		tagName := parent.Get(0).Data
		if strings.Contains(parentSelectors, strings.ToLower(tagName)) {
			return true
		}
		parent = parent.Parent()
	}
	return false
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

// Функция для очистки текста от лишних пробелов и переносов
func cleanText(text string) string {
	// Заменяем множественные переводы строк на двойные
	reg := regexp.MustCompile(`\n{3,}`)
	text = reg.ReplaceAllString(text, "\n\n")

	// Заменяем множественные пробелы на одинарные, но сохраняем переводы строк
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		reg := regexp.MustCompile(`[ \t]+`)
		lines[i] = reg.ReplaceAllString(strings.TrimSpace(line), " ")
	}

	text = strings.Join(lines, "\n")

	// Убираем пробелы в начале и конце
	text = strings.TrimSpace(text)

	return text
}
