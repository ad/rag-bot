package parser

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ad/rag-bot/internal/types"
)

type MarkdownParser struct{}

func NewMarkdownParser() *MarkdownParser {
	return &MarkdownParser{}
}

func (p *MarkdownParser) ParseDirectory(dirPath string) ([]types.Document, error) {
	var documents []types.Document

	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if filepath.Ext(path) == ".md" {
			doc, err := p.ParseFile(path)
			if err != nil {
				fmt.Printf("Ошибка парсинга файла %s: %v\n", path, err)
				return nil
			}
			documents = append(documents, doc)
		}

		return nil
	})

	return documents, err
}

func (p *MarkdownParser) ParseFile(filePath string) (types.Document, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return types.Document{}, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var title, url, content string
	var lines []string

	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return types.Document{}, err
	}

	// Парсим заголовок
	for i, line := range lines {
		if strings.HasPrefix(line, "# ") {
			title = strings.TrimPrefix(line, "# ")
			lines = lines[i+1:]
			break
		}
	}

	// Парсим URL
	urlRegex := regexp.MustCompile(`\*\*URL:\*\*\s+(.+)`)
	for i, line := range lines {
		if match := urlRegex.FindStringSubmatch(line); len(match) > 1 {
			url = strings.TrimSpace(match[1])
			lines = lines[i+1:]
			break
		}
	}

	content = strings.TrimSpace(strings.Join(lines, "\n"))

	// Заменяем html-ссылки на markdown-ссылки
	htmlLinkRegex := regexp.MustCompile(`<a\s+href="([^"]+)"[^>]*>(.*?)<\/a>`)
	content = htmlLinkRegex.ReplaceAllStringFunc(content, func(s string) string {
		matches := htmlLinkRegex.FindStringSubmatch(s)
		if len(matches) == 3 {
			return "[" + matches[2] + "](" + matches[1] + ")"
		}
		return s
	})

	id := strings.TrimSuffix(filepath.Base(filePath), ".md")

	return types.Document{
		ID:      id,
		Title:   title,
		URL:     url,
		Content: content,
	}, nil
}
