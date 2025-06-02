package types

import (
	"crypto/md5"
	"fmt"
)

type Document struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	URL       string    `json:"url"`
	Content   string    `json:"content"`
	Embedding []float32 `json:"embedding,omitempty"`
}

// GetContentHash возвращает MD5 хеш содержимого документа для проверки изменений
func (d *Document) GetContentHash() string {
	content := d.Title + "\n" + d.Content
	hash := md5.Sum([]byte(content))
	return fmt.Sprintf("%x", hash)
}
