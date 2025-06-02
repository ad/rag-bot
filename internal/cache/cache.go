package cache

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/ad/rag-bot/internal/types"
)

type EmbeddingCache struct {
	cachePath string
	cache     map[string]CachedEmbedding
	mutex     sync.RWMutex
	loaded    bool
}

type CachedEmbedding struct {
	DocumentID  string    `json:"document_id"`
	ContentHash string    `json:"content_hash"`
	Embedding   []float32 `json:"embedding"`
	CreatedAt   time.Time `json:"created_at"`
}

type CacheData struct {
	Version    string            `json:"version"`
	CreatedAt  time.Time         `json:"created_at"`
	Embeddings []CachedEmbedding `json:"embeddings"`
}

func NewEmbeddingCache(cachePath string) *EmbeddingCache {
	return &EmbeddingCache{
		cachePath: cachePath,
		cache:     make(map[string]CachedEmbedding),
		loaded:    false,
	}
}

func (ec *EmbeddingCache) ensureCacheDir() error {
	dir := filepath.Dir(ec.cachePath)
	return os.MkdirAll(dir, 0755)
}

// loadCacheOnce загружает кэш только один раз при первом обращении
func (ec *EmbeddingCache) loadCacheOnce() error {
	ec.mutex.Lock()
	defer ec.mutex.Unlock()

	// Если уже загружен, ничего не делаем
	if ec.loaded {
		return nil
	}

	if err := ec.ensureCacheDir(); err != nil {
		return fmt.Errorf("failed to ensure cache directory: %w", err)
	}

	// Проверяем, существует ли файл кэша
	if _, err := os.Stat(ec.cachePath); os.IsNotExist(err) {
		fmt.Println("Файл кэша эмбеддингов не найден, будет создан новый")
		ec.loaded = true
		return nil
	}

	// Читаем файл кэша
	data, err := os.ReadFile(ec.cachePath)
	if err != nil {
		return fmt.Errorf("failed to read cache file: %w", err)
	}

	var cacheData CacheData
	if err := json.Unmarshal(data, &cacheData); err != nil {
		fmt.Printf("Ошибка парсинга кэша (будет пересоздан): %v\n", err)
		ec.loaded = true
		return nil
	}

	// Заполняем карту кэша
	for _, embedding := range cacheData.Embeddings {
		key := ec.getCacheKey(embedding.DocumentID, embedding.ContentHash)
		ec.cache[key] = embedding
	}

	ec.loaded = true
	fmt.Printf("Загружено %d эмбеддингов из кэша\n", len(ec.cache))
	return nil
}

// SaveCache сохраняет весь кэш в файл
func (ec *EmbeddingCache) SaveCache() error {
	ec.mutex.RLock()
	defer ec.mutex.RUnlock()

	if err := ec.ensureCacheDir(); err != nil {
		return fmt.Errorf("failed to ensure cache directory: %w", err)
	}

	// Конвертируем карту в массив
	embeddings := make([]CachedEmbedding, 0, len(ec.cache))
	for _, embedding := range ec.cache {
		embeddings = append(embeddings, embedding)
	}

	cacheData := CacheData{
		Version:    "1.0",
		CreatedAt:  time.Now(),
		Embeddings: embeddings,
	}

	// Сериализуем в JSON
	data, err := json.MarshalIndent(cacheData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal cache data: %w", err)
	}

	// Записываем во временный файл, затем перемещаем (атомарная операция)
	tempPath := ec.cachePath + ".tmp"
	if err := os.WriteFile(tempPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write temp cache file: %w", err)
	}

	if err := os.Rename(tempPath, ec.cachePath); err != nil {
		os.Remove(tempPath) // Очищаем временный файл при ошибке
		return fmt.Errorf("failed to move temp cache file: %w", err)
	}

	return nil
}

// GetEmbedding получает эмбеддинг из кэша
func (ec *EmbeddingCache) GetEmbedding(doc types.Document) ([]float32, bool) {
	// Загружаем кэш, если еще не загружен
	if err := ec.loadCacheOnce(); err != nil {
		fmt.Printf("Ошибка загрузки кэша: %v\n", err)
		return nil, false
	}

	ec.mutex.RLock()
	defer ec.mutex.RUnlock()

	key := ec.getCacheKey(doc.ID, doc.GetContentHash())
	if cached, exists := ec.cache[key]; exists {
		return cached.Embedding, true
	}

	return nil, false
}

// SetEmbedding сохраняет эмбеддинг в кэш (в памяти)
func (ec *EmbeddingCache) SetEmbedding(doc types.Document, embedding []float32) error {
	// Загружаем кэш, если еще не загружен
	if err := ec.loadCacheOnce(); err != nil {
		return fmt.Errorf("failed to load cache: %w", err)
	}

	ec.mutex.Lock()
	defer ec.mutex.Unlock()

	key := ec.getCacheKey(doc.ID, doc.GetContentHash())
	ec.cache[key] = CachedEmbedding{
		DocumentID:  doc.ID,
		ContentHash: doc.GetContentHash(),
		Embedding:   embedding,
		CreatedAt:   time.Now(),
	}

	return nil
}

// FlushCache сохраняет кэш на диск
func (ec *EmbeddingCache) FlushCache() error {
	return ec.SaveCache()
}

func (ec *EmbeddingCache) getCacheKey(documentID, contentHash string) string {
	return fmt.Sprintf("%s:%s", documentID, contentHash)
}

// GetCacheStats возвращает статистику кэша
func (ec *EmbeddingCache) GetCacheStats() (int, error) {
	if err := ec.loadCacheOnce(); err != nil {
		return 0, err
	}

	ec.mutex.RLock()
	defer ec.mutex.RUnlock()

	return len(ec.cache), nil
}

// ClearCache очищает кэш в памяти
func (ec *EmbeddingCache) ClearCache() {
	ec.mutex.Lock()
	defer ec.mutex.Unlock()

	ec.cache = make(map[string]CachedEmbedding)
}

// GetCacheSize возвращает размер кэша в памяти
func (ec *EmbeddingCache) GetCacheSize() int {
	ec.mutex.RLock()
	defer ec.mutex.RUnlock()

	return len(ec.cache)
}
