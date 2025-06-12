package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	_ "github.com/joho/godotenv/autoload"
	"golang.org/x/sync/singleflight"
)

type LLMEngine interface {
	GenerateResponse(prompt string, params map[string]interface{}) (string, error)
	GenerateEmbedding(text string) ([]float32, error)
	Answer(query string, docs []Document) (string, error)
}

func GetLLMModel() string {
	model := os.Getenv("LLM_MODEL")
	if model == "" {
		return "gemma3:1b"
	}
	return model
}

func GetLLMEmbeddingsModel() string {
	embedModel := os.Getenv("LLM_EMBEDDINGS_MODEL")
	if embedModel == "" {
		return "mxbai-embed-large"
	}
	return embedModel
}

func GetApiURL() string {
	apiURL := os.Getenv("LLM_API_URL")
	if apiURL == "" {
		return "http://localhost:11434"
	}
	return apiURL
}

type HTTPLLMEngine struct {
	apiURL     string
	client     *http.Client
	sf         singleflight.Group
	modelCache map[string]bool // кэш для проверки доступности моделей
	cacheMutex sync.RWMutex    // мьютекс для безопасного доступа к кэшу
}

func NewHTTPLLM(apiURL string) *HTTPLLMEngine {
	return &HTTPLLMEngine{
		apiURL: apiURL,
		client: &http.Client{
			Timeout: 600 * time.Second,
		},
		modelCache: make(map[string]bool),
	}
}

// ...existing structs...

func (h *HTTPLLMEngine) GenerateResponse(prompt string, params map[string]interface{}) (string, error) {
	modelName := GetLLMModel()

	// Проверяем доступность модели без лишнего логирования
	if err := h.ensureModelAvailableQuiet(modelName); err != nil {
		return "", fmt.Errorf("model not available: %w", err)
	}

	if params == nil {
		params = map[string]interface{}{
			"temperature":    0.7,
			"num_predict":    1024,
			"top_k":          40,
			"top_p":          0.95,
			"repeat_penalty": 1.1,
		}
	}

	// Подготовка запроса для Ollama
	reqBody := OllamaRequest{
		Model:   modelName,
		Prompt:  prompt,
		Stream:  false,
		Options: params,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("ошибка сериализации запроса: %w", err)
	}

	// Отправка запроса к Ollama API
	resp, err := h.client.Post(h.apiURL+"/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("ошибка HTTP запроса: %w", err)
	}
	defer resp.Body.Close()

	// Проверка статуса ответа
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HTTP ошибка: %d, ответ: %s", resp.StatusCode, string(bodyBytes))
	}

	// Чтение тела ответа
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("ошибка чтения ответа: %w", err)
	}

	// Парсинг ответа
	var respBody OllamaResponse
	if err := json.Unmarshal(bodyBytes, &respBody); err != nil {
		return "", fmt.Errorf("ошибка десериализации ответа: %w", err)
	}

	return respBody.Response, nil
}

// Проверка модели из кэша
func (h *HTTPLLMEngine) isModelCached(modelName string) bool {
	h.cacheMutex.RLock()
	defer h.cacheMutex.RUnlock()
	return h.modelCache[modelName]
}

// Обновление кэша модели
func (h *HTTPLLMEngine) cacheModel(modelName string, available bool) {
	h.cacheMutex.Lock()
	defer h.cacheMutex.Unlock()
	h.modelCache[modelName] = available
}

func (h *HTTPLLMEngine) checkModelAvailability(modelName string) error {
	resp, err := h.client.Get(h.apiURL + "/api/tags")
	if err != nil {
		return fmt.Errorf("failed to get models list: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("API error when getting models: status %d", resp.StatusCode)
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read models response: %w", err)
	}

	var modelsResp OllamaModelsResponse
	if err := json.Unmarshal(bodyBytes, &modelsResp); err != nil {
		return fmt.Errorf("failed to decode models response: %w", err)
	}

	// Проверяем, есть ли нужная модель в списке
	for _, model := range modelsResp.Models {
		if strings.Contains(model.Name, modelName) || model.Name == modelName {
			h.cacheModel(modelName, true)
			return nil
		}
	}

	return fmt.Errorf("model %s not found in available models", modelName)
}

type OllamaPullRequest struct {
	Name   string `json:"name"`
	Stream bool   `json:"stream"`
}
type OllamaPullResponse struct {
	Name      string `json:"name"`
	Status    string `json:"status"`
	Total     int    `json:"total"`
	Completed int    `json:"completed"`
}
type OllamaModelsResponse struct {
	Models []OllamaModel `json:"models"`
}
type OllamaModel struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Size        int    `json:"size"`
	CreatedAt   string `json:"created_at"`
	UpdatedAt   string `json:"updated_at"`
}

type OllamaRequest struct {
	Model    string                 `json:"model"`
	Prompt   string                 `json:"prompt"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
	System   string                 `json:"system,omitempty"`   // Для системных инструкций
	Template string                 `json:"template,omitempty"` // Для поддержки шаблонов
}
type OllamaResponse struct {
	Response string `json:"response"`
	Usage    struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

func (h *HTTPLLMEngine) pullModel(modelName string) error {
	fmt.Printf("Скачивание модели: %s\n", modelName)

	pullReq := OllamaPullRequest{
		Name:   modelName,
		Stream: true,
	}

	jsonData, err := json.Marshal(pullReq)
	if err != nil {
		return fmt.Errorf("failed to marshal pull request: %w", err)
	}

	resp, err := h.client.Post(h.apiURL+"/api/pull", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to send pull request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error when pulling model: status %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	// Читаем stream ответ для отслеживания прогресса
	decoder := json.NewDecoder(resp.Body)
	var lastStatus string
	var lastProgress float64

	for {
		var pullResp OllamaPullResponse
		if err := decoder.Decode(&pullResp); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("failed to decode pull response: %w", err)
		}

		// Выводим прогресс только при значительных изменениях
		if pullResp.Total > 0 && pullResp.Completed > 0 {
			percentage := float64(pullResp.Completed) / float64(pullResp.Total) * 100
			// Логируем прогресс только если изменение больше 10%
			if percentage-lastProgress >= 10.0 {
				fmt.Printf("Скачивание %s: %.0f%%\n", modelName, percentage)
				lastProgress = percentage
			}
		} else if pullResp.Status != lastStatus && pullResp.Status != "" {
			fmt.Printf("Модель %s: %s\n", modelName, pullResp.Status)
			lastStatus = pullResp.Status
		}
	}

	fmt.Printf("Модель %s скачана успешно\n", modelName)
	return nil
}

// Тихая проверка модели (без логирования)
func (h *HTTPLLMEngine) ensureModelAvailableQuiet(modelName string) error {
	// Проверяем кэш
	if h.isModelCached(modelName) {
		return nil
	}

	// Используем singleflight для предотвращения одновременного скачивания
	_, err, _ := h.sf.Do(modelName, func() (interface{}, error) {
		// Сначала проверяем, есть ли модель
		if err := h.checkModelAvailability(modelName); err == nil {
			return nil, nil
		}

		fmt.Printf("Модель %s не найдена, начинаем скачивание...\n", modelName)

		// Если модели нет, скачиваем её
		if err := h.pullModel(modelName); err != nil {
			return nil, fmt.Errorf("failed to download model %s: %w", modelName, err)
		}

		// Проверяем ещё раз после скачивания
		if err := h.checkModelAvailability(modelName); err != nil {
			return nil, fmt.Errorf("model %s still not available after download: %w", modelName, err)
		}

		return nil, nil
	})

	return err
}

// Document represents a document with header, link, and keywords
type Document struct {
	Header string
	Link   string
	Text   string
}

func (h *HTTPLLMEngine) Answer(query string, docs []Document) (string, error) {
	modelName := GetLLMModel()

	// Проверяем доступность модели без лишнего логирования
	if err := h.ensureModelAvailableQuiet(modelName); err != nil {
		return "", fmt.Errorf("model not available: %w", err)
	}

	// Формирование контекста из документов
	context := ""
	for i, doc := range docs {
		context += fmt.Sprintf("ДОКУМЕНТ %d:\nЗАГОЛОВОК: %s\nССЫЛКА: %s\nТЕКСТ: %s\n\n",
			i+1, doc.Header, doc.Link, doc.Text)
	}

	// Подготовка запроса для Ollama
	reqBody := OllamaRequest{
		Model:  modelName,
		Stream: false,
		Prompt: fmt.Sprintf("ВОПРОС ПОЛЬЗОВАТЕЛЯ: %s\n\nКОНТЕКСТ:\n%s\n\nОТВЕТ:", context, query),
		System: `Ты - специалист технической поддержки компании Nethouse. Анализируй предоставленные документы и отвечай на вопросы пользователей.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из подходящего документа
2. Если в документах есть хотя бы частичная информация - дай ответ на основе этой информации
3. Указывай ССЫЛКУ на источник
4. Не задавай вопросы, не используй фразы "я не знаю" или "не могу ответить"
5. Не используй форматирование

ФОРМАТ ОТВЕТА:
- Прямой ответ на вопрос
- Конкретные шаги или инструкции

НЕ ОТКАЗЫВАЙСЯ отвечать если есть хоть какая-то релевантная информация в документах.`,
		Options: map[string]interface{}{
			"temperature":    0.3,
			"num_predict":    800,
			"top_k":          20,
			"top_p":          0.8,
			"repeat_penalty": 1.3,
			// "stop":           []string{"Вопрос:", "ДОКУМЕНТ"},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// Отправка запроса к Ollama API
	resp, err := h.client.Post(h.apiURL+"/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Проверка статуса ответа
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API error: status %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	// Чтение тела ответа
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	// Парсинг ответа
	var respBody OllamaResponse
	if err := json.Unmarshal(bodyBytes, &respBody); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	// Post-processing: проверяем наличие источников и добавляем их при необходимости
	response := respBody.Response
	removeWords := []string{"[ЗАГОЛОВОК]: ", "ЗАГОЛОВОК: ", "[ССЫЛКА]: ", "ССЫЛКА: ", "[Источник]: ", "**Источник:** ", "[СОДЕРЖАНИЕ]:", "СОДЕРЖАНИЕ:", "Прямой ответ на вопрос: "}
	for _, word := range removeWords {
		response = strings.ReplaceAll(response, word, "")
	}

	if response == "" {
		return "Пожалуйста, уточните вопрос или напишите на support@nethouse.ru", nil
	}

	return response, nil
}

func (h *HTTPLLMEngine) GenerateEmbedding(text string) ([]float32, error) {
	client := NewOllamaClient()
	return client.GenerateEmbedding(text)
}

type OllamaClient struct {
	httpEngine *HTTPLLMEngine
}

func NewOllamaClient() *OllamaClient {
	return &OllamaClient{
		httpEngine: NewHTTPLLM(GetApiURL()),
	}
}

// EmbeddingRequest альтернативная структура запроса
type EmbeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// EmbeddingResponse структура ответа от Ollama API
type EmbeddingResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}

func (c *OllamaClient) GenerateEmbedding(text string) ([]float32, error) {
	// Проверяем входной текст
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("входной текст пустой")
	}

	// Проверяем доступность модели БЕЗ логирования
	if err := c.httpEngine.ensureModelAvailableQuiet(GetLLMEmbeddingsModel()); err != nil {
		return nil, fmt.Errorf("model not available: %w", err)
	}

	request := EmbeddingRequest{
		Model: GetLLMEmbeddingsModel(),
		Input: text,
	}

	reqBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("ошибка сериализации запроса: %w", err)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Post(GetApiURL()+"/api/embed", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("ошибка HTTP запроса: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP ошибка: %d, ответ: %s", resp.StatusCode, string(body))
	}

	// Читаем ответ
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ошибка чтения ответа: %w", err)
	}

	var response EmbeddingResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("ошибка десериализации ответа: %w, тело ответа: %s", err, string(body))
	}

	// Проверяем, что есть хотя бы один эмбеддинг
	if len(response.Embeddings) == 0 {
		return nil, fmt.Errorf("API вернул пустой массив эмбеддингов")
	}

	// Возвращаем первый эмбеддинг
	if len(response.Embeddings[0]) == 0 {
		return nil, fmt.Errorf("эмбеддинг пустой")
	}

	return response.Embeddings[0], nil
}
