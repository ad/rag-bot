package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ad/rag-bot/internal/retrieval"
	"golang.org/x/sync/singleflight"
)

func getLLMModel() string {
	model := os.Getenv("LLM_MODEL")
	if model == "" {
		return "smollm2:135m" // значение по умолчанию
	}
	return model
}

type HTTPLLMEngine struct {
	apiURL string
	client *http.Client
	sf     singleflight.Group // для предотвращения одновременных запросов на скачивание
}

func NewHTTPLLM(apiURL string) *HTTPLLMEngine {
	return &HTTPLLMEngine{
		apiURL: apiURL,
		client: &http.Client{
			Timeout: 600 * time.Second,
		},
	}
}

// Структуры для Ollama API
type OllamaRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Stream  bool                   `json:"stream"`
	Options map[string]interface{} `json:"options,omitempty"`
}

type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

type OllamaModelsResponse struct {
	Models []OllamaModel `json:"models"`
}

type OllamaModel struct {
	Name     string `json:"name"`
	Size     int64  `json:"size"`
	Digest   string `json:"digest"`
	Modified string `json:"modified_at"`
}

// Структура для запроса на скачивание модели
type OllamaPullRequest struct {
	Name   string `json:"name"`
	Stream bool   `json:"stream"`
}

// Структура ответа при скачивании модели
type OllamaPullResponse struct {
	Status          string `json:"status"`
	Digest          string `json:"digest,omitempty"`
	Total           int64  `json:"total,omitempty"`
	Completed       int64  `json:"completed,omitempty"`
	CompletedLength int64  `json:"completed_length,omitempty"`
}

func (h *HTTPLLMEngine) checkModelAvailability(modelName string) error {
	// Получаем список доступных моделей
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
			fmt.Printf("Model %s found and available\n", modelName)
			return nil
		}
	}

	return fmt.Errorf("model %s not found in available models", modelName)
}

func (h *HTTPLLMEngine) pullModel(modelName string) error {
	fmt.Printf("Starting to download model: %s\n", modelName)

	pullReq := OllamaPullRequest{
		Name:   modelName,
		Stream: true, // используем stream для отслеживания прогресса
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

	for {
		var pullResp OllamaPullResponse
		if err := decoder.Decode(&pullResp); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("failed to decode pull response: %w", err)
		}

		// Выводим прогресс только при изменении статуса
		if pullResp.Status != lastStatus {
			fmt.Printf("Model %s: %s\n", modelName, pullResp.Status)
			lastStatus = pullResp.Status
		}

		// Если есть информация о прогрессе, выводим её
		if pullResp.Total > 0 && pullResp.Completed > 0 {
			percentage := float64(pullResp.Completed) / float64(pullResp.Total) * 100
			fmt.Printf("Model %s download progress: %.1f%%\n", modelName, percentage)
		}
	}

	fmt.Printf("Model %s downloaded successfully\n", modelName)
	return nil
}

func (h *HTTPLLMEngine) ensureModelAvailable(modelName string) error {
	// Используем singleflight для предотвращения одновременного скачивания
	_, err, _ := h.sf.Do(modelName, func() (interface{}, error) {
		// Сначала проверяем, есть ли модель
		if err := h.checkModelAvailability(modelName); err == nil {
			return nil, nil
		}

		fmt.Printf("Model %s not found, attempting to download...\n", modelName)

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

func (h *HTTPLLMEngine) waitForModel(modelName string, maxWaitTime time.Duration) error {
	start := time.Now()

	// Сначала пытаемся обеспечить доступность модели
	if err := h.ensureModelAvailable(modelName); err != nil {
		return err
	}

	// Затем ждём, пока модель станет полностью готова
	for time.Since(start) < maxWaitTime {
		if err := h.checkModelAvailability(modelName); err == nil {
			return nil
		}

		fmt.Printf("Model %s not ready yet, waiting...\n", modelName)
		time.Sleep(5 * time.Second)
	}

	return fmt.Errorf("model %s not available after %v", modelName, maxWaitTime)
}

func (h *HTTPLLMEngine) Answer(query string, docs []retrieval.Document) (string, error) {
	start := time.Now()
	defer func() {
		fmt.Printf("LLM request took: %v\n", time.Since(start))
	}()

	modelName := getLLMModel()

	// Проверяем доступность модели с ожиданием до 10 минут (для скачивания)
	fmt.Printf("Ensuring model %s is available...\n", modelName)
	if err := h.waitForModel(modelName, 10*time.Minute); err != nil {
		return "", fmt.Errorf("model not available: %w", err)
	}

	// ...existing code...
	// Формирование контекста из документов
	context := ""
	for i, doc := range docs {
		context += fmt.Sprintf("Документ %d:\nЗаголовок: %s\nСсылка: %s\nКлючевые фразы: %s\n\n",
			i+1, doc.Header, doc.Link, doc.Keywords)
	}

	// Формирование промпта
	prompt := fmt.Sprintf(`Ты — помощник, который анализирует список документов и выбирает самый подходящий для ответа на вопрос пользователя.
%s

Вопрос пользователя:
%s

Не добавляй пояснений.
Ответь только заголовком и ссылкой на один наиболее релевантный документ:
`, context, query)

	// Подготовка запроса для Ollama
	reqBody := OllamaRequest{
		Model:  modelName,
		Prompt: prompt,
		Stream: false,
		Options: map[string]interface{}{
			"temperature": 0.3,
			"num_predict": 512,
			"top_k":       40,
			"top_p":       0.95,
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// log request for debugging
	fmt.Printf("Request JSON: %s\n", jsonData)

	// Отправка запроса к Ollama API
	resp, err := h.client.Post(h.apiURL+"/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Проверка статуса ответа
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		fmt.Printf("Error response: %s\n", string(bodyBytes))
		return "", fmt.Errorf("API error: status %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	// Чтение тела ответа
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}
	fmt.Printf("Response body: %s\n", string(bodyBytes))

	// Парсинг ответа
	var respBody OllamaResponse
	if err := json.Unmarshal(bodyBytes, &respBody); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	return respBody.Response, nil
}
