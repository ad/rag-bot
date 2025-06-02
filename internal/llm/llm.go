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
}

func NewHTTPLLM(apiURL string) *HTTPLLMEngine {
	return &HTTPLLMEngine{
		apiURL: apiURL,
		client: &http.Client{
			Timeout: 300 * time.Second,
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

func (h *HTTPLLMEngine) waitForModel(modelName string, maxWaitTime time.Duration) error {
	start := time.Now()
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

	// Проверяем доступность модели с ожиданием до 2 минут
	fmt.Printf("Checking if model %s is available...\n", modelName)
	if err := h.waitForModel(modelName, 2*time.Minute); err != nil {
		return "", fmt.Errorf("model not available: %w", err)
	}

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
Ответь только заголовком и ссылкой на один наиболее релевантный документ в формате Markdown`, context, query)

	// Подготовка запроса для Ollama
	reqBody := OllamaRequest{
		Model:  modelName,
		Prompt: prompt,
		Stream: false,
		Options: map[string]interface{}{
			"temperature": 0.3,
			"num_predict": 200, // Уменьшили для быстрого ответа
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
