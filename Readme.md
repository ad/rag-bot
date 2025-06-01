# RAG Chat Bot

Telegram-бот для поиска и предоставления информации из базы знаний с использованием технологии RAG (Retrieval-Augmented Generation).

## Описание

RAG Chat Bot — это интеллектуальный Telegram-бот, который:

- 📚 Ищет релевантные документы в базе знаний по запросу пользователя
- 🤖 Использует LLM (Large Language Model) для генерации ответов на основе найденных документов
- 🔍 Обрабатывает запросы на русском и английском языках
- ⚡ Фильтрует стоп-слова и оптимизирует поиск
- 🐳 Полностью контейнеризован с Docker

## Архитектура

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Telegram Bot  │───▶│  RAG System  │───▶│   Ollama    │
│                 │    │              │    │   (LLM)     │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  Documents   │
                       │   (CSV)      │
                       └──────────────┘
```

## Технологический стек

- **Backend**: Go 1.24+
- **LLM**: Ollama (Gemma 2B)
- **Containerization**: Docker, Docker Compose
- **Bot Framework**: go-telegram/bot
- **Data Format**: CSV (заголовок, ссылка, ключевые фразы)

## Быстрый старт

### Предварительные требования

- Docker и Docker Compose
- Telegram Bot Token

### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/ragchat.git
cd ragchat
```

### 2. Настройка переменных окружения

Создайте файл `.env`:

```bash
cp .env.example .env
```

Заполните `.env`:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

### 3. Подготовка данных

Создайте файл `data/docs.csv` с вашей базой знаний:

```csv
Заголовок,Ссылка,Ключевые фразы
Как установить Docker,https://docs.docker.com/install,docker установка инсталляция контейнеры
Настройка Kubernetes,https://kubernetes.io/docs/setup,kubernetes k8s настройка кластер
Работа с Git,https://git-scm.com/docs,git версии контроль репозиторий коммит
```

### 4. Запуск

```bash
# Запуск всех сервисов
docker-compose up --build

# Или в фоновом режиме
docker-compose up -d --build
```

### 5. Использование

1. Найдите своего бота в Telegram
2. Отправьте любой вопрос, например: "Как установить Docker?"
3. Получите ответ с релевантной ссылкой

## Команды управления

### Основные команды

```bash
# Запуск сервисов
docker-compose up --build

# Остановка сервисов
docker-compose down

# Просмотр логов
docker-compose logs -f

# Просмотр логов конкретного сервиса
docker-compose logs -f rag-bot
docker-compose logs -f ollama
```

### Управление моделями

```bash
# Список загруженных моделей
docker exec -it ollama ollama list

# Загрузка новой модели
docker exec -it ollama ollama pull llama2:7b

# Удаление модели
docker exec -it ollama ollama rm gemma:2b
```

### Отладка

```bash
# Проверка API Ollama
curl http://localhost:11434/api/tags

# Тест генерации
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma:2b", "prompt": "Hello", "stream": false}'

# Подключение к контейнеру
docker exec -it rag-bot sh
docker exec -it ollama sh
```

## Конфигурация

### Структура проекта

```
ragchat/
├── cmd/
│   └── main.go              # Главный файл приложения
├── internal/
│   ├── llm/
│   │   └── llm.go          # LLM клиент для Ollama
│   └── retrieval/
│       └── retrieval.go    # Система поиска документов
├── data/
│   └── docs.csv            # База знаний
├── docker-compose.yml      # Конфигурация сервисов
├── Dockerfile              # Образ для бота
├── .env                    # Переменные окружения
└── README.md
```

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `TELEGRAM_BOT_TOKEN` | Токен Telegram бота | - |
| `LLM_API_URL` | URL API Ollama | `http://ollama:11434` |

### Настройки поиска

В файле `internal/retrieval/retrieval.go` можно настроить:

- Стоп-слова для фильтрации
- Веса для разных типов совпадений
- Алгоритм ранжирования результатов

### Настройки LLM

В файле `internal/llm/llm.go` можно изменить:

- Модель (`gemma:2b`, `llama2:7b`, etc.)
- Параметры генерации (temperature, top_k, top_p)
- Промпты для генерации ответов

## Устранение неполадок

### Бот не отвечает

1. Проверьте токен бота:
   ```bash
   echo $TELEGRAM_BOT_TOKEN
   ```

2. Проверьте логи:
   ```bash
   docker-compose logs rag-bot
   ```

### Модель не загружается

1. Проверьте статус Ollama:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Перезагрузите модель:
   ```bash
   docker exec -it ollama ollama pull gemma:2b
   ```

### Медленные ответы

1. Используйте более легкую модель:
   ```bash
   docker exec -it ollama ollama pull gemma:2b
   ```

2. Уменьшите `num_predict` в настройках LLM

### Проблемы с памятью

1. Добавьте ограничения в `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 4G
   ```

## Разработка

### Локальная разработка

```bash
# Установка зависимостей
go mod download

# Запуск только Ollama
docker-compose up ollama

# Запуск бота локально
export LLM_API_URL=http://localhost:11434
export TELEGRAM_BOT_TOKEN=your_token
go run cmd/main.go
```

### Добавление новых возможностей

1. **Новые типы документов**: Обновите структуру `Document` в `retrieval.go`
2. **Другие LLM**: Реализуйте интерфейс в `llm.go`
3. **Дополнительные команды**: Расширьте обработчики в `main.go`

## Лицензия

MIT License. См. файл [LICENSE](LICENSE) для подробностей.

## Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## Поддержка

Если у вас есть вопросы или проблемы:

1. Проверьте [Issues](https://github.com/yourusername/ragchat/issues)
2. Создайте новый Issue с детальным описанием
3. Укажите версию Docker, логи и шаги для воспроизведения