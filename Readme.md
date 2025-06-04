# RAG Chat Bot

Telegram-бот для поиска и предоставления информации из базы знаний с использованием технологии RAG (Retrieval-Augmented Generation).

## Описание

RAG Chat Bot — это интеллектуальный Telegram-бот, который:

- 📚 Ищет релевантные документы в базе знаний по запросу пользователя
- 🤖 Использует LLM (Large Language Model) для генерации ответов на основе найденных документов
- 🔍 Обрабатывает запросы на русском и английском языках
- ⚡ Фильтрует стоп-слова и оптимизирует поиск
- 🐳 Полностью контейнеризован с Docker

## Особенности проекта

### ⚡ Производительность
- **Кэширование эмбеддингов**: Векторные представления сохраняются в `cache/embeddings.json`
- **Rate Limiting**: Встроенная защита от чрезмерной нагрузки
- **Модульная архитектура**: Четкое разделение ответственности между компонентами

### 🛠 Инструменты разработчика
- **Утилиты тестирования**: 4 готовые утилиты для отладки
- **Загрузчик контента**: Автоматическое извлечение контента с веб-сайтов
- **Makefile**: Удобные команды для управления проектом

### 🔧 Гибкость настроек
- Поддержка различных LLM моделей через Ollama
- Настраиваемые алгоритмы поиска и ранжирования
- Модульная система интеграций

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
                       │    (.md)     │
                       └──────────────┘
```

## Технологический стек

- **Backend**: Go 1.24+
- **LLM**: Ollama (gemma3:1b)
- **Containerization**: Docker, Docker Compose
- **Bot Framework**: go-telegram/bot v1.15.0
- **Web Scraping**: gocolly/colly v2.2.0
- **Data Format**: Markdown (заголовок, ссылка, текст статьи)
- **Rate Limiting**: Встроенный ограничитель скорости

## Быстрый старт

### Предварительные требования

- Docker и Docker Compose (или Ollama https://ollama.com/download)
- Telegram Bot Token

### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/rag-bot.git
cd rag-bot
```

### 2. Настройка переменных окружения

Создайте файл `.env` из примера:

```bash
cp .env.example .env
```

Заполните `.env`:

```env
# Telegram Bot Token (получите у @BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Внутренний URL API (не изменяйте в Docker)
LLM_API_URL=http://ollama:11434
```

### 3. Запуск сервисов в Docker

```bash
# Запуск всех сервисов
docker-compose up --build

# Или в фоновом режиме
docker-compose up -d --build
```

### 4. Локальный запуск (опционально)
Если вы хотите запустить бота локально без Docker (например, в Mac OS не поддерживается работа с GPU), отредактируйте .env, установите и запустите ollama (https://ollama.com/download), установите зависимости и запустите:

```bash
# Убедитесь, что Ollama запущен
ollama serve

# Установка зависимостей
go mod download

# Запуск бота
go run .
```

### 5. Использование

1. Найдите своего бота в Telegram (при запуске его username будет виден в логах)
2. Отправьте вопрос, ответ на который должен быть в базе знаний
3. Получите релевантный ответ

## Команды управления

### Основные команды

```bash
# Запуск сервисов
docker-compose up --build

# Или с помощью Makefile
make up

# Остановка сервисов
docker-compose down
# или
make down

# Просмотр логов
docker-compose logs -f
# или
make logs

# Просмотр логов конкретного сервиса
docker-compose logs -f rag-bot
docker-compose logs -f ollama
```

### Makefile команды

Проект включает удобные команды через Makefile:

```bash
# Запуск всех сервисов
make up

# Остановка сервисов
make down

# Сборка образа бота
make build

# Загрузка модели LLM
make pull-model

# Просмотр логов
make logs

# Полная очистка (остановка + удаление образов)
make clean
```

### Управление моделями

```bash
# Список загруженных моделей
docker exec -it ollama ollama list

# Загрузка новой модели
docker exec -it ollama ollama pull llama2:7b

# Удаление модели
docker exec -it ollama ollama rm gemma3:1b
```

### Отладка

```bash
# Проверка API Ollama
curl http://localhost:11434/api/tags

# Тест генерации
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3:1b", "prompt": "Hello", "stream": false}'

# Подключение к контейнеру
docker exec -it rag-bot sh
docker exec -it ollama sh
```

## Конфигурация

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `TELEGRAM_BOT_TOKEN` | Токен Telegram бота | - |
| `LLM_API_URL` | URL API Ollama | `http://ollama:11434` |
| `LLM_MODEL` | Модель языковой модели | `gemma3:1b` |
| `LLM_LLM_EMBEDDINGS_MODEL` | Модель векторизации | `mxbai-embed-large` |
| `OLLAMA_CONTEXT_LENGTH` | Длина контекста | `4096` |

### Настройка модели

Вы можете использовать различные модели LLM:

```bash
# Запуск с моделью по умолчанию
docker-compose up

# Запуск с другой моделью
LLM_MODEL=llama2:7b docker-compose up

# Или создайте .env файл
echo "LLM_MODEL=mistral:7b" > .env
docker-compose up
```

### Доступные модели

Примеры поддерживаемых моделей:
- `gemma3:1b` (по умолчанию, быстрая)
- `llama2:7b`
- `mistral:7b`

Полный список доступных моделей можно найти на [Ollama Library](https://ollama.ai/library).


### Структура проекта

```
rag-bot/
├── cmd/                             # Утилиты и инструменты
│   ├── downloader/
│   │   └── main.go                  # Загрузчик контента с веб-сайтов
│   ├── parser/
│   │   └── main.go                  # Парсер Markdown документов
│   ├── llm_embeddings_test/
│   │   └── main.go                  # Тест генерации эмбеддингов
│   └── vectorstore_test/
│       └── main.go                  # Тест векторного хранилища
├── internal/                        # Внутренние модули
│   ├── cache/                       # Кэширование данных
│   ├── llm/                         # LLM клиент для Ollama
│   ├── parser/                      # Парсер документов
│   ├── retrieval/                   # Система поиска документов
│   ├── types/                       # Общие типы данных
│   └── vectorstore/                 # Векторное хранилище
├── data/                            # База знаний
│   ├── avtomatizatsiya.md
│   ├── budet_li_na_moem_sajte_reklama.md
│   └── ...                          # Статьи в формате Markdown
├── cache/
│   └── embeddings.json              # Кэш векторных представлений
├── main.go                          # Главный файл Telegram бота
├── ratelimiter.go                   # Ограничитель скорости запросов
├── docker-compose.yml              # Конфигурация сервисов
├── Dockerfile                       # Образ для бота
├── Makefile                         # Команды сборки и управления
├── go.mod                           # Зависимости Go
├── go.sum                           # Контрольные суммы зависимостей
├── .env.example                     # Пример переменных окружения
├── .env                            # Переменные окружения (создать)
├── .gitignore                       # Исключения для Git
├── LICENSE                          # Лицензия MIT
└── README.md
```

### Утилиты разработки

В папке `cmd/` находятся вспомогательные утилиты для разработки и тестирования:

#### downloader
Утилита для автоматической загрузки контента с веб-сайтов по sitemap.xml:

```bash
# Запуск загрузчика
go run cmd/downloader/main.go
```

Функциональность:
- Парсинг sitemap.xml для получения списка страниц
- Автоматическое извлечение контента с веб-страниц
- Сохранение в формате Markdown
- Настройка максимального количества страниц

#### parser
Утилита для тестирования парсера Markdown документов:

```bash
# Тест парсера
go run cmd/parser/main.go
```

Функциональность:
- Проверка корректности парсинга документов из папки `data/`
- Вывод статистики найденных документов
- Валидация структуры документов

#### llm_embeddings_test
Утилита для тестирования генерации векторных представлений:

```bash
# Тест эмбеддингов
go run cmd/llm_embeddings_test/main.go
```

Функциональность:
- Проверка подключения к Ollama
- Тестирование генерации эмбеддингов
- Вывод размерности векторов
- Диагностика проблем с LLM

#### vectorstore_test
Комплексная утилита для тестирования всей RAG системы:

```bash
# Полный тест RAG системы
go run cmd/vectorstore_test/main.go
```

Функциональность:
- Тестирование парсинга документов
- Генерация и сохранение эмбеддингов
- Проверка векторного поиска
- Интеграционное тестирование компонентов

### Настройки поиска

В файле `internal/retrieval/retrieval.go` можно настроить:

- Стоп-слова для фильтрации
- Веса для разных типов совпадений
- Алгоритм ранжирования результатов

### Настройки LLM

В файле `internal/llm/llm.go` можно изменить:

- Модель (`gemma3:1b`, `llama2:7b`, etc.)
- Параметры генерации (temperature, top_k, top_p)
- Промпты для генерации ответов (в том числе системный)

### Rate Limiting

Проект включает встроенный ограничитель скорости (`ratelimiter.go`) для предотвращения чрезмерной нагрузки на web-сервер при скачивании документов.

## Устранение неполадок

### Бот не отвечает

1. Проверьте токен бота:
   ```bash
   echo $TELEGRAM_BOT_TOKEN
   ```

2. Проверьте логи:
   ```bash
   docker-compose logs -f
   ```

### Модель не загружается

1. Проверьте статус Ollama:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Перезагрузите модель:
   ```bash
   docker exec -it ollama ollama pull gemma3:1b
   ```

### Медленные ответы

1. Используйте более легкую модель:
   ```bash
   docker exec -it ollama ollama pull gemma3:1b
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
go run .

# Или с помощью Makefile
make up  # запуск всех сервисов
make down  # остановка
```

### Добавление новых возможностей

1. **Новые типы документов**: Обновите структуры в `internal/types/` и логику парсинга в `internal/parser/`
2. **Другие LLM**: Реализуйте интерфейс в `internal/llm/`
3. **Дополнительные команды**: Расширьте обработчики в `main.go`
4. **Кэширование**: Используйте модуль `internal/cache/` для оптимизации производительности
5. **Rate Limiting**: Настройте параметры в `ratelimiter.go`

## Лицензия

MIT License. Copyright (c) 2025 Daniel Apatin. См. файл [LICENSE](LICENSE) для подробностей.
