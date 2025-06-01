# 1. Билд-стейдж: собираем бинарник
FROM golang:1.24-alpine AS builder

WORKDIR /app

COPY go.* ./

RUN go mod download

COPY internal internal
COPY ratelimiter.go ratelimiter.go
COPY main.go main.go

# Собираем бинарник
RUN go build -o rag-bot .

# 2. Финальный образ — минимальный Alpine
FROM alpine:latest

WORKDIR /app

# Копируем бинарник из билд-стейджа
COPY --from=builder /app/rag-bot .
COPY data /app/data

# Пробрасываем порт (если нужен)
EXPOSE 8080

# Запуск бота
CMD ["./rag-bot"]