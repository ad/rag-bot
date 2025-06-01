
BOT_DOCKERFILE = Dockerfile

# Образы
BOT_IMAGE = rag-bot

# Запуск docker-compose (в файле docker-compose.yml)
COMPOSE_FILE = docker-compose.yml

.PHONY: all build up down clean logs

all: download up

pull-model:
	# @docker exec -it ollama ollama pull llama2
	@docker exec -it ollama ollama pull llama2:7b
	@docker exec -it ollama ollama list

build:
	docker build -t $(BOT_IMAGE) -f $(BOT_DOCKERFILE) .

up:
	docker-compose -f $(COMPOSE_FILE) up -d --build

down:
	docker-compose -f $(COMPOSE_FILE) down

clean: down
	docker rmi $(BOT_IMAGE) || true
	docker system prune -f

logs:
	docker-compose -f $(COMPOSE_FILE) logs -f

