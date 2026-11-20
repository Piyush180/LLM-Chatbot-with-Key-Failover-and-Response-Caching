.PHONY: up down test

ENV_CHECK = $(if $(wildcard .env),,$(error .env not found - copy .env.example to .env and add your keys))

## Build and start backend (:8000) and frontend (:3000)
up:
	$(ENV_CHECK)
	docker compose up -d --build

## Stop the stack (the app-data volume, and so the SQLite history, is kept)
down:
	$(ENV_CHECK)
	docker compose down

## Run the backend test suite inside Docker
test:
	docker build --target test -t llm-chatbot-backend-test ./backend
	docker run --rm llm-chatbot-backend-test
