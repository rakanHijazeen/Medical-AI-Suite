.PHONY: help build up down logs shell restart clean format lint test coverage docs

DOCKER_COMPOSE := docker-compose -f .devcontainer/docker-compose.yml
CONTAINER_NAME := medical_ai_dev
POSTGRES_CONTAINER := medical_ai_postgres

help:
	@echo "Medical AI Suite Development Commands"
	@echo "======================================"
	@echo ""
	@echo "Container Management:"
	@echo "  make build              Build dev container"
	@echo "  make up                 Start all services"
	@echo "  make down               Stop all services"
	@echo "  make restart            Restart all services"
	@echo "  make shell              Open shell in app container"
	@echo "  make logs               View container logs"
	@echo ""
	@echo "Database:"
	@echo "  make db-shell           Connect to PostgreSQL"
	@echo "  make db-reset           Reset database (⚠️ destructive)"
	@echo ""
	@echo "Development:"
	@echo "  make format             Format code with Black & isort"
	@echo "  make lint               Run linters (Pylint, Flake8)"
	@echo "  make type-check         Run type checking (mypy)"
	@echo "  make test               Run pytest suite"
	@echo "  make coverage           Run tests with coverage report"
	@echo ""
	@echo "Running Apps:"
	@echo "  make streamlit          Start Streamlit app"
	@echo "  make jupyter            Start Jupyter Lab"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean              Remove cache files and artifacts"
	@echo "  make freeze             Update requirements.txt"
	@echo "  make install            Install dependencies"
	@echo ""

# Container Management
build:
	$(DOCKER_COMPOSE) build --no-cache

up:
	$(DOCKER_COMPOSE) up -d
	@echo "✓ Services started"
	@echo "  Streamlit: http://localhost:8501"
	@echo "  Jupyter:   http://localhost:8888"
	@echo "  PostgreSQL: localhost:5432"

down:
	$(DOCKER_COMPOSE) down

restart:
	$(DOCKER_COMPOSE) restart
	@echo "✓ Services restarted"

shell:
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) /bin/bash

logs:
	$(DOCKER_COMPOSE) logs -f $(CONTAINER_NAME)

logs-db:
	$(DOCKER_COMPOSE) logs -f $(POSTGRES_CONTAINER)

# Database
db-shell:
	$(DOCKER_COMPOSE) exec $(POSTGRES_CONTAINER) \
		psql -U postgres -d medical_ai

db-reset:
	@echo "⚠️  WARNING: This will delete all database data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) down -v; \
		$(DOCKER_COMPOSE) up -d; \
		echo "✓ Database reset"; \
	else \
		echo "Cancelled"; \
	fi

# Development
format:
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) black .
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) isort .
	@echo "✓ Code formatted"

lint:
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) pylint app/ --disable=R,C
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) flake8 app/

type-check:
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) mypy app/

test:
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) pytest -v

coverage:
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) \
		pytest --cov=app --cov-report=html --cov-report=term-missing
	@echo "✓ Coverage report generated in htmlcov/"

# Running Apps
streamlit:
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) \
		streamlit run app/main.py --logger.level=debug

jupyter:
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) \
		jupyter lab --ip=0.0.0.0 --allow-root

# Utilities
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name ".DS_Store" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✓ Cleaned up artifacts"

freeze:
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) \
		pip freeze > requirements.txt
	@echo "✓ requirements.txt updated"

install:
	$(DOCKER_COMPOSE) exec $(CONTAINER_NAME) \
		pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# View help
.DEFAULT_GOAL := help
