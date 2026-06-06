#!/bin/bash
# Quick start script for Medical AI Suite Dev Container
# Usage: bash .devcontainer/quick-start.sh

set -e

echo "🚀 Medical AI Suite - Dev Container Quick Start"
echo "=============================================="
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop."
    echo "   https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo "✓ Docker found"

if ! command -v code &> /dev/null; then
    echo "⚠️  VS Code not found in PATH (optional)"
fi
echo "✓ Prerequisites checked"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "📝 Creating .env from template..."
    cp .devcontainer/.env.example .env
    echo "✓ .env created - please update with your GROQ_API_KEY"
    echo ""
fi

# Build container
echo "🔨 Building dev container..."
echo "   (This may take a few minutes on first run)"
docker-compose -f .devcontainer/docker-compose.yml build --no-cache
echo "✓ Container built successfully"
echo ""

# Start services
echo "🚀 Starting services..."
docker-compose -f .devcontainer/docker-compose.yml up -d
echo "✓ Services started"
echo ""

# Wait for PostgreSQL
echo "⏳ Waiting for PostgreSQL to be ready..."
max_attempts=30
attempt=1
while [ $attempt -le $max_attempts ]; do
    if docker-compose -f .devcontainer/docker-compose.yml exec postgres \
        pg_isready -U postgres &>/dev/null; then
        echo "✓ PostgreSQL is ready"
        break
    fi
    if [ $attempt -eq $max_attempts ]; then
        echo "❌ PostgreSQL failed to start"
        exit 1
    fi
    printf "."
    sleep 1
    attempt=$((attempt + 1))
done
echo ""

# Show next steps
echo "✨ Setup complete!"
echo ""
echo "📌 Next Steps:"
echo ""
echo "1. Open VS Code:"
echo "   code ."
echo ""
echo "2. Reopen in container (when prompted, or Ctrl+Shift+P → Reopen in Container)"
echo ""
echo "3. Or use the available services:"
echo ""
echo "   🎨 Streamlit App:"
echo "      make streamlit"
echo "      → http://localhost:8501"
echo ""
echo "   📊 Jupyter Lab:"
echo "      make jupyter"
echo "      → http://localhost:8888"
echo ""
echo "   💾 PostgreSQL:"
echo "      make db-shell"
echo ""
echo "4. Common commands:"
echo "   make help       - Show all available commands"
echo "   make test       - Run tests"
echo "   make format     - Format code"
echo "   make lint       - Check code quality"
echo ""
echo "🔗 Useful links:"
echo "   Project README:  ./README.md"
echo "   Dev Container:   ./.devcontainer/README.md"
echo "   Makefile:        ./Makefile"
echo ""
echo "Happy coding! 🎉"
