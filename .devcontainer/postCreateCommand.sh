#!/bin/bash
set -e

echo "🚀 Setting up Medical AI Suite Dev Container..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}📦 Installing core dependencies...${NC}"
pip install -q --upgrade pip setuptools wheel

echo -e "${BLUE}📚 Installing project dependencies...${NC}"
if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Requirements installed${NC}"
else
    echo -e "${YELLOW}⚠ requirements.txt not found${NC}"
fi

echo -e "${BLUE}🛠️  Installing development tools...${NC}"
pip install -q \
    pytest \
    pytest-cov \
    jupyter \
    jupyterlab \
    black \
    isort \
    pylint \
    flake8 \
    mypy
echo -e "${GREEN}✓ Development tools installed${NC}"

echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p /workspace/logs
mkdir -p /workspace/models
mkdir -p /workspace/data
mkdir -p /workspace/.venv
echo -e "${GREEN}✓ Directories created${NC}"

echo -e "${BLUE}✅ Setting up environment variables...${NC}"
if [ ! -f /workspace/.env ]; then
    echo -e "${YELLOW}⚠ .env file not found in workspace${NC}"
    echo "Please ensure your .env file is set up with:"
    echo "  - DB_HOST=postgres"
    echo "  - DB_PORT=5432"
    echo "  - DB_NAME=medical_ai"
    echo "  - DB_USER=postgres"
    echo "  - DB_PASSWORD=postgres_dev_password"
    echo "  - GROQ_API_KEY=your_groq_api_key"
fi

echo -e "${BLUE}🔍 Verifying PostgreSQL connection...${NC}"
if command -v psql &> /dev/null; then
    for i in {1..30}; do
        if PGPASSWORD=postgres_dev_password psql -h postgres -U postgres -d medical_ai -c "SELECT 1" &>/dev/null; then
            echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${YELLOW}⚠ PostgreSQL not ready after 30 seconds${NC}"
        fi
        sleep 1
    done
else
    echo -e "${YELLOW}⚠ psql command not found${NC}"
fi

echo -e "${BLUE}🎯 Initializing Git...${NC}"
git config --global --add safe.directory /workspace

echo -e "${GREEN}✨ Dev container setup complete!${NC}"
echo ""
echo -e "${BLUE}Quick Start:${NC}"
echo "  • Start Streamlit app: streamlit run app/main.py"
echo "  • Start Jupyter Lab: jupyter lab --ip=0.0.0.0"
echo "  • Run tests: pytest"
echo "  • Format code: black ."
echo "  • Check imports: isort ."
echo ""
