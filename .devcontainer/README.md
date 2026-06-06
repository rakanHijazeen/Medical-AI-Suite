# Dev Container Setup for Medical AI Suite

This directory contains all necessary files for setting up a fully containerized development environment for the Medical AI Suite using VS Code's Remote - Containers extension.

## 📋 Overview

The dev container setup includes:

- **Python 3.11** runtime environment
- **PostgreSQL 16** database service
- **Jupyter Lab** for notebook development
- **Streamlit** development server
- All project dependencies pre-configured
- Development tools (pytest, black, isort, etc.)

## 🚀 Quick Start

### Prerequisites

1. **Docker Desktop** (Windows/Mac) or Docker Engine (Linux)
   - [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
2. **VS Code** with Remote - Containers extension
   - [Install Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Setup Steps

1. **Open project in VS Code**

   ```bash
   code Medical_AI_Suite
   ```

2. **Reopen in Dev Container**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Search for: **Remote-Containers: Reopen in Container**
   - VS Code will build the container and initialize your environment

3. **Wait for initialization**
   - The first build takes 5-10 minutes (downloading images, installing dependencies)
   - Subsequent builds are much faster (cached layers)

4. **Verify setup**
   ```bash
   python --version          # Should show Python 3.11.x
   psql --version           # Should show PostgreSQL client
   pip list | grep streamlit # Verify Streamlit is installed
   ```

## 📁 File Structure

```
.devcontainer/
├── devcontainer.json       # VS Code dev container configuration
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Orchestrate multiple services
├── postCreateCommand.sh    # Setup script after container creation
├── init-db.sql            # PostgreSQL initialization
├── .dockerignore           # Files to exclude from Docker builds
└── README.md              # This file
```

## 🔧 Configuration Files

### `devcontainer.json`

Main configuration for VS Code integration. Defines:

- Container service (`app`)
- Docker Compose setup
- VS Code extensions to install
- Port forwarding
- Environment variables
- Python path and settings

### `Dockerfile`

Builds the development container with:

- Python 3.11-slim base image
- System dependencies (PostgreSQL client, build tools)
- Python development tools (jupyter, black, pytest, etc.)
- Pre-installed scientific packages (numpy, pandas, scikit-learn, torch)

### `docker-compose.yml`

Orchestrates three services:

- **app**: Main development container
- **postgres**: PostgreSQL database (port 5432)
- **jupyter**: Jupyter Lab server (port 8888)

### `postCreateCommand.sh`

Runs after container is created:

- Upgrades pip and setuptools
- Installs project dependencies
- Creates necessary directories
- Verifies PostgreSQL connectivity
- Sets up git configuration

### `init-db.sql`

PostgreSQL initialization script:

- Creates `medical_ai` schema
- Sets up evaluation logs table
- Creates knowledge base table for RAG
- Creates audit trail table
- Creates indexes for performance
- Enables necessary PostgreSQL extensions

## 🚢 Running Services

### Start Streamlit App

```bash
cd /workspace
streamlit run app/main.py --logger.level=debug
```

Access at: **http://localhost:8501**

### Start Jupyter Lab

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Access at: **http://localhost:8888**

### Access PostgreSQL

```bash
# From inside container
psql -h postgres -U postgres -d medical_ai

# From host machine
psql -h localhost -p 5432 -U postgres -d medical_ai
```

## 🔌 Port Mappings

| Service        | Container Port | Host Port | Purpose              |
| -------------- | -------------- | --------- | -------------------- |
| Streamlit      | 8501           | 8501      | Web UI               |
| Jupyter        | 8888           | 8888      | Notebook development |
| PostgreSQL     | 5432           | 5432      | Database             |
| PostgreSQL Alt | 5432           | 5433      | Alternate host port  |

## 🔐 Environment Variables

The container automatically sets:

```
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=/workspace
DB_HOST=postgres
DB_PORT=5432
DB_NAME=medical_ai
DB_USER=postgres
DB_PASSWORD=postgres_dev_password
```

**Important**: The `GROQ_API_KEY` is not set by default. Add it:

- Copy `.env.example` to `.env` (if available)
- Or set it in `.env` in the workspace root
- The container will read it via Docker Compose

## 🧪 Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

### Code Formatting

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Lint with Pylint
pylint app/
```

### Type Checking

```bash
mypy app/
```

### Starting Python Interactive Shell

```bash
python
# or IPython
ipython
```

## 📚 Working with Jupyter Notebooks

1. **Via Jupyter Lab** (recommended):

   ```bash
   jupyter lab --ip=0.0.0.0 --no-browser
   ```

2. **Via VS Code**:
   - Open `.ipynb` file
   - Kernel auto-selects from container Python
   - Select "Python" kernel if prompted

3. **Notebooks in project**:
   - `notebooks/01_heart_training.ipynb`
   - `notebooks/02_diabetes_training.ipynb`
   - `notebooks/03_stroke_training.ipynb`
   - `notebooks/04_kidney_training.ipynb`

## 🐛 Troubleshooting

### Container won't start

```bash
# Check Docker daemon
docker --version
docker ps

# Rebuild container
# In VS Code: Remote-Containers: Rebuild Container
```

### PostgreSQL connection fails

```bash
# Verify PostgreSQL is running
docker ps | grep postgres

# Check logs
docker logs medical_ai_postgres

# Manually connect
PGPASSWORD=postgres_dev_password psql -h localhost -U postgres -d medical_ai
```

### Port conflicts

If ports 8501, 8888, or 5432 are in use:

1. Edit `docker-compose.yml`
2. Change host port mappings
3. Rebuild container

### Out of disk space

```bash
# Clean up unused images and containers
docker system prune -a --volumes
```

### Dependencies not installing

```bash
# Rebuild from scratch
docker system prune -a
# Then: Remote-Containers: Rebuild Container
```

## 🔄 Updating Dependencies

To update `requirements.txt` and install new packages:

1. **Add package to requirements.txt**
2. **Rebuild container**:
   - `Ctrl+Shift+P` → Remote-Containers: Rebuild Container
3. **Or install directly**:
   ```bash
   pip install package_name
   pip freeze > requirements.txt  # Update requirements file
   ```

## 💾 Data Persistence

- **PostgreSQL data**: Stored in `postgres_data` volume (persists after container restart)
- **Jupyter config**: Stored in `jupyter_data` volume
- **Workspace files**: Mounted from host at `/workspace` (live editing)

To reset PostgreSQL data:

```bash
docker volume rm medical_ai_postgres  # ⚠️ Destructive!
docker-compose down -v  # Remove all volumes
```

## 🎓 VS Code Extensions Included

The container automatically installs:

- **Python**: Language support and debugging
- **Pylance**: Advanced type checking
- **Jupyter**: Notebook integration
- **Ruff**: Fast Python linter
- **Black Formatter**: Code formatting
- **GitLens**: Git integration
- **Docker**: Container management
- **GitHub Copilot**: AI assistance

## 📖 Useful Commands

```bash
# View running containers
docker ps

# View container logs
docker logs medical_ai_dev

# Restart services
docker-compose restart

# Stop container
docker-compose down

# Start with rebuild
docker-compose up --build

# Execute command in running container
docker-compose exec app bash
```

## 🔗 Resources

- [VS Code Remote Containers](https://code.visualstudio.com/docs/remote/containers)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Python in Containers](https://docs.docker.com/language/python/)

## ✅ Verification Checklist

After setup, verify:

- [ ] Container builds successfully
- [ ] Python 3.11 is active
- [ ] All requirements installed (`pip list`)
- [ ] PostgreSQL is accessible
- [ ] Streamlit app starts without errors
- [ ] Jupyter Lab is accessible
- [ ] VS Code extensions are loaded
- [ ] Workspace files are mounted

## 🆘 Getting Help

1. Check Docker and VS Code logs
2. Review container build output
3. Test services individually
4. Check `.env` file for required credentials (GROQ_API_KEY)

## 📝 Notes

- Container runs as root for simplicity in development
- PostgreSQL uses development password (change for production)
- First build is slower; subsequent builds use cached layers
- All Python packages are installed in container, not host
- Workspace files remain on host machine (safe)

---

**Happy coding! 🚀**
