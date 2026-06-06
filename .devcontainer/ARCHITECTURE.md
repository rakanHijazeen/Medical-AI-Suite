# Dev Container Architecture & Details

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VS Code Host Machine                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ workspace files (mounted via volumes)                │   │
│  │ Medical_AI_Suite/                                    │   │
│  │ ├── app/                                             │   │
│  │ ├── notebooks/                                       │   │
│  │ ├── models/                                          │   │
│  │ ├── data/                                            │   │
│  │ └── ...                                              │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                            ↕️
                    [Docker Desktop]
                            ↕️
┌──────────────────────────────────────────────────────────────┐
│                   Docker Compose Network                      │
│  medical_ai_network (bridge)                                 │
│                                                              │
│  ┌─────────────────────┐  ┌──────────────────┐             │
│  │   app Container     │  │  postgres        │             │
│  │ (Medical AI Suite)  │  │  Container       │             │
│  │                     │  │                  │             │
│  │ Python 3.11         │  │ PostgreSQL 16    │             │
│  │ Streamlit 1.57.0    │  │ medical_ai DB    │             │
│  │ PyTorch 2.12.0      │  │                  │             │
│  │ All ML packages     │  │ Port: 5432       │             │
│  │                     │  │                  │             │
│  │ Ports:              │  │ Health Check:    │             │
│  │ - 8501 (Streamlit)  │  │ pg_isready ✓     │             │
│  │ - 8888 (Jupyter)    │  │                  │             │
│  │                     │  └──────────────────┘             │
│  └─────────────────────┘                                    │
│         ↕️ (depends_on)                                      │
│    Host Ports:                                              │
│    8501 → Streamlit                                         │
│    8888 → Jupyter                                           │
│    5432 → PostgreSQL                                        │
│    5433 → PostgreSQL (alt)                                  │
└──────────────────────────────────────────────────────────────┘
```

## 📦 Container Specifications

### App Container (`medical_ai_dev`)

- **Base Image**: `python:3.11-slim`
- **Size**: ~2.5GB (after dependencies)
- **Key Packages**:
  - Streamlit 1.57.0
  - PyTorch 2.12.0 (CPU)
  - scikit-learn 1.8.0
  - transformers 5.9.0
  - sentence-transformers 5.5.1
  - pandas 3.0.2
  - numpy 2.4.4
  - All tools in requirements.txt

### PostgreSQL Container (`medical_ai_postgres`)

- **Base Image**: `postgres:16-alpine`
- **Version**: PostgreSQL 16
- **Database**: medical_ai
- **Port**: 5432
- **Volume**: `postgres_data` (persistent storage)
- **Health Check**: `pg_isready` every 10 seconds

### Jupyter Container (Optional)

- **Base Image**: Same as app container
- **Port**: 8888
- **Features**:
  - JupyterLab (not classic Jupyter)
  - All ML packages available
  - Access to workspace files
  - PostgreSQL accessible

## 🔄 Volume Mounts

| Volume        | Host Path     | Container Path             | Purpose        | Persistent |
| ------------- | ------------- | -------------------------- | -------------- | ---------- |
| workspace     | `./`          | `/workspace`               | Project files  | Via host   |
| postgres_data | Docker volume | `/var/lib/postgresql/data` | DB storage     | ✓ Yes      |
| jupyter_data  | Docker volume | `/root/.jupyter`           | Jupyter config | ✓ Yes      |
| .venv         | Docker volume | `/workspace/.venv`         | Python cache   | ✓ Yes      |

## 🌐 Network Configuration

- **Network Type**: `bridge`
- **Network Name**: `medical_ai_network`
- **Service Discovery**: DNS by service name (app, postgres, jupyter)

Service communication:

- App → PostgreSQL: `postgres:5432`
- Jupyter → PostgreSQL: `postgres:5432`
- Host → App: `localhost:8501`
- Host → PostgreSQL: `localhost:5432`

## 🔐 Security Considerations

### Development Environment

- Running as root (acceptable for dev, not production)
- PostgreSQL dev credentials hardcoded (change for prod)
- No encryption on containers (dev only)
- Groq API key should be in `.env` (excluded from git)

### For Production Conversion

1. Create non-root user in Dockerfile
2. Use secrets management (Docker Secrets, Kubernetes)
3. Enable PostgreSQL SSL/TLS
4. Implement network policies
5. Add resource limits (CPU, memory)
6. Use read-only file systems where possible

## 📊 Performance Characteristics

### Build Time

- **First build**: 5-10 minutes (downloads ~2.5GB)
- **Subsequent builds**: 30-60 seconds (cached layers)
- **Layer caching**: Enabled for all stages

### Runtime Performance

- **Container startup**: 5-10 seconds
- **Streamlit startup**: 15-30 seconds
- **ML model loading**: 2-5 seconds
- **PostgreSQL startup**: 2-5 seconds

### Memory Usage

- **App container**: 2-4 GB (depends on loaded models)
- **PostgreSQL**: 200-500 MB (small database)
- **Jupyter**: 1-2 GB (when running notebooks)
- **Total**: ~3-6 GB recommended

## 🔧 Customization Guide

### Adding System Dependencies

Edit `Dockerfile`:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    existing-packages \
    new-package-name \
    && rm -rf /var/lib/apt/lists/*
```

### Adding Python Packages

Option 1 - Update `requirements.txt`:

```bash
pip install new-package
pip freeze > requirements.txt
docker-compose build --no-cache
```

Option 2 - Direct installation:

```bash
pip install new-package
# (saved only in running container, lost on restart)
```

### Increasing Resource Limits

Edit `docker-compose.yml`:

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G
```

### Changing Ports

Edit `docker-compose.yml`:

```yaml
ports:
  - "9000:8501" # Streamlit on 9000
  - "9888:8888" # Jupyter on 9888
```

### Adding More Services

Example - Redis cache:

```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  networks:
    - medical_ai_network
```

## 🐛 Common Issues & Solutions

### Issue: "Docker daemon not running"

```bash
# Windows: Start Docker Desktop
# Mac: Start Docker Desktop
# Linux: sudo systemctl start docker
```

### Issue: "Port 8501 is already in use"

```bash
# Find what's using port 8501
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Then either:
# 1. Kill the process
# 2. Change port in docker-compose.yml
```

### Issue: "Container runs out of memory"

```bash
# Increase Docker Desktop resources:
# Settings → Resources → Memory slider

# Or set container limits in docker-compose.yml
```

### Issue: "PostgreSQL won't start"

```bash
# Check logs
docker logs medical_ai_postgres

# Reset database
docker volume rm medical_ai_postgres
docker-compose up -d postgres
```

### Issue: "Slow model loading in container"

```bash
# Increase virtual memory/disk space
# Models are cached in container after first load
# Subsequent runs are faster
```

## 📈 Advanced Usage

### Debugging in Container

```bash
# Start debugger breakpoint listener
# VSCode: Run → Start Debugging (launch.json configured)

# Or use pdb
python -m pdb app/main.py
```

### Running Background Tasks

```bash
# Inside container
screen          # Start screen session
# Run long task
Ctrl+A, D       # Detach from screen

# Later, reattach
screen -r       # Reattach to session
```

### Database Snapshots

```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres medical_ai > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres medical_ai < backup.sql
```

### Multi-container Debugging

```bash
# View all containers
docker-compose ps

# View specific container logs
docker-compose logs postgres -f

# Execute command in container
docker-compose exec app python -c "print('hello')"
```

## 🔍 Monitoring & Logging

### View Container Logs

```bash
docker-compose logs app           # App logs
docker-compose logs postgres      # DB logs
docker-compose logs -f            # Follow all logs
```

### Container Stats

```bash
docker stats medical_ai_dev       # CPU, memory, network
```

### PostgreSQL Queries

```bash
docker-compose exec postgres \
  psql -U postgres -d medical_ai \
  -c "SELECT * FROM medical_ai.evaluation_logs LIMIT 5;"
```

## 📚 Additional Resources

- [Docker Official Docs](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Python Docker Best Practices](https://docs.docker.com/language/python/build-images/)
- [VS Code Remote Development](https://code.visualstudio.com/docs/remote/remote-overview)

---

**Last updated**: 2026-06-06
