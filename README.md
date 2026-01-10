# Blob Storage

A Python library for uploading and managing documents in local blob storage. Supports both **Azurite** (Azure Storage Emulator) and **local filesystem** storage.

## Features

- 📤 Upload documents via CLI or REST API
- 📥 Download documents from storage
- 📋 List all stored documents
- 🗑️ Delete documents
- 📁 Create custom containers
- 🔄 Two storage backends: Azurite or Local Filesystem
- 🌐 **REST API** with FastAPI and interactive docs

## Project Structure

```
blob-storage/
├── src/
│   └── blob_storage/
│       ├── __init__.py      # Package exports
│       ├── config.py        # Configuration module
│       ├── storage.py       # Storage implementations
│       ├── cli.py           # CLI application
│       └── api.py           # FastAPI REST API
├── tests/
│   ├── __init__.py
│   ├── test_storage.py      # Storage unit tests
│   └── test_api.py          # API integration tests
├── pyproject.toml           # Project configuration
├── requirements.txt         # Core dependencies
├── requirements-dev.txt     # Development dependencies
├── env.example              # Environment template
└── README.md
```

## Installation

### From Source (Development)

```bash
# Clone the repository
cd blob-storage

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Dependencies Only

```bash
pip install -r requirements.txt
```

## Configuration

Copy `env.example` to `.env` and customize:

```bash
cp env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_MODE` | `azurite` | Storage backend: `azurite` or `local` |
| `AZURE_STORAGE_CONNECTION_STRING` | (Azurite default) | Connection string for Azure storage |
| `BLOB_CONTAINER_NAME` | `documents` | Default container name |
| `LOCAL_STORAGE_PATH` | `./local_blob_storage` | Path for local filesystem storage |

## Storage Options

### Option 1: Azurite (Azure Storage Emulator) - Recommended

```bash
# Install Azurite
npm install -g azurite

# Start Azurite
azurite --silent --location ./azurite-data --debug ./azurite-debug.log
```

### Option 2: Local Filesystem Storage

No additional setup required:

```bash
export STORAGE_MODE=local
```

---

## REST API Usage

### Start the API Server

```bash
# Start with local filesystem storage
blob-storage --mode local serve

# Or with custom host/port
blob-storage --mode local serve --host 127.0.0.1 --port 8080

# Enable auto-reload for development
blob-storage --mode local serve --reload
```

The API will be available at `http://localhost:8000` with:
- **Interactive docs**: http://localhost:8000/docs
- **OpenAPI spec**: http://localhost:8000/openapi.json

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health status |
| `GET` | `/status` | Storage status and configuration |
| `POST` | `/upload` | Upload a single file |
| `POST` | `/upload/multiple` | Upload multiple files |
| `GET` | `/files` | List all files |
| `GET` | `/files/{name}` | Download a file |
| `GET` | `/files/{name}/info` | Get file information |
| `DELETE` | `/files/{name}` | Delete a file |
| `POST` | `/containers/{name}` | Create a container |
| `GET` | `/containers/{name}/files` | List files in container |

### Example API Requests

#### Upload a file

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

#### Upload with custom name

```bash
curl -X POST "http://localhost:8000/upload?blob_name=reports/2024/doc.pdf" \
  -F "file=@document.pdf"
```

#### Upload multiple files

```bash
curl -X POST "http://localhost:8000/upload/multiple" \
  -F "files=@file1.pdf" \
  -F "files=@file2.pdf"
```

#### List all files

```bash
curl "http://localhost:8000/files"
```

#### Download a file

```bash
curl -O "http://localhost:8000/files/document.pdf"
```

#### Get file info

```bash
curl "http://localhost:8000/files/document.pdf/info"
```

#### Delete a file

```bash
curl -X DELETE "http://localhost:8000/files/document.pdf"
```

#### Create a container

```bash
curl -X POST "http://localhost:8000/containers/my-container"
```

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Upload a file
with open("document.pdf", "rb") as f:
    response = requests.post(f"{BASE_URL}/upload", files={"file": f})
    print(response.json())

# List files
response = requests.get(f"{BASE_URL}/files")
print(response.json())

# Download a file
response = requests.get(f"{BASE_URL}/files/document.pdf")
with open("downloaded.pdf", "wb") as f:
    f.write(response.content)

# Delete a file
response = requests.delete(f"{BASE_URL}/files/document.pdf")
print(response.json())
```

---

## CLI Usage

### Check status

```bash
blob-storage --mode local status
```

### Upload a document

```bash
# Upload a single file
blob-storage upload ./path/to/document.pdf

# Upload with custom name
blob-storage upload ./document.pdf --name "reports/2024/annual-report.pdf"

# Upload to specific container
blob-storage upload ./document.pdf --container my-container
```

### Upload multiple documents

```bash
# Upload all files from a directory
blob-storage upload-dir ./documents/

# Upload with file pattern
blob-storage upload-dir ./documents/ --pattern "*.pdf"

# Upload recursively
blob-storage upload-dir ./documents/ --recursive
```

### List documents

```bash
# List all documents
blob-storage list

# List as JSON
blob-storage list --json-output
```

### Download a document

```bash
blob-storage download document.pdf ./downloaded.pdf
```

### Delete a document

```bash
blob-storage delete document.pdf --yes
```

### Start API server

```bash
blob-storage --mode local serve --port 8000
```

---

## Programmatic Usage

### Using the Storage Classes

```python
from blob_storage import get_blob_storage

# Get storage instance
storage = get_blob_storage("local")  # or "azurite"

# Upload a file
result = storage.upload_file("./document.pdf")
print(f"Uploaded: {result['blob_name']}")

# Upload raw data
data = b"Hello, World!"
result = storage.upload_data(data, "hello.txt")

# List all blobs
blobs = storage.list_blobs()
for blob in blobs:
    print(f"{blob['name']} - {blob['size']} bytes")

# Download a file
storage.download_file("document.pdf", "./downloaded.pdf")

# Check if blob exists
if storage.blob_exists("document.pdf"):
    print("Document exists!")

# Delete a blob
storage.delete_blob("document.pdf")
```

### Using the FastAPI App

```python
from blob_storage import create_app, run_server

# Get the FastAPI app instance
app = create_app()

# Or run the server directly
run_server(host="0.0.0.0", port=8000, reload=True)
```

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=blob_storage --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

### Code Formatting

```bash
# Format code
black src tests
isort src tests

# Lint code
ruff check src tests
mypy src
```

---

## Troubleshooting

### Azurite Connection Error

If you see connection errors when using Azurite mode:

1. Make sure Azurite is running:
   ```bash
   azurite --silent --location ./azurite-data
   ```

2. Check the default port (10000) is available

3. Verify the connection string in `.env`

### Switch to Local Storage

If Azurite is not available:

```bash
blob-storage --mode local upload ./document.pdf
```

Or set in environment:

```bash
export STORAGE_MODE=local
```

## License

MIT License
