#!/bin/bash
# Скрипт для первоначальной настройки проекта MASS

set -e

echo "🚀 Setting up MASS (Metrics Analytics Super System) project..."

# Проверка версии Python
echo "🐍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.10 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $(python3 --version)"

# Создать venv если его нет
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "ℹ️  Virtual environment already exists"
fi

# Активировать venv
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Обновить pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Установить зависимости
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Установить пакет в режиме разработки
echo "🔧 Installing package in development mode..."
pip install -e .

# Проверка конфига YDB
if [ ! -f "config/ydb_qa_config.json" ]; then
    echo "⚠️  YDB config not found. Creating from example..."
    if [ -f "config/ydb_qa_config.json.example" ]; then
        cp config/ydb_qa_config.json.example config/ydb_qa_config.json
        echo "📝 Please edit config/ydb_qa_config.json with your YDB settings"
    else
        echo "⚠️  YDB config example not found. You may need to create config/ydb_qa_config.json manually"
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Configure YDB credentials (if needed):"
echo "     export CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS=/path/to/credentials.json"
echo "     # or create .env file with this variable"
echo ""
echo "  3. Edit config/ydb_qa_config.json with your YDB endpoint and path"
echo ""
echo "  4. Run analytics:"
echo "     python -m mass.core.analytics_job configs/example.yaml --dry-run"
echo ""
echo "  5. Or start the web UI:"
echo "     cd mass/ui && python app.py"
echo ""

