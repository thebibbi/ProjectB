#!/bin/bash

# ProjectB Setup Script
# Automated setup for development environment

set -e  # Exit on error

echo "================================="
echo "ProjectB Setup Script"
echo "================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
echo "Checking prerequisites..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3.10+ is required but not found"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    print_success "Docker found"
else
    print_error "Docker is required but not found"
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    print_success "Docker Compose found"
else
    print_error "Docker Compose is required but not found"
    exit 1
fi

echo ""
echo "================================="
echo "Step 1: Virtual Environment"
echo "================================="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

echo ""
echo "================================="
echo "Step 2: Python Dependencies"
echo "================================="

echo "Installing Python packages (this may take a few minutes)..."
pip install -r requirements.txt > /dev/null 2>&1
print_success "Python dependencies installed"

echo ""
echo "================================="
echo "Step 3: Configuration"
echo "================================="

# Create .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    print_success ".env file created"
    print_warning "IMPORTANT: Edit .env and set NEO4J_PASSWORD before proceeding"
else
    print_warning ".env file already exists"
fi

# Create data directories
echo "Creating data directories..."
mkdir -p data/{test_videos,models,indexes,output}
mkdir -p logs
print_success "Data directories created"

echo ""
echo "================================="
echo "Step 4: Docker Services"
echo "================================="

echo "Starting Docker services (Neo4j and Redis)..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    print_success "Docker services started"
else
    print_error "Docker services failed to start"
    echo "Run 'docker-compose logs' to see the error"
    exit 1
fi

echo ""
echo "================================="
echo "Step 5: Neo4j Schema"
echo "================================="

echo "Initializing Neo4j schema..."
echo "Note: You may need to set NEO4J_PASSWORD in .env first"
echo ""

read -p "Do you want to initialize Neo4j schema now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/setup_neo4j.py
    print_success "Neo4j schema initialized"
else
    print_warning "Skipped Neo4j initialization"
    echo "Run 'python scripts/setup_neo4j.py' later to initialize"
fi

echo ""
echo "================================="
echo "Step 6: Pre-commit Hooks (Optional)"
echo "================================="

read -p "Do you want to install pre-commit hooks? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "Skipped pre-commit hooks"
fi

echo ""
echo "================================="
echo "Setup Complete! ✓"
echo "================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and set your NEO4J_PASSWORD"
echo "2. Restart services: docker-compose restart"
echo "3. Initialize schema: python scripts/setup_neo4j.py"
echo "4. Run tests: pytest"
echo ""
echo "Useful commands:"
echo "  - Start services:  docker-compose up -d"
echo "  - Stop services:   docker-compose down"
echo "  - View logs:       docker-compose logs -f"
echo "  - Neo4j browser:   http://localhost:7474"
echo ""
echo "For more information, see README.md"
echo ""
