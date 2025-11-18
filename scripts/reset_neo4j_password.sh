#!/bin/bash

# Neo4j Password Reset Script
# Fixes authentication issues by resetting Neo4j with correct password

set -e

echo "================================="
echo "Neo4j Password Reset Tool"
echo "================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if .env exists
if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    echo "Run: cp .env.example .env"
    exit 1
fi

# Load current password from .env
source .env

echo "Current configuration:"
echo "  NEO4J_URI: $NEO4J_URI"
echo "  NEO4J_USER: $NEO4J_USER"
echo "  NEO4J_PASSWORD: ${NEO4J_PASSWORD:0:3}***"
echo ""

print_warning "This script will:"
echo "  1. Stop Neo4j container"
echo "  2. Remove Neo4j data volume (⚠️  deletes all data)"
echo "  3. Update docker-compose.yml with your password"
echo "  4. Restart Neo4j with correct password"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 1: Stopping Neo4j..."
docker-compose stop neo4j
print_success "Neo4j stopped"

echo ""
echo "Step 2: Removing Neo4j data volume..."
docker-compose down
docker volume rm projectb_neo4j_data 2>/dev/null || true
print_success "Data volume removed"

echo ""
echo "Step 3: Updating docker-compose.yml..."

# Update NEO4J_AUTH in docker-compose.yml
sed -i.bak "s/NEO4J_AUTH=neo4j\/[^\"']*/NEO4J_AUTH=neo4j\/$NEO4J_PASSWORD/" docker-compose.yml

print_success "docker-compose.yml updated"

echo ""
echo "Step 4: Starting Neo4j with new password..."
docker-compose up -d neo4j

echo ""
echo "Waiting for Neo4j to start (30 seconds)..."
sleep 30

print_success "Neo4j restarted"

echo ""
echo "Step 5: Testing connection..."

# Test connection
python3 << EOF
import sys
from neo4j import GraphDatabase

try:
    driver = GraphDatabase.driver(
        '$NEO4J_URI',
        auth=('$NEO4J_USER', '$NEO4J_PASSWORD')
    )
    with driver.session() as session:
        result = session.run('RETURN 1 AS test')
        value = result.single()['test']
        print('✓ Connection successful!')
    driver.close()
    sys.exit(0)
except Exception as e:
    print(f'✗ Connection failed: {e}')
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "Neo4j is ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: python scripts/setup_neo4j.py"
    echo "  2. Access Neo4j Browser: http://localhost:7474"
    echo "  3. Login with: neo4j / $NEO4J_PASSWORD"
else
    print_error "Connection test failed"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Wait longer (Neo4j can take 1-2 minutes to start)"
    echo "  2. Check logs: docker-compose logs neo4j"
    echo "  3. Verify password in .env matches docker-compose.yml"
fi
