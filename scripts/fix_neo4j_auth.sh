#!/bin/bash

# Quick Fix for Neo4j Authentication Issues
# This script resets Neo4j with the correct password from .env

echo "🔧 Neo4j Authentication Quick Fix"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Run this script from the ProjectB directory"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "   Run: cp .env.example .env"
    exit 1
fi

echo "This will:"
echo "  1. Stop Neo4j"
echo "  2. Delete Neo4j data (⚠️  all data will be lost)"
echo "  3. Restart with password from .env"
echo ""
echo "⚠️  WARNING: This deletes all Neo4j data!"
echo ""
read -p "Continue? [y/N] " -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "📦 Stopping services..."
docker-compose down
echo "✓ Stopped"

echo ""
echo "🗑️  Removing Neo4j data volume..."
docker volume rm projectb_neo4j_data 2>/dev/null || echo "  (volume didn't exist)"
echo "✓ Removed"

echo ""
echo "🚀 Starting services with correct password..."
docker-compose up -d
echo "✓ Started"

echo ""
echo "⏳ Waiting for Neo4j to initialize (30 seconds)..."
for i in {30..1}; do
    printf "\r   %2d seconds remaining..." $i
    sleep 1
done
echo ""
echo "✓ Wait complete"

echo ""
echo "🧪 Testing connection..."

# Source the .env file to get the password
source .env

# Test connection
python3 -c "
from neo4j import GraphDatabase
import sys

try:
    driver = GraphDatabase.driver(
        '${NEO4J_URI}',
        auth=('${NEO4J_USER}', '${NEO4J_PASSWORD}')
    )
    with driver.session() as session:
        result = session.run('RETURN 1')
        result.single()
    driver.close()
    print('✓ Connection successful!')
    sys.exit(0)
except Exception as e:
    print(f'✗ Connection failed: {e}')
    print()
    print('Troubleshooting:')
    print('  - Wait longer (Neo4j can take 1-2 minutes)')
    print('  - Check logs: docker-compose logs neo4j')
    print('  - Verify .env password matches docker-compose.yml')
    sys.exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Neo4j is ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Initialize schema: python scripts/setup_neo4j.py"
    echo "  2. Access browser: http://localhost:7474"
    echo "  3. Login with credentials from .env"
else
    echo ""
    echo "❌ Connection test failed"
    echo ""
    echo "Try:"
    echo "  docker-compose logs neo4j"
fi
