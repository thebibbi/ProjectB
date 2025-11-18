# Phase 0 Docker Services Testing Guide

This guide will help you test the Docker services (Neo4j and Redis) on your local machine.

---

## Prerequisites

Before starting, ensure you have:
- **Docker** installed and running
- **Docker Compose** installed
- **ProjectB** repository cloned

Check your installations:
```bash
docker --version
docker-compose --version
```

---

## Step-by-Step Testing

### 1. Configure Environment

```bash
cd ProjectB

# Create .env file from template
cp .env.example .env

# Edit .env and set a secure password for Neo4j
nano .env  # or use your preferred editor
```

**Important**: Change the `NEO4J_PASSWORD` value in `.env` to something secure!

Example:
```bash
NEO4J_PASSWORD=your_secure_password_here_123
```

### 2. Start Docker Services

```bash
# Start services in detached mode
docker-compose up -d

# You should see output like:
# Creating projectb-neo4j  ... done
# Creating projectb-redis  ... done
```

### 3. Verify Services Are Running

```bash
# Check service status
docker-compose ps

# Expected output:
#        Name                    Command             State           Ports
# -------------------------------------------------------------------------------
# projectb-neo4j   /docker-entrypoint.sh neo...   Up      0.0.0.0:7474->7474/tcp
#                                                          0.0.0.0:7687->7687/tcp
# projectb-redis   redis-server --appendon...     Up      0.0.0.0:6379->6379/tcp
```

Both services should show **State: Up**.

###4. Test Neo4j Access

#### Browser Access
1. Open your web browser
2. Navigate to: http://localhost:7474
3. You should see the Neo4j Browser interface
4. Login with:
   - **Username**: `neo4j`
   - **Password**: (the password you set in `.env`)

#### Connection Test
Once logged in, run this Cypher query:
```cypher
RETURN "Neo4j is working!" AS message
```

You should see the message displayed in the results.

#### Command Line Test
```bash
# Test connection using Python
python3 -c "
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    'bolt://localhost:7687',
    auth=('neo4j', 'your_password_here')  # Use your actual password
)

with driver.session() as session:
    result = session.run('RETURN 1 AS test')
    print('✓ Neo4j connection successful!')
    print(f'  Result: {result.single()[\"test\"]}')

driver.close()
"
```

### 5. Test Redis Access

```bash
# Test Redis ping
docker exec -it projectb-redis redis-cli ping

# Expected output: PONG
```

#### Python Test
```bash
# Install redis if not already installed
pip install redis

# Test connection
python3 -c "
import redis

r = redis.Redis(host='localhost', port=6379, db=0)
r.set('test_key', 'Hello from ProjectB!')
value = r.get('test_key').decode('utf-8')
print(f'✓ Redis connection successful!')
print(f'  Test value: {value}')
r.delete('test_key')
"
```

### 6. Initialize Neo4j Schema

```bash
# Activate your virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the schema initialization script
python scripts/setup_neo4j.py
```

You should see output like:
```
=== Neo4j Schema Setup ===
Connected to Neo4j at bolt://localhost:7687
Creating constraints...
✓ Created constraint
✓ Created constraint
✓ Created constraint
Creating indexes...
✓ Created index
✓ Created index
...
Verifying schema...
Found 3 constraints
Found 7 indexes

Create sample cameras? (y/n): y
✓ Created camera: Main Entrance
✓ Created camera: Hallway 1
✓ Created camera: Lobby
✓ Created adjacency: cam_entrance <-> cam_hallway_1
✓ Created adjacency: cam_hallway_1 <-> cam_lobby

✓ Neo4j schema setup complete!
```

### 7. Verify Schema in Neo4j Browser

Go back to Neo4j Browser (http://localhost:7474) and run:

```cypher
// View all cameras
MATCH (c:Camera)
RETURN c

// View camera relationships
MATCH (c1:Camera)-[r:ADJACENT_TO]->(c2:Camera)
RETURN c1.name, r.distance, c2.name

// Show constraints
SHOW CONSTRAINTS

// Show indexes
SHOW INDEXES
```

---

## Viewing Logs

### View All Logs
```bash
docker-compose logs -f
```

### View Neo4j Logs Only
```bash
docker-compose logs -f neo4j
```

### View Redis Logs Only
```bash
docker-compose logs -f redis
```

---

## Common Issues & Solutions

### Issue: Port Already in Use

**Error**: `Bind for 0.0.0.0:7687 failed: port is already allocated`

**Solution**:
```bash
# Check what's using the port
lsof -i :7687  # On Linux/Mac
netstat -ano | findstr :7687  # On Windows

# Either stop the conflicting service or change ports in docker-compose.yml
```

To change ports, edit `docker-compose.yml`:
```yaml
ports:
  - "7475:7474"  # Changed from 7474
  - "7688:7687"  # Changed from 7687
```

Then update `.env` to match:
```bash
NEO4J_URI=bolt://localhost:7688
```

### Issue: Connection Refused

**Error**: `Connection to localhost:7687 failed`

**Solution**:
1. Check if services are running: `docker-compose ps`
2. If not running: `docker-compose up -d`
3. Wait 10-15 seconds for Neo4j to fully start
4. Check logs: `docker-compose logs neo4j`

### Issue: Authentication Failed

**Error**: `Authentication failed`

**Solution**:
1. Verify password in `.env` matches what you're using to connect
2. If you forgot the password, reset Neo4j:
   ```bash
   docker-compose down
   docker volume rm projectb_neo4j_data
   docker-compose up -d
   # Then go to http://localhost:7474 and set a new password
   ```

### Issue: Neo4j Takes Too Long to Start

**Symptom**: Browser shows "ServiceUnavailable" for several minutes

**Solution**:
This is normal on first startup. Neo4j can take 30-60 seconds to fully initialize.
```bash
# Monitor logs
docker-compose logs -f neo4j

# Look for this message:
# "Started."
```

---

## Stopping Services

```bash
# Stop services (keeps data)
docker-compose stop

# Stop and remove containers (keeps data volumes)
docker-compose down

# Stop and remove EVERYTHING including data (⚠️ CAUTION)
docker-compose down -v
```

---

## Data Persistence

Your data is persisted in Docker volumes:

```bash
# List volumes
docker volume ls | grep projectb

# You should see:
# projectb_neo4j_data
# projectb_neo4j_logs
# projectb_neo4j_import
# projectb_neo4j_plugins
# projectb_redis_data
```

Even if you run `docker-compose down`, your data remains in these volumes.

---

## Performance Monitoring

### Neo4j Memory Usage
```bash
# Check Neo4j container stats
docker stats projectb-neo4j
```

### Redis Memory Usage
```bash
# Check Redis memory info
docker exec -it projectb-redis redis-cli INFO memory
```

---

## Backup & Restore

### Backup Neo4j Database
```bash
# Create backup
docker exec projectb-neo4j neo4j-admin database dump neo4j --to=/var/lib/neo4j/backups/backup-$(date +%Y%m%d).dump

# Copy backup to host
docker cp projectb-neo4j:/var/lib/neo4j/backups/ ./neo4j-backups/
```

### Restore Neo4j Database
```bash
# Stop Neo4j
docker-compose stop neo4j

# Restore from backup
docker exec projectb-neo4j neo4j-admin database load neo4j --from=/var/lib/neo4j/backups/backup-20251118.dump --overwrite-destination

# Start Neo4j
docker-compose start neo4j
```

---

## Next Steps

Once all services are verified:

✅ Neo4j accessible at http://localhost:7474
✅ Neo4j schema initialized
✅ Redis accessible
✅ Sample cameras created

You're ready to proceed with **Phase 1: Core Pipeline Development!**

---

## Quick Reference

| Service | Port | URL | Credentials |
|---------|------|-----|-------------|
| Neo4j Browser | 7474 | http://localhost:7474 | neo4j / (from .env) |
| Neo4j Bolt | 7687 | bolt://localhost:7687 | neo4j / (from .env) |
| Redis | 6379 | localhost:6379 | (no auth by default) |

---

**Questions or Issues?**

Check the main README.md troubleshooting section or review Docker logs:
```bash
docker-compose logs -f
```
