# Neo4j Authentication Error Fix Guide

This guide helps you resolve the authentication errors you're experiencing with Neo4j.

---

## The Problem

You're seeing these errors:
```
The client is unauthorized due to authentication failure
The client has provided incorrect authentication details too many times in a row
```

**Why this happens:**
- Your `.env` file has one password
- Neo4j container was started with a different password (from `docker-compose.yml`)
- After several failed attempts, Neo4j rate-limited your connection

---

## Quick Fix Options

### Option 1: Automated Fix (Recommended)

Use the automated reset script:

```bash
cd ProjectB

# Make script executable
chmod +x scripts/reset_neo4j_password.sh

# Run the reset script
./scripts/reset_neo4j_password.sh
```

This will:
1. Stop Neo4j
2. Remove old data (⚠️ deletes everything)
3. Update docker-compose.yml with your .env password
4. Restart Neo4j
5. Test the connection

---

### Option 2: Manual Fix (Reset Everything)

If you don't mind losing any existing Neo4j data:

```bash
# 1. Stop all services
docker-compose down

# 2. Remove Neo4j data volume
docker volume rm projectb_neo4j_data

# 3. Edit docker-compose.yml
nano docker-compose.yml

# Find this line under neo4j service:
#   - NEO4J_AUTH=neo4j/projectb_password

# Change it to match your .env password:
#   - NEO4J_AUTH=neo4j/YOUR_PASSWORD_FROM_ENV

# 4. Start services
docker-compose up -d

# 5. Wait 30 seconds for Neo4j to start
sleep 30

# 6. Test connection
python scripts/setup_neo4j.py
```

---

### Option 3: Update .env to Match Docker Compose

If you want to keep the Docker Compose password:

```bash
# 1. Check what password is in docker-compose.yml
grep NEO4J_AUTH docker-compose.yml
# You'll see something like: NEO4J_AUTH=neo4j/projectb_password

# 2. Edit .env to match
nano .env

# Change this line:
NEO4J_PASSWORD=your_password_here

# To this (use the password from step 1):
NEO4J_PASSWORD=projectb_password

# 3. Wait for rate limit to expire (5-10 minutes)
# OR restart Neo4j:
docker-compose restart neo4j
sleep 30

# 4. Test connection
python scripts/setup_neo4j.py
```

---

### Option 4: Quick Reset (One Command)

Complete reset in one go:

```bash
docker-compose down && \
docker volume rm projectb_neo4j_data && \
docker-compose up -d && \
sleep 30 && \
python scripts/setup_neo4j.py
```

**Note**: This uses the default password from `docker-compose.yml`. Make sure your `.env` matches!

---

## Verify the Fix

After applying any fix, verify it works:

### Test 1: Python Connection

```bash
python3 << 'EOF'
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
user = os.getenv('NEO4J_USER', 'neo4j')
password = os.getenv('NEO4J_PASSWORD')

print(f"Testing connection to {uri}...")
print(f"User: {user}")
print(f"Password: {password[:3]}***")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run('RETURN "Success!" AS message')
        print(f"✓ {result.single()['message']}")
    driver.close()
except Exception as e:
    print(f"✗ Failed: {e}")
EOF
```

### Test 2: Neo4j Browser

1. Open: http://localhost:7474
2. Login with credentials from your `.env`:
   - Username: `neo4j`
   - Password: (from `NEO4J_PASSWORD` in `.env`)
3. Run this query:
   ```cypher
   RETURN "Connected!" AS status
   ```

### Test 3: Schema Setup

```bash
python scripts/setup_neo4j.py
```

Should complete without authentication errors.

---

## Understanding the Passwords

There are two places where the Neo4j password is set:

### 1. docker-compose.yml
```yaml
environment:
  - NEO4J_AUTH=neo4j/projectb_password
```
This sets the password when Neo4j **first starts**. If data already exists, this is **ignored**.

### 2. .env file
```bash
NEO4J_PASSWORD=your_password_here
```
This is what your Python scripts use to **connect** to Neo4j.

**They must match!**

---

## Prevention

To avoid this issue in the future:

### 1. Use Consistent Passwords

Before starting services for the first time:

```bash
# 1. Set password in .env
echo "NEO4J_PASSWORD=my_secure_password_123" >> .env

# 2. Update docker-compose.yml
# Edit the NEO4J_AUTH line to match

# 3. Then start services
docker-compose up -d
```

### 2. Use Environment Variables in Docker Compose

Edit `docker-compose.yml` to use the .env password:

```yaml
environment:
  - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
```

Then Docker Compose will automatically use the password from `.env`.

---

## Common Questions

### Q: How long does the rate limit last?

**A:** Usually 5-10 minutes. You can bypass it by restarting Neo4j:
```bash
docker-compose restart neo4j
```

### Q: Will I lose my data?

**A:** Only if you remove the Docker volume (`docker volume rm projectb_neo4j_data`). If you just restart the container, data is preserved.

### Q: Can I change the password after Neo4j is running?

**A:** Yes, but it's complex. Easier to reset by removing the volume and starting fresh (for development).

### Q: What if I forget my password?

**A:** Reset Neo4j:
```bash
docker-compose down
docker volume rm projectb_neo4j_data
docker-compose up -d
```
Then use the password from `docker-compose.yml`.

---

## Still Having Issues?

### Check Neo4j Logs

```bash
docker-compose logs neo4j | tail -50
```

Look for:
- `Started.` - Neo4j is ready
- Authentication errors
- Port binding issues

### Check if Neo4j is Running

```bash
docker-compose ps neo4j
```

Should show `State: Up`.

### Check Connection from Container

```bash
docker exec -it projectb-neo4j cypher-shell -u neo4j -p YOUR_PASSWORD

# Try running:
RETURN 1;
```

### Nuclear Option: Complete Reset

```bash
# Stop everything
docker-compose down -v

# Remove all volumes
docker volume prune -f

# Remove .env
rm .env

# Start fresh
cp .env.example .env
nano .env  # Set password

# Update docker-compose.yml to match

# Start services
docker-compose up -d
sleep 30

# Initialize
python scripts/setup_neo4j.py
```

---

## Need More Help?

If none of these solutions work:

1. Share the output of:
   ```bash
   docker-compose logs neo4j
   cat .env | grep NEO4J
   grep NEO4J_AUTH docker-compose.yml
   ```

2. Check that Docker is running:
   ```bash
   docker ps
   ```

3. Verify no other Neo4j is running on port 7687:
   ```bash
   lsof -i :7687  # Linux/Mac
   netstat -ano | findstr :7687  # Windows
   ```

---

**Remember**: For Phase 0, losing Neo4j data is fine - we haven't created anything important yet!
