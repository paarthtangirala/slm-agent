# 🚀 SLM Personal Agent - Production Deployment Guide

Complete guide for deploying the SLM Personal Agent to production environments.

## 📋 Pre-Deployment Checklist

### System Requirements
- [ ] Docker 20.10+ installed
- [ ] Docker Compose v2.0+ installed
- [ ] 4GB+ RAM available
- [ ] 10GB+ disk space
- [ ] Internet connection for initial setup

### Optional Requirements
- [ ] Domain name (for SSL/HTTPS)
- [ ] SSL certificates
- [ ] SerpAPI key for web search
- [ ] Bing Search API key (alternative)

## 🏗️ Deployment Architecture

```
Internet
    │
    ▼
┌─────────────────┐
│     Nginx       │  ← Reverse Proxy (Port 80/443)
│  Load Balancer  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   SLM Agent     │  ← FastAPI Application (Port 8000)
│   Application   │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│     Ollama      │  ← LLM Inference Server (Port 11434)
│   LLM Server    │
└─────────────────┘

Persistent Volumes:
├── uploads/          ← User uploaded files
├── chromadb/         ← Vector database
├── conversations.db  ← SQLite database
└── ollama-data/      ← Model storage
```

## 🚀 Quick Deployment

### 1. Download and Setup
```bash
# Clone repository
git clone <your-repo-url>
cd slm-personal-agent

# Make deployment script executable
chmod +x deploy.sh
```

### 2. Configure Environment
```bash
# Copy production template
cp .env.production .env

# Edit configuration (required)
nano .env
```

**Required Configuration:**
```env
# Add your API keys
SERPAPI_KEY=your_actual_serpapi_key_here
BING_SEARCH_KEY=your_actual_bing_key_here

# Set production values
ENVIRONMENT=production
SECRET_KEY=your_strong_secret_key_here
```

### 3. Deploy
```bash
# Full deployment (includes model download)
./deploy.sh deploy
```

### 4. Verify Deployment
```bash
# Check status
./deploy.sh status

# Test health endpoint
curl http://localhost/health
```

## 🔧 Advanced Configuration

### Custom Domain Setup

1. **Point Domain to Server**
   ```bash
   # Example DNS A record
   your-domain.com → your.server.ip
   ```

2. **Update Nginx Configuration**
   ```bash
   # Edit nginx.conf
   server_name your-domain.com;
   ```

3. **Setup SSL (Let's Encrypt)**
   ```bash
   # Install certbot
   sudo apt-get install certbot python3-certbot-nginx
   
   # Get certificate
   sudo certbot --nginx -d your-domain.com
   ```

### Environment Variables Reference

```env
# Core Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=phi3:mini
ENVIRONMENT=production
DEBUG=false

# Search APIs
SERPAPI_KEY=your_serpapi_key
BING_SEARCH_KEY=your_bing_key
SEARCH_PROVIDER=serpapi

# Security
SECRET_KEY=your_secret_key_change_in_production
JWT_SECRET=your_jwt_secret_change_in_production

# File Upload Limits
MAX_UPLOAD_SIZE=100MB
ALLOWED_EXTENSIONS=.pdf,.docx,.xlsx,.txt,.md,.py,.js,.png,.jpg,.jpeg

# Performance
WORKERS=4
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# CORS (adjust for security)
ALLOWED_ORIGINS=https://your-domain.com
ALLOWED_METHODS=GET,POST,PUT,DELETE,OPTIONS
ALLOWED_HEADERS=Content-Type,Authorization
```

## 🔒 Security Hardening

### 1. Firewall Configuration
```bash
# UFW Example
sudo ufw enable
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw deny 8000   # Block direct app access
sudo ufw deny 11434  # Block direct Ollama access
```

### 2. SSL/TLS Setup
```nginx
# nginx.conf HTTPS configuration
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Strong SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # Your app configuration...
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### 3. Access Control
```nginx
# Restrict admin endpoints
location /admin {
    allow 192.168.1.0/24;  # Your network
    deny all;
    proxy_pass http://slm_backend;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;

location /api/ {
    limit_req zone=api burst=20 nodelay;
    # ... proxy config
}
```

## 📊 Monitoring & Observability

### Health Checks
```bash
# Application health
curl http://localhost/health

# Detailed service status
./deploy.sh status

# Container logs
./deploy.sh logs
```

### Log Management
```bash
# View real-time logs
docker-compose logs -f

# Service-specific logs
docker-compose logs slm-agent
docker-compose logs ollama
docker-compose logs nginx

# Log rotation (add to crontab)
0 2 * * * docker system prune -f
```

### Metrics Collection
```bash
# Enable metrics in .env
ENABLE_METRICS=true
METRICS_PORT=9090

# Access metrics
curl http://localhost:9090/metrics
```

## 🔄 Maintenance & Updates

### Regular Maintenance
```bash
# Weekly backup
./deploy.sh backup

# Monthly cleanup
./deploy.sh cleanup

# Update to latest version
./deploy.sh update
```

### Backup Strategy
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/opt/backups/slm-agent"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup database
cp conversations.db "$BACKUP_DIR/conversations_$DATE.db"

# Backup uploads
tar -czf "$BACKUP_DIR/uploads_$DATE.tar.gz" uploads/

# Backup vector database
tar -czf "$BACKUP_DIR/chromadb_$DATE.tar.gz" chromadb/

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.db" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
```

### Update Process
```bash
# 1. Backup current state
./deploy.sh backup

# 2. Pull latest changes
git pull origin main

# 3. Update deployment
./deploy.sh update

# 4. Verify health
curl http://localhost/health
```

## 🐳 Docker Commands Reference

### Container Management
```bash
# View running containers
docker-compose ps

# Start specific service
docker-compose start slm-agent

# Stop specific service
docker-compose stop slm-agent

# Restart with new config
docker-compose up -d --force-recreate

# View container resources
docker stats
```

### Troubleshooting
```bash
# Execute commands in container
docker-compose exec slm-agent bash
docker-compose exec ollama bash

# Check container logs
docker-compose logs --tail=100 slm-agent

# Rebuild containers
docker-compose build --no-cache
docker-compose up -d
```

## ⚡ Performance Optimization

### Resource Allocation
```yaml
# docker-compose.yml optimizations
services:
  slm-agent:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
  
  ollama:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

### Nginx Optimization
```nginx
# nginx.conf performance settings
worker_processes auto;
worker_connections 2048;

# Enable gzip compression
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types text/plain text/css application/json application/javascript;

# Cache static files
location /static/ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## 🚨 Troubleshooting Guide

### Common Issues

#### 1. Application Won't Start
```bash
# Check logs
./deploy.sh logs

# Common causes:
# - Port conflicts
# - Missing environment variables
# - Insufficient memory
# - Ollama model not downloaded

# Solutions:
./deploy.sh stop
./deploy.sh start
```

#### 2. Ollama Model Issues
```bash
# Check if model is downloaded
docker-compose exec ollama ollama list

# Re-download model
docker-compose exec ollama ollama pull phi3:mini

# Check Ollama health
curl http://localhost:11434/api/tags
```

#### 3. Upload Issues
```bash
# Check upload directory permissions
ls -la uploads/

# Fix permissions
sudo chown -R 1000:1000 uploads/
sudo chmod -R 755 uploads/
```

#### 4. Database Issues
```bash
# Check database file
ls -la conversations.db

# Backup and reset if corrupted
cp conversations.db conversations.db.backup
rm conversations.db
./deploy.sh restart
```

### Emergency Recovery
```bash
# Complete reset (destructive)
./deploy.sh stop
docker-compose down -v
docker system prune -a -f
./deploy.sh deploy
```

## 📞 Support & Maintenance

### Regular Monitoring
- [ ] Daily health checks
- [ ] Weekly log reviews
- [ ] Monthly updates
- [ ] Quarterly security audits

### Performance Metrics
- Response times < 5s for most requests
- Memory usage < 80% of available
- Disk usage monitoring
- Error rate < 1%

### Contact & Support
- Health endpoint: `/health`
- Application logs: `./deploy.sh logs`
- System metrics: `docker stats`
- Documentation: This guide + README.md

---

**🎯 Production Ready!**

Your SLM Personal Agent is now deployed and ready for production use. Monitor the health endpoint regularly and keep backups current.