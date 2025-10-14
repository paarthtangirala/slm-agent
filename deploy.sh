#!/bin/bash

# SLM Personal Agent - Deployment Script
# This script handles deployment of the SLM Personal Agent to production

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="slm-personal-agent"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.production"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "System requirements satisfied"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        if [ -f "$ENV_FILE" ]; then
            log_info "Copying production environment template..."
            cp "$ENV_FILE" .env
            log_warning "Please edit .env file with your actual API keys and configuration"
        else
            log_error "No environment file found. Please create .env file."
            exit 1
        fi
    fi
    
    # Create necessary directories
    mkdir -p uploads chromadb ssl logs
    
    log_success "Environment setup complete"
}

pull_ollama_model() {
    log_info "Setting up Ollama model..."
    
    # Start Ollama service first
    docker-compose up -d ollama
    
    # Wait for Ollama to be ready
    log_info "Waiting for Ollama to be ready..."
    max_attempts=30
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose exec -T ollama curl -f http://localhost:11434/api/tags &> /dev/null; then
            break
        fi
        sleep 5
        attempt=$((attempt + 1))
        echo -n "."
    done
    echo
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "Ollama failed to start properly"
        exit 1
    fi
    
    # Pull the phi3:mini model
    log_info "Pulling phi3:mini model (this may take a while)..."
    docker-compose exec -T ollama ollama pull phi3:mini
    
    log_success "Ollama model setup complete"
}

deploy_application() {
    log_info "Deploying SLM Personal Agent..."
    
    # Build and start all services
    docker-compose build --no-cache
    docker-compose up -d
    
    # Wait for application to be ready
    log_info "Waiting for application to be ready..."
    max_attempts=30
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            break
        fi
        sleep 5
        attempt=$((attempt + 1))
        echo -n "."
    done
    echo
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "Application failed to start properly"
        docker-compose logs slm-agent
        exit 1
    fi
    
    log_success "Application deployed successfully"
}

show_status() {
    log_info "Deployment Status:"
    echo
    docker-compose ps
    echo
    log_info "Application is available at:"
    echo "  - Main App: http://localhost"
    echo "  - Direct API: http://localhost:8000"
    echo "  - Health Check: http://localhost:8000/health"
    echo "  - Ollama API: http://localhost:11434"
    echo
}

backup_data() {
    log_info "Creating backup of existing data..."
    
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="backups/backup_$timestamp"
    
    mkdir -p "$backup_dir"
    
    # Backup database
    if [ -f conversations.db ]; then
        cp conversations.db "$backup_dir/"
    fi
    
    # Backup uploads
    if [ -d uploads ]; then
        cp -r uploads "$backup_dir/"
    fi
    
    # Backup chromadb
    if [ -d chromadb ]; then
        cp -r chromadb "$backup_dir/"
    fi
    
    log_success "Backup created at $backup_dir"
}

cleanup() {
    log_info "Cleaning up old containers and images..."
    
    # Remove old containers
    docker container prune -f
    
    # Remove old images
    docker image prune -f
    
    log_success "Cleanup complete"
}

# Main deployment process
main() {
    echo
    log_info "🚀 Starting SLM Personal Agent Deployment"
    echo
    
    # Parse command line arguments
    case "${1:-deploy}" in
        "deploy")
            check_requirements
            setup_environment
            backup_data
            pull_ollama_model
            deploy_application
            show_status
            ;;
        "start")
            log_info "Starting existing deployment..."
            docker-compose start
            show_status
            ;;
        "stop")
            log_info "Stopping deployment..."
            docker-compose stop
            log_success "Deployment stopped"
            ;;
        "restart")
            log_info "Restarting deployment..."
            docker-compose restart
            show_status
            ;;
        "logs")
            docker-compose logs -f
            ;;
        "status")
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "backup")
            backup_data
            ;;
        "update")
            log_info "Updating deployment..."
            backup_data
            docker-compose pull
            docker-compose build --no-cache
            docker-compose up -d
            show_status
            ;;
        *)
            echo "Usage: $0 {deploy|start|stop|restart|logs|status|cleanup|backup|update}"
            echo
            echo "Commands:"
            echo "  deploy  - Full deployment (default)"
            echo "  start   - Start existing deployment"
            echo "  stop    - Stop deployment"
            echo "  restart - Restart deployment"
            echo "  logs    - Show application logs"
            echo "  status  - Show deployment status"
            echo "  cleanup - Clean up old containers and images"
            echo "  backup  - Backup data"
            echo "  update  - Update and restart deployment"
            exit 1
            ;;
    esac
    
    echo
    log_success "✅ Operation completed successfully!"
}

# Trap to handle script interruption
trap 'log_warning "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"