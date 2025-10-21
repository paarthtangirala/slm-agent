"""
Integration Hub for SLM Personal Agent
Manages external API connections and service integrations
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
import base64
import hashlib
from urllib.parse import urlencode, parse_qs, urlparse
import os

import httpx
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, JSON
from sqlalchemy.ext.asyncio import AsyncSession

from .database import Base, memory_manager
from .ollama_client import call_ollama

logger = logging.getLogger(__name__)

class IntegrationStatus(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"
    AUTHENTICATING = "authenticating"
    EXPIRED = "expired"

class IntegrationType(str, Enum):
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    WEBHOOK = "webhook"
    CUSTOM = "custom"

@dataclass
class IntegrationConfig:
    id: str
    name: str
    type: IntegrationType
    description: str
    auth_url: Optional[str] = None
    token_url: Optional[str] = None
    api_base_url: Optional[str] = None
    scopes: List[str] = None
    required_fields: List[str] = None
    webhook_events: List[str] = None

class Integration(Base):
    __tablename__ = "integrations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    service_id = Column(String, nullable=False)  # github, slack, gdrive, etc.
    service_name = Column(String, nullable=False)
    user_id = Column(String, default="default")  # For multi-user support later
    
    # Authentication
    auth_type = Column(String, nullable=False)  # oauth2, api_key, etc.
    access_token = Column(Text)  # Encrypted
    refresh_token = Column(Text)  # Encrypted
    token_expires_at = Column(DateTime)
    api_key = Column(Text)  # Encrypted
    
    # Configuration
    config_data = Column(JSON)  # Service-specific configuration
    webhook_url = Column(String)
    webhook_secret = Column(String)
    
    # Status
    status = Column(String, default=IntegrationStatus.INACTIVE.value)
    last_sync = Column(DateTime)
    error_message = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

@dataclass
class IntegrationAction:
    id: str
    service_id: str
    action_type: str  # read, write, sync, notify
    name: str
    description: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    executed_at: Optional[datetime] = None

class IntegrationHub:
    def __init__(self):
        self.engine = memory_manager.engine
        self.async_session = memory_manager.async_session
        self._initialized = False
        self._encryption_key = self._get_encryption_key()
        
        # Built-in service configurations
        self.service_configs = {
            "github": IntegrationConfig(
                id="github",
                name="GitHub",
                type=IntegrationType.OAUTH2,
                description="Connect to GitHub for repository management and code collaboration",
                auth_url="https://github.com/login/oauth/authorize",
                token_url="https://github.com/login/oauth/access_token",
                api_base_url="https://api.github.com",
                scopes=["repo", "user", "notifications"],
                required_fields=["client_id", "client_secret"]
            ),
            "slack": IntegrationConfig(
                id="slack",
                name="Slack",
                type=IntegrationType.OAUTH2,
                description="Connect to Slack for team communication and notifications",
                auth_url="https://slack.com/oauth/v2/authorize",
                token_url="https://slack.com/api/oauth.v2.access",
                api_base_url="https://slack.com/api",
                scopes=["chat:write", "channels:read", "users:read"],
                required_fields=["client_id", "client_secret"]
            ),
            "gdrive": IntegrationConfig(
                id="gdrive",
                name="Google Drive",
                type=IntegrationType.OAUTH2,
                description="Connect to Google Drive for document storage and collaboration",
                auth_url="https://accounts.google.com/o/oauth2/auth",
                token_url="https://oauth2.googleapis.com/token",
                api_base_url="https://www.googleapis.com/drive/v3",
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
                required_fields=["client_id", "client_secret"]
            ),
            "notion": IntegrationConfig(
                id="notion",
                name="Notion",
                type=IntegrationType.OAUTH2,
                description="Connect to Notion for knowledge management and note-taking",
                auth_url="https://api.notion.com/v1/oauth/authorize",
                token_url="https://api.notion.com/v1/oauth/token",
                api_base_url="https://api.notion.com/v1",
                scopes=["read", "update"],
                required_fields=["client_id", "client_secret"]
            ),
            "trello": IntegrationConfig(
                id="trello",
                name="Trello",
                type=IntegrationType.API_KEY,
                description="Connect to Trello for project management and task tracking",
                api_base_url="https://api.trello.com/1",
                required_fields=["api_key", "token"]
            ),
            "calendar": IntegrationConfig(
                id="calendar",
                name="Google Calendar",
                type=IntegrationType.OAUTH2,
                description="Connect to Google Calendar for schedule management",
                auth_url="https://accounts.google.com/o/oauth2/auth",
                token_url="https://oauth2.googleapis.com/token",
                api_base_url="https://www.googleapis.com/calendar/v3",
                scopes=["https://www.googleapis.com/auth/calendar.readonly"],
                required_fields=["client_id", "client_secret"]
            )
        }
    
    async def initialize(self):
        """Initialize the integration hub"""
        if self._initialized:
            return
            
        # Create integration tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        self._initialized = True
        logger.info("Integration Hub initialized")
    
    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data"""
        key_file = "integration_key.txt"
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        else:
            # Generate new key
            key = os.urandom(32)
            with open(key_file, "wb") as f:
                f.write(key)
            return key
    
    def _encrypt_data(self, data: str) -> str:
        """Simple base64 encoding for demo - use proper encryption in production"""
        if not data:
            return ""
        encoded = base64.b64encode(data.encode()).decode()
        return encoded
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Simple base64 decoding for demo - use proper decryption in production"""
        if not encrypted_data:
            return ""
        try:
            decoded = base64.b64decode(encrypted_data.encode()).decode()
            return decoded
        except Exception:
            return ""
    
    async def get_available_integrations(self) -> List[Dict[str, Any]]:
        """Get list of available integrations"""
        return [
            {
                "id": config.id,
                "name": config.name,
                "type": config.type.value,
                "description": config.description,
                "auth_url": config.auth_url,
                "required_fields": config.required_fields or [],
                "status": await self._get_integration_status(config.id)
            }
            for config in self.service_configs.values()
        ]
    
    async def _get_integration_status(self, service_id: str) -> str:
        """Get the current status of an integration"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select
            stmt = select(Integration).where(Integration.service_id == service_id)
            result = await session.execute(stmt)
            integration = result.scalar_one_or_none()
            
            if not integration:
                return IntegrationStatus.INACTIVE.value
            
            # Check if token is expired
            if integration.token_expires_at and integration.token_expires_at < datetime.utcnow():
                return IntegrationStatus.EXPIRED.value
            
            return integration.status
    
    async def start_oauth_flow(self, service_id: str, redirect_uri: str, state: str = None) -> str:
        """Start OAuth2 authorization flow"""
        config = self.service_configs.get(service_id)
        if not config or config.type != IntegrationType.OAUTH2:
            raise ValueError(f"Invalid OAuth service: {service_id}")
        
        # Read client credentials from environment or config
        client_id = os.getenv(f"{service_id.upper()}_CLIENT_ID")
        if not client_id:
            raise ValueError(f"Missing client ID for {service_id}")
        
        # Build authorization URL
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(config.scopes or [])
        }
        
        if state:
            params["state"] = state
        
        auth_url = f"{config.auth_url}?{urlencode(params)}"
        
        # Store pending auth state
        await self._store_auth_state(service_id, state, redirect_uri)
        
        return auth_url
    
    async def complete_oauth_flow(self, service_id: str, code: str, state: str = None) -> bool:
        """Complete OAuth2 authorization flow with authorization code"""
        config = self.service_configs.get(service_id)
        if not config or config.type != IntegrationType.OAUTH2:
            raise ValueError(f"Invalid OAuth service: {service_id}")
        
        # Get client credentials
        client_id = os.getenv(f"{service_id.upper()}_CLIENT_ID")
        client_secret = os.getenv(f"{service_id.upper()}_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            raise ValueError(f"Missing OAuth credentials for {service_id}")
        
        # Exchange code for token
        async with httpx.AsyncClient() as client:
            token_data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code"
            }
            
            # Get redirect URI from stored state
            auth_state = await self._get_auth_state(service_id, state)
            if auth_state and "redirect_uri" in auth_state:
                token_data["redirect_uri"] = auth_state["redirect_uri"]
            
            response = await client.post(config.token_url, data=token_data)
            response.raise_for_status()
            
            token_info = response.json()
            
            # Store integration
            await self._store_integration(
                service_id=service_id,
                auth_type=IntegrationType.OAUTH2.value,
                access_token=token_info.get("access_token"),
                refresh_token=token_info.get("refresh_token"),
                expires_in=token_info.get("expires_in")
            )
            
            return True
    
    async def add_api_key_integration(self, service_id: str, api_key: str, additional_config: Dict = None) -> bool:
        """Add an API key-based integration"""
        config = self.service_configs.get(service_id)
        if not config or config.type != IntegrationType.API_KEY:
            raise ValueError(f"Invalid API key service: {service_id}")
        
        # Test the API key
        is_valid = await self._test_api_key(service_id, api_key, additional_config)
        if not is_valid:
            raise ValueError("Invalid API key or configuration")
        
        # Store integration
        await self._store_integration(
            service_id=service_id,
            auth_type=IntegrationType.API_KEY.value,
            api_key=api_key,
            config_data=additional_config
        )
        
        return True
    
    async def _test_api_key(self, service_id: str, api_key: str, config: Dict = None) -> bool:
        """Test if an API key is valid"""
        config_obj = self.service_configs.get(service_id)
        if not config_obj:
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                if service_id == "trello":
                    # Test Trello API key
                    token = config.get("token") if config else None
                    if not token:
                        return False
                    
                    url = f"{config_obj.api_base_url}/members/me"
                    params = {"key": api_key, "token": token}
                    response = await client.get(url, params=params)
                    return response.status_code == 200
                
                # Add more API key tests for other services
                return True
                
        except Exception as e:
            logger.error(f"API key test failed for {service_id}: {e}")
            return False
    
    async def _store_integration(self, service_id: str, auth_type: str, access_token: str = None, 
                                refresh_token: str = None, api_key: str = None, expires_in: int = None,
                                config_data: Dict = None):
        """Store integration in database"""
        await self.initialize()
        
        async with self.async_session() as session:
            # Check if integration already exists
            from sqlalchemy import select
            stmt = select(Integration).where(Integration.service_id == service_id)
            result = await session.execute(stmt)
            integration = result.scalar_one_or_none()
            
            if integration:
                # Update existing
                if access_token:
                    integration.access_token = self._encrypt_data(access_token)
                if refresh_token:
                    integration.refresh_token = self._encrypt_data(refresh_token)
                if api_key:
                    integration.api_key = self._encrypt_data(api_key)
                if expires_in:
                    integration.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                if config_data:
                    integration.config_data = config_data
                
                integration.status = IntegrationStatus.ACTIVE.value
                integration.updated_at = datetime.utcnow()
                integration.error_message = None
            else:
                # Create new
                expires_at = None
                if expires_in:
                    expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                
                integration = Integration(
                    service_id=service_id,
                    service_name=self.service_configs[service_id].name,
                    auth_type=auth_type,
                    access_token=self._encrypt_data(access_token) if access_token else None,
                    refresh_token=self._encrypt_data(refresh_token) if refresh_token else None,
                    api_key=self._encrypt_data(api_key) if api_key else None,
                    token_expires_at=expires_at,
                    config_data=config_data,
                    status=IntegrationStatus.ACTIVE.value
                )
                session.add(integration)
            
            await session.commit()
    
    async def _store_auth_state(self, service_id: str, state: str, redirect_uri: str):
        """Store OAuth state for verification"""
        # Simple in-memory storage for demo - use proper storage in production
        if not hasattr(self, '_auth_states'):
            self._auth_states = {}
        
        self._auth_states[f"{service_id}_{state}"] = {
            "redirect_uri": redirect_uri,
            "created_at": datetime.utcnow()
        }
    
    async def _get_auth_state(self, service_id: str, state: str) -> Dict:
        """Get stored OAuth state"""
        if not hasattr(self, '_auth_states'):
            return {}
        
        return self._auth_states.get(f"{service_id}_{state}", {})
    
    async def get_user_integrations(self) -> List[Dict[str, Any]]:
        """Get user's active integrations"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select
            stmt = select(Integration).order_by(Integration.created_at.desc())
            result = await session.execute(stmt)
            integrations = result.scalars().all()
            
            return [
                {
                    "id": integration.id,
                    "service_id": integration.service_id,
                    "service_name": integration.service_name,
                    "status": integration.status,
                    "auth_type": integration.auth_type,
                    "last_sync": integration.last_sync.isoformat() if integration.last_sync else None,
                    "created_at": integration.created_at.isoformat(),
                    "error_message": integration.error_message
                }
                for integration in integrations
            ]
    
    async def execute_integration_action(self, service_id: str, action_type: str, 
                                       parameters: Dict[str, Any]) -> IntegrationAction:
        """Execute an action with an integrated service"""
        await self.initialize()
        
        # Get integration
        async with self.async_session() as session:
            from sqlalchemy import select
            stmt = select(Integration).where(Integration.service_id == service_id)
            result = await session.execute(stmt)
            integration = result.scalar_one_or_none()
            
            if not integration or integration.status != IntegrationStatus.ACTIVE.value:
                raise ValueError(f"Integration {service_id} not available")
            
            # Execute action based on service and action type
            action_result = await self._execute_service_action(
                integration, action_type, parameters
            )
            
            # Update last sync
            integration.last_sync = datetime.utcnow()
            await session.commit()
            
            return IntegrationAction(
                id=str(uuid.uuid4()),
                service_id=service_id,
                action_type=action_type,
                name=f"{service_id}_{action_type}",
                description=f"Execute {action_type} on {service_id}",
                parameters=parameters,
                result=action_result,
                executed_at=datetime.utcnow()
            )
    
    async def _execute_service_action(self, integration: Integration, action_type: str, 
                                    parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific service action"""
        service_id = integration.service_id
        config = self.service_configs.get(service_id)
        
        if not config:
            raise ValueError(f"Unknown service: {service_id}")
        
        # Get authentication token/key
        auth_header = {}
        if integration.access_token:
            token = self._decrypt_data(integration.access_token)
            auth_header["Authorization"] = f"Bearer {token}"
        elif integration.api_key:
            api_key = self._decrypt_data(integration.api_key)
            if service_id == "trello":
                # Trello uses key and token as query parameters
                pass
            else:
                auth_header["Authorization"] = f"Bearer {api_key}"
        
        async with httpx.AsyncClient(headers=auth_header) as client:
            if service_id == "github":
                return await self._execute_github_action(client, action_type, parameters)
            elif service_id == "slack":
                return await self._execute_slack_action(client, action_type, parameters)
            elif service_id == "gdrive":
                return await self._execute_gdrive_action(client, action_type, parameters)
            elif service_id == "notion":
                return await self._execute_notion_action(client, action_type, parameters)
            elif service_id == "trello":
                api_key = self._decrypt_data(integration.api_key)
                token = integration.config_data.get("token") if integration.config_data else ""
                return await self._execute_trello_action(client, action_type, parameters, api_key, token)
            elif service_id == "calendar":
                return await self._execute_calendar_action(client, action_type, parameters)
            else:
                raise ValueError(f"Action execution not implemented for {service_id}")
    
    async def _execute_github_action(self, client: httpx.AsyncClient, action_type: str, 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub-specific actions"""
        base_url = "https://api.github.com"
        
        if action_type == "list_repos":
            response = await client.get(f"{base_url}/user/repos")
            response.raise_for_status()
            repos = response.json()
            return {"repositories": [{"name": repo["name"], "url": repo["html_url"]} for repo in repos[:10]]}
        
        elif action_type == "get_notifications":
            response = await client.get(f"{base_url}/notifications")
            response.raise_for_status()
            notifications = response.json()
            return {"notifications": [{"title": notif["subject"]["title"], "type": notif["subject"]["type"]} for notif in notifications[:5]]}
        
        elif action_type == "create_issue":
            repo = parameters.get("repo")
            title = parameters.get("title")
            body = parameters.get("body", "")
            
            if not repo or not title:
                raise ValueError("Repository and title are required for creating issues")
            
            data = {"title": title, "body": body}
            response = await client.post(f"{base_url}/repos/{repo}/issues", json=data)
            response.raise_for_status()
            issue = response.json()
            return {"issue": {"number": issue["number"], "url": issue["html_url"]}}
        
        else:
            raise ValueError(f"Unknown GitHub action: {action_type}")
    
    async def _execute_slack_action(self, client: httpx.AsyncClient, action_type: str, 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Slack-specific actions"""
        base_url = "https://slack.com/api"
        
        if action_type == "list_channels":
            response = await client.get(f"{base_url}/conversations.list")
            response.raise_for_status()
            data = response.json()
            channels = data.get("channels", [])
            return {"channels": [{"name": ch["name"], "id": ch["id"]} for ch in channels[:10]]}
        
        elif action_type == "send_message":
            channel = parameters.get("channel")
            text = parameters.get("text")
            
            if not channel or not text:
                raise ValueError("Channel and text are required for sending messages")
            
            data = {"channel": channel, "text": text}
            response = await client.post(f"{base_url}/chat.postMessage", json=data)
            response.raise_for_status()
            result = response.json()
            return {"message_sent": result.get("ok", False)}
        
        else:
            raise ValueError(f"Unknown Slack action: {action_type}")
    
    async def _execute_gdrive_action(self, client: httpx.AsyncClient, action_type: str, 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Google Drive-specific actions"""
        base_url = "https://www.googleapis.com/drive/v3"
        
        if action_type == "list_files":
            params = {"pageSize": 10}
            response = await client.get(f"{base_url}/files", params=params)
            response.raise_for_status()
            data = response.json()
            files = data.get("files", [])
            return {"files": [{"name": f["name"], "id": f["id"]} for f in files]}
        
        else:
            raise ValueError(f"Unknown Google Drive action: {action_type}")
    
    async def _execute_notion_action(self, client: httpx.AsyncClient, action_type: str, 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Notion-specific actions"""
        base_url = "https://api.notion.com/v1"
        client.headers["Notion-Version"] = "2022-06-28"
        
        if action_type == "list_databases":
            response = await client.post(f"{base_url}/search", json={"filter": {"object": "database"}})
            response.raise_for_status()
            data = response.json()
            databases = data.get("results", [])
            return {"databases": [{"title": db.get("title", [{}])[0].get("plain_text", ""), "id": db["id"]} for db in databases[:5]]}
        
        else:
            raise ValueError(f"Unknown Notion action: {action_type}")
    
    async def _execute_trello_action(self, client: httpx.AsyncClient, action_type: str, 
                                   parameters: Dict[str, Any], api_key: str, token: str) -> Dict[str, Any]:
        """Execute Trello-specific actions"""
        base_url = "https://api.trello.com/1"
        auth_params = {"key": api_key, "token": token}
        
        if action_type == "list_boards":
            response = await client.get(f"{base_url}/members/me/boards", params=auth_params)
            response.raise_for_status()
            boards = response.json()
            return {"boards": [{"name": board["name"], "id": board["id"]} for board in boards[:10]]}
        
        elif action_type == "create_card":
            list_id = parameters.get("list_id")
            name = parameters.get("name")
            desc = parameters.get("desc", "")
            
            if not list_id or not name:
                raise ValueError("List ID and name are required for creating cards")
            
            data = {"name": name, "desc": desc, "idList": list_id}
            data.update(auth_params)
            response = await client.post(f"{base_url}/cards", data=data)
            response.raise_for_status()
            card = response.json()
            return {"card": {"name": card["name"], "id": card["id"], "url": card["url"]}}
        
        else:
            raise ValueError(f"Unknown Trello action: {action_type}")
    
    async def _execute_calendar_action(self, client: httpx.AsyncClient, action_type: str, 
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Google Calendar-specific actions"""
        base_url = "https://www.googleapis.com/calendar/v3"
        
        if action_type == "list_events":
            calendar_id = parameters.get("calendar_id", "primary")
            params = {"maxResults": 10, "singleEvents": True, "orderBy": "startTime"}
            response = await client.get(f"{base_url}/calendars/{calendar_id}/events", params=params)
            response.raise_for_status()
            data = response.json()
            events = data.get("items", [])
            return {"events": [{"summary": event.get("summary", ""), "start": event.get("start", {})} for event in events]}
        
        else:
            raise ValueError(f"Unknown Calendar action: {action_type}")
    
    async def disconnect_integration(self, service_id: str) -> bool:
        """Disconnect an integration"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select, delete
            stmt = select(Integration).where(Integration.service_id == service_id)
            result = await session.execute(stmt)
            integration = result.scalar_one_or_none()
            
            if integration:
                delete_stmt = delete(Integration).where(Integration.service_id == service_id)
                await session.execute(delete_stmt)
                await session.commit()
                return True
            
            return False
    
    async def get_integration_suggestions(self, conversation_text: str) -> List[Dict[str, Any]]:
        """Get AI-powered integration suggestions based on conversation"""
        try:
            system_prompt = """You are an integration advisor. Analyze the conversation and suggest relevant integrations that could help the user be more productive. Focus on practical, actionable integrations."""
            
            prompt = f"""Based on this conversation, suggest 2-3 integrations that would be most helpful:

{conversation_text}

Available integrations: GitHub, Slack, Google Drive, Notion, Trello, Google Calendar

Return suggestions as JSON array:
[
  {{
    "service_id": "github",
    "reason": "why this integration would help",
    "actions": ["specific actions they could take"]
  }}
]"""
            
            response = await call_ollama(prompt, system_prompt)
            
            # Parse AI response
            try:
                clean_response = response.strip()
                if clean_response.startswith("```"):
                    lines = clean_response.split('\n')
                    clean_response = '\n'.join(lines[1:-1])
                
                start_idx = clean_response.find('[')
                end_idx = clean_response.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    json_str = clean_response[start_idx:end_idx+1]
                    suggestions = json.loads(json_str)
                    return suggestions[:3]  # Limit to 3 suggestions
            except json.JSONDecodeError:
                logger.warning("Failed to parse integration suggestions")
            
        except Exception as e:
            logger.error(f"Integration suggestion error: {e}")
        
        return []

# Global integration hub instance
integration_hub = IntegrationHub()