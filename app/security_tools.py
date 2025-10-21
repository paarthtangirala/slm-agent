"""
Security and Privacy Tools for SLM Personal Agent
Comprehensive data protection, encryption, access control, and privacy management
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
import base64
from pathlib import Path
import shutil
import tempfile

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, JSON, LargeBinary
from sqlalchemy.ext.asyncio import AsyncSession

from .database import Base, memory_manager
from .ollama_client import call_ollama

logger = logging.getLogger(__name__)

class SecurityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DataCategory(str, Enum):
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    CONFIDENTIAL = "confidential"
    PUBLIC = "public"

class AuditEventType(str, Enum):
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export" 
    DATA_DELETE = "data_delete"
    INTEGRATION_AUTH = "integration_auth"
    SECURITY_SCAN = "security_scan"
    LOGIN_ATTEMPT = "login_attempt"
    CONFIG_CHANGE = "config_change"

@dataclass
class SecurityAlert:
    id: str
    level: SecurityLevel
    title: str
    description: str
    category: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = None

class SecurityPolicy(Base):
    __tablename__ = "security_policies"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)  # DataCategory enum
    description = Column(Text)
    
    # Policy settings
    encryption_required = Column(Boolean, default=True)
    retention_days = Column(Integer, default=365)
    access_logging = Column(Boolean, default=True)
    auto_deletion = Column(Boolean, default=False)
    
    # Access controls
    allowed_integrations = Column(JSON)  # List of integration IDs
    restricted_hours = Column(JSON)  # Time-based access controls
    geolocation_restrictions = Column(JSON)  # Location-based controls
    
    # Privacy settings
    anonymization_enabled = Column(Boolean, default=False)
    data_masking = Column(Boolean, default=False)
    consent_required = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type = Column(String, nullable=False)  # AuditEventType enum
    user_id = Column(String, default="default")
    session_id = Column(String)
    
    # Event details
    action = Column(String, nullable=False)
    resource = Column(String)  # What was accessed/modified
    ip_address = Column(String)
    user_agent = Column(String)
    
    # Security context
    risk_score = Column(Integer, default=0)  # 0-100 risk assessment
    anomaly_detected = Column(Boolean, default=False)
    security_level = Column(String, default=SecurityLevel.LOW.value)
    
    # Data
    event_data = Column(JSON)  # Additional event metadata
    timestamp = Column(DateTime, default=datetime.utcnow)

class DataClassification(Base):
    __tablename__ = "data_classifications"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    data_type = Column(String, nullable=False)  # conversation, document, integration_token, etc.
    data_id = Column(String, nullable=False)  # Reference to actual data
    
    # Classification
    category = Column(String, nullable=False)  # DataCategory enum
    sensitivity_score = Column(Integer, default=1)  # 1-10 scale
    contains_pii = Column(Boolean, default=False)
    contains_secrets = Column(Boolean, default=False)
    
    # Protection applied
    encrypted = Column(Boolean, default=False)
    anonymized = Column(Boolean, default=False)
    access_restricted = Column(Boolean, default=False)
    
    # Retention
    retention_policy_id = Column(String)
    expires_at = Column(DateTime)
    auto_delete = Column(Boolean, default=False)
    
    classified_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime)

class SecurityToolsManager:
    def __init__(self):
        self.engine = memory_manager.engine
        self.async_session = memory_manager.async_session
        self._initialized = False
        self._encryption_key = None
        self._security_patterns = self._load_security_patterns()
        
    async def initialize(self):
        """Initialize security tools and create tables"""
        if self._initialized:
            return
            
        # Create security tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize encryption
        self._encryption_key = self._get_or_create_encryption_key()
        
        # Create default security policies
        await self._create_default_policies()
        
        self._initialized = True
        logger.info("Security Tools initialized")
    
    def _get_or_create_encryption_key(self) -> Fernet:
        """Get or create encryption key for data protection"""
        key_file = Path("security_key.key")
        
        if key_file.exists():
            with open(key_file, "rb") as f:
                key = f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            # Set secure permissions
            os.chmod(key_file, 0o600)
        
        return Fernet(key)
    
    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for detecting sensitive data"""
        return {
            "credit_card": [
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Credit card numbers
                r'\b\d{13,19}\b'  # General card pattern
            ],
            "ssn": [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
                r'\b\d{9}\b'  # 9 digit sequence
            ],
            "email": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            "phone": [
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
            ],
            "api_key": [
                r'\b[A-Za-z0-9]{32,}\b',  # 32+ char alphanumeric
                r'sk-[A-Za-z0-9]{20,}',  # OpenAI style
                r'xoxb-[A-Za-z0-9-]+'  # Slack style
            ],
            "password": [
                r'password[\'\":\s]*[\'\"]\w+[\'\""]',
                r'pwd[\'\":\s]*[\'\"]\w+[\'\""]'
            ]
        }
    
    async def _create_default_policies(self):
        """Create default security policies"""
        default_policies = [
            {
                "name": "Personal Data Protection",
                "category": DataCategory.PERSONAL.value,
                "description": "Standard protection for personal information",
                "encryption_required": True,
                "retention_days": 365,
                "access_logging": True,
                "anonymization_enabled": True
            },
            {
                "name": "Sensitive Data High Security",
                "category": DataCategory.SENSITIVE.value,
                "description": "Enhanced protection for sensitive information",
                "encryption_required": True,
                "retention_days": 180,
                "access_logging": True,
                "auto_deletion": True,
                "consent_required": True
            },
            {
                "name": "Confidential Data Maximum Security",
                "category": DataCategory.CONFIDENTIAL.value,
                "description": "Maximum protection for confidential data",
                "encryption_required": True,
                "retention_days": 90,
                "access_logging": True,
                "auto_deletion": True,
                "data_masking": True,
                "consent_required": True
            }
        ]
        
        async with self.async_session() as session:
            for policy_data in default_policies:
                # Check if policy already exists
                from sqlalchemy import select
                stmt = select(SecurityPolicy).where(SecurityPolicy.name == policy_data["name"])
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()
                
                if not existing:
                    policy = SecurityPolicy(**policy_data)
                    session.add(policy)
            
            await session.commit()
    
    async def classify_data(self, data_type: str, data_id: str, content: str) -> Dict[str, Any]:
        """Classify data and determine security requirements"""
        await self.initialize()
        
        # Analyze content for sensitive information
        pii_detected = False
        secrets_detected = False
        sensitivity_score = 1
        detected_patterns = []
        
        for pattern_type, patterns in self._security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    detected_patterns.append(pattern_type)
                    if pattern_type in ['credit_card', 'ssn', 'phone']:
                        pii_detected = True
                        sensitivity_score = max(sensitivity_score, 8)
                    elif pattern_type in ['api_key', 'password']:
                        secrets_detected = True
                        sensitivity_score = max(sensitivity_score, 9)
                    elif pattern_type == 'email':
                        pii_detected = True
                        sensitivity_score = max(sensitivity_score, 5)
        
        # Determine category based on detection
        category = DataCategory.PUBLIC
        if secrets_detected:
            category = DataCategory.CONFIDENTIAL
        elif pii_detected or sensitivity_score >= 7:
            category = DataCategory.SENSITIVE
        elif sensitivity_score >= 3:
            category = DataCategory.PERSONAL
        
        # Create or update classification
        async with self.async_session() as session:
            from sqlalchemy import select
            stmt = select(DataClassification).where(
                DataClassification.data_type == data_type,
                DataClassification.data_id == data_id
            )
            result = await session.execute(stmt)
            classification = result.scalar_one_or_none()
            
            if classification:
                # Update existing
                classification.category = category.value
                classification.sensitivity_score = sensitivity_score
                classification.contains_pii = pii_detected
                classification.contains_secrets = secrets_detected
                classification.last_accessed = datetime.utcnow()
            else:
                # Create new
                classification = DataClassification(
                    data_type=data_type,
                    data_id=data_id,
                    category=category.value,
                    sensitivity_score=sensitivity_score,
                    contains_pii=pii_detected,
                    contains_secrets=secrets_detected,
                    encrypted=category in [DataCategory.SENSITIVE, DataCategory.CONFIDENTIAL],
                    access_restricted=category == DataCategory.CONFIDENTIAL
                )
                session.add(classification)
            
            await session.commit()
        
        return {
            "category": category.value,
            "sensitivity_score": sensitivity_score,
            "contains_pii": pii_detected,
            "contains_secrets": secrets_detected,
            "detected_patterns": detected_patterns,
            "recommendation": self._get_security_recommendation(category, sensitivity_score)
        }
    
    def _get_security_recommendation(self, category: DataCategory, score: int) -> str:
        """Get security recommendations based on classification"""
        if category == DataCategory.CONFIDENTIAL:
            return "Enable encryption, restrict access, implement auto-deletion"
        elif category == DataCategory.SENSITIVE:
            return "Enable encryption, log all access, consider anonymization"
        elif category == DataCategory.PERSONAL:
            return "Enable access logging, consider retention limits"
        else:
            return "Standard security measures sufficient"
    
    async def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self._encryption_key:
            self._encryption_key = self._get_or_create_encryption_key()
        
        return self._encryption_key.encrypt(data.encode()).decode()
    
    async def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self._encryption_key:
            self._encryption_key = self._get_or_create_encryption_key()
        
        try:
            return self._encryption_key.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")
    
    async def log_security_event(self, event_type: AuditEventType, action: str, 
                                resource: str = None, event_data: Dict = None,
                                session_id: str = None, ip_address: str = None) -> str:
        """Log security events for audit trail"""
        await self.initialize()
        
        # Calculate risk score based on event type and patterns
        risk_score = self._calculate_risk_score(event_type, action, event_data)
        
        # Detect anomalies
        anomaly_detected = await self._detect_anomaly(event_type, action, event_data)
        
        # Determine security level
        security_level = SecurityLevel.LOW
        if risk_score >= 80:
            security_level = SecurityLevel.CRITICAL
        elif risk_score >= 60:
            security_level = SecurityLevel.HIGH
        elif risk_score >= 40:
            security_level = SecurityLevel.MEDIUM
        
        # Create audit log entry
        audit_log = AuditLog(
            event_type=event_type.value,
            action=action,
            resource=resource,
            session_id=session_id or str(uuid.uuid4()),
            ip_address=ip_address or "localhost",
            risk_score=risk_score,
            anomaly_detected=anomaly_detected,
            security_level=security_level.value,
            event_data=event_data or {}
        )
        
        async with self.async_session() as session:
            session.add(audit_log)
            await session.commit()
        
        # Generate security alert if high risk
        if risk_score >= 70 or anomaly_detected:
            await self._create_security_alert(audit_log)
        
        return audit_log.id
    
    def _calculate_risk_score(self, event_type: AuditEventType, action: str, 
                            event_data: Dict = None) -> int:
        """Calculate risk score for security events"""
        base_scores = {
            AuditEventType.DATA_ACCESS: 20,
            AuditEventType.DATA_EXPORT: 60,
            AuditEventType.DATA_DELETE: 80,
            AuditEventType.INTEGRATION_AUTH: 40,
            AuditEventType.SECURITY_SCAN: 10,
            AuditEventType.LOGIN_ATTEMPT: 30,
            AuditEventType.CONFIG_CHANGE: 50
        }
        
        score = base_scores.get(event_type, 20)
        
        # Increase score for sensitive actions
        if action in ['export_all', 'bulk_delete', 'admin_access']:
            score += 30
        
        # Increase score for off-hours access
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            score += 20
        
        # Check event data for risk indicators
        if event_data:
            if event_data.get('bulk_operation'):
                score += 25
            if event_data.get('external_ip'):
                score += 15
            if event_data.get('unusual_location'):
                score += 20
        
        return min(score, 100)
    
    async def _detect_anomaly(self, event_type: AuditEventType, action: str, 
                            event_data: Dict = None) -> bool:
        """Detect anomalous security events"""
        # Simple anomaly detection - in production, use ML models
        anomaly_indicators = [
            action in ['mass_export', 'bulk_delete', 'admin_override'],
            event_data and event_data.get('failed_attempts', 0) > 3,
            event_data and event_data.get('unusual_time', False),
            event_data and event_data.get('new_location', False)
        ]
        
        return any(anomaly_indicators)
    
    async def _create_security_alert(self, audit_log: AuditLog):
        """Create security alert for high-risk events"""
        level = SecurityLevel(audit_log.security_level)
        
        alert = SecurityAlert(
            id=str(uuid.uuid4()),
            level=level,
            title=f"Security Event: {audit_log.event_type}",
            description=f"High-risk {audit_log.action} detected (Risk: {audit_log.risk_score}/100)",
            category=audit_log.event_type,
            timestamp=audit_log.timestamp,
            metadata={
                "audit_log_id": audit_log.id,
                "risk_score": audit_log.risk_score,
                "anomaly": audit_log.anomaly_detected
            }
        )
        
        # Store alert (simplified - in production, integrate with alerting system)
        logger.warning(f"SECURITY ALERT: {alert.title} - {alert.description}")
    
    async def anonymize_data(self, data: str, anonymization_level: str = "standard") -> str:
        """Anonymize data while preserving utility"""
        anonymized = data
        
        # Replace PII patterns with anonymized versions
        if anonymization_level in ["standard", "aggressive"]:
            # Email addresses
            anonymized = re.sub(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                '[EMAIL_REDACTED]',
                anonymized
            )
            
            # Phone numbers
            anonymized = re.sub(
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                '[PHONE_REDACTED]',
                anonymized
            )
            
            # Credit card numbers
            anonymized = re.sub(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                '[CARD_REDACTED]',
                anonymized
            )
        
        if anonymization_level == "aggressive":
            # Names (simple heuristic)
            anonymized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME_REDACTED]', anonymized)
            
            # Addresses (basic pattern)
            anonymized = re.sub(r'\d+ [A-Za-z\s]+ (Street|St|Avenue|Ave|Road|Rd|Drive|Dr)', '[ADDRESS_REDACTED]', anonymized)
        
        return anonymized
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select, func, desc
            
            # Recent audit events
            recent_events = await session.execute(
                select(AuditLog).order_by(desc(AuditLog.timestamp)).limit(10)
            )
            events = recent_events.scalars().all()
            
            # Risk metrics
            high_risk_events = await session.execute(
                select(func.count(AuditLog.id)).where(AuditLog.risk_score >= 70)
            )
            high_risk_count = high_risk_events.scalar()
            
            # Anomalies
            anomaly_events = await session.execute(
                select(func.count(AuditLog.id)).where(AuditLog.anomaly_detected == True)
            )
            anomaly_count = anomaly_events.scalar()
            
            # Data classifications
            classified_data = await session.execute(
                select(DataClassification.category, func.count(DataClassification.id))
                .group_by(DataClassification.category)
            )
            classification_stats = dict(classified_data.fetchall())
            
            # Security policies
            policies = await session.execute(select(SecurityPolicy))
            policy_list = policies.scalars().all()
        
        return {
            "summary": {
                "total_events": len(events),
                "high_risk_events": high_risk_count,
                "anomaly_count": anomaly_count,
                "classification_stats": classification_stats,
                "active_policies": len(policy_list)
            },
            "recent_events": [
                {
                    "id": event.id,
                    "type": event.event_type,
                    "action": event.action,
                    "risk_score": event.risk_score,
                    "timestamp": event.timestamp.isoformat(),
                    "anomaly": event.anomaly_detected
                }
                for event in events
            ],
            "policies": [
                {
                    "id": policy.id,
                    "name": policy.name,
                    "category": policy.category,
                    "encryption_required": policy.encryption_required,
                    "retention_days": policy.retention_days
                }
                for policy in policy_list
            ]
        }
    
    async def run_security_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan"""
        await self.initialize()
        
        scan_results = {
            "scan_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "issues": [],
            "recommendations": [],
            "score": 100
        }
        
        # Check for unencrypted sensitive data
        async with self.async_session() as session:
            from sqlalchemy import select, and_
            
            unencrypted_sensitive = await session.execute(
                select(DataClassification).where(
                    and_(
                        DataClassification.category.in_([
                            DataCategory.SENSITIVE.value,
                            DataCategory.CONFIDENTIAL.value
                        ]),
                        DataClassification.encrypted == False
                    )
                )
            )
            unencrypted_count = len(unencrypted_sensitive.scalars().all())
            
            if unencrypted_count > 0:
                scan_results["issues"].append({
                    "severity": "high",
                    "type": "unencrypted_sensitive_data",
                    "description": f"{unencrypted_count} sensitive data items are not encrypted",
                    "count": unencrypted_count
                })
                scan_results["score"] -= 30
            
            # Check for old data without retention policy
            old_data = await session.execute(
                select(DataClassification).where(
                    DataClassification.classified_at < datetime.utcnow() - timedelta(days=365)
                )
            )
            old_data_count = len(old_data.scalars().all())
            
            if old_data_count > 0:
                scan_results["issues"].append({
                    "severity": "medium",
                    "type": "old_data_retention",
                    "description": f"{old_data_count} data items are older than 1 year",
                    "count": old_data_count
                })
                scan_results["score"] -= 15
            
            # Check for recent high-risk events
            recent_high_risk = await session.execute(
                select(func.count(AuditLog.id)).where(
                    and_(
                        AuditLog.risk_score >= 70,
                        AuditLog.timestamp >= datetime.utcnow() - timedelta(days=7)
                    )
                )
            )
            high_risk_count = recent_high_risk.scalar()
            
            if high_risk_count > 5:
                scan_results["issues"].append({
                    "severity": "high",
                    "type": "frequent_high_risk_events",
                    "description": f"{high_risk_count} high-risk events in the last week",
                    "count": high_risk_count
                })
                scan_results["score"] -= 25
        
        # Generate recommendations
        if scan_results["score"] < 90:
            scan_results["recommendations"].extend([
                "Enable encryption for all sensitive data",
                "Implement automated data retention policies",
                "Review and adjust security monitoring thresholds",
                "Consider implementing additional access controls"
            ])
        
        # Log the security scan
        await self.log_security_event(
            AuditEventType.SECURITY_SCAN,
            "comprehensive_scan",
            event_data={
                "scan_id": scan_results["scan_id"],
                "issues_found": len(scan_results["issues"]),
                "security_score": scan_results["score"]
            }
        )
        
        return scan_results

# Global security tools instance
security_tools = SecurityToolsManager()