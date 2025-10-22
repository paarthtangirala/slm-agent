"""
Local Memory Management for Human-Centered AI Assistant
Privacy-first user preference and feedback storage
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class LocalMemoryManager:
    """
    Local storage for user preferences and feedback
    
    Human-Centered Design:
    - All data stays on user's machine
    - Transparent storage in human-readable JSON
    - Clear data organization
    - Easy backup and export capabilities
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.preferences_file = self.data_dir / "user_preferences.json"
        self.feedback_file = self.data_dir / "user_feedback.json"
        
        # Initialize files if they don't exist
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage files with default values"""
        # Default preferences
        if not self.preferences_file.exists():
            default_prefs = {
                "tone": "friendly",
                "length": "medium",
                "voice": "system",
                "disable_web": False,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            self._save_json(self.preferences_file, default_prefs)
        
        # Initialize feedback storage
        if not self.feedback_file.exists():
            default_feedback = {
                "feedback_entries": [],
                "stats": {
                    "total_feedback": 0,
                    "average_rating": 0.0,
                    "last_feedback": None
                },
                "created_at": datetime.utcnow().isoformat()
            }
            self._save_json(self.feedback_file, default_feedback)
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file with error handling"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {}
    
    def _save_json(self, file_path: Path, data: Dict[str, Any]):
        """Save JSON file with error handling"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
    
    def get_preferences(self) -> Dict[str, Any]:
        """
        Get current user preferences
        
        Returns user preferences with defaults for missing values
        """
        prefs = self._load_json(self.preferences_file)
        
        # Ensure all expected keys exist with defaults
        defaults = {
            "tone": "friendly",
            "length": "medium", 
            "voice": "system",
            "disable_web": False
        }
        
        for key, default_value in defaults.items():
            if key not in prefs:
                prefs[key] = default_value
        
        return prefs
    
    def update_preferences(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user preferences with transparency
        
        Args:
            updates: Dictionary of preference updates
            
        Returns:
            Updated preferences dictionary
        """
        current_prefs = self.get_preferences()
        
        # Track what's being updated for transparency
        changed_keys = []
        for key, value in updates.items():
            if key in current_prefs and current_prefs[key] != value:
                changed_keys.append(key)
            current_prefs[key] = value
        
        # Add metadata
        current_prefs["updated_at"] = datetime.utcnow().isoformat()
        current_prefs["last_changes"] = changed_keys
        
        self._save_json(self.preferences_file, current_prefs)
        
        logger.info(f"Updated preferences: {changed_keys}")
        return current_prefs
    
    def add_feedback(
        self,
        task_type: str,
        user_input: str,
        ai_response: str,
        rating: int,
        correction: Optional[str] = None
    ) -> str:
        """
        Add user feedback for learning and improvement
        
        Args:
            task_type: Type of task (summarize, email, query)
            user_input: Original user request
            ai_response: AI's response
            rating: User rating 1-5
            correction: Optional correction/suggestion from user
            
        Returns:
            Feedback ID for reference
        """
        feedback_data = self._load_json(self.feedback_file)
        
        # Create feedback entry
        feedback_id = f"feedback_{int(time.time())}_{len(feedback_data['feedback_entries'])}"
        
        entry = {
            "id": feedback_id,
            "timestamp": datetime.utcnow().isoformat(),
            "task_type": task_type,
            "user_input": user_input[:500],  # Limit for privacy
            "ai_response": ai_response[:1000],  # Limit for storage
            "rating": max(1, min(5, rating)),  # Ensure 1-5 range
            "correction": correction[:500] if correction else None,
            "metadata": {
                "input_length": len(user_input),
                "response_length": len(ai_response)
            }
        }
        
        # Add to feedback list
        feedback_data["feedback_entries"].append(entry)
        
        # Update statistics
        ratings = [f["rating"] for f in feedback_data["feedback_entries"]]
        feedback_data["stats"] = {
            "total_feedback": len(feedback_data["feedback_entries"]),
            "average_rating": round(sum(ratings) / len(ratings), 2),
            "last_feedback": datetime.utcnow().isoformat(),
            "rating_distribution": {
                str(i): ratings.count(i) for i in range(1, 6)
            }
        }
        
        self._save_json(self.feedback_file, feedback_data)
        
        logger.info(f"Added feedback: {feedback_id}, rating: {rating}")
        return feedback_id
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics for user transparency"""
        feedback_data = self._load_json(self.feedback_file)
        return feedback_data.get("stats", {
            "total_feedback": 0,
            "average_rating": 0.0,
            "last_feedback": None
        })
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent feedback entries for analysis"""
        feedback_data = self._load_json(self.feedback_file)
        entries = feedback_data.get("feedback_entries", [])
        
        # Return most recent entries, removing sensitive data
        recent = entries[-limit:] if entries else []
        
        # Clean entries for safe return
        cleaned = []
        for entry in recent:
            cleaned.append({
                "id": entry["id"],
                "timestamp": entry["timestamp"],
                "task_type": entry["task_type"],
                "rating": entry["rating"],
                "has_correction": bool(entry.get("correction")),
                "input_length": entry.get("metadata", {}).get("input_length", 0),
                "response_length": entry.get("metadata", {}).get("response_length", 0)
            })
        
        return cleaned
    
    def export_data(self) -> Dict[str, Any]:
        """
        Export all user data for backup/portability
        
        Human-Centered Design: User owns their data
        """
        return {
            "preferences": self.get_preferences(),
            "feedback_stats": self.get_feedback_stats(),
            "export_timestamp": datetime.utcnow().isoformat(),
            "data_location": str(self.data_dir.absolute())
        }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Generate insights from feedback for transparency
        
        Shows user how their feedback is being used
        """
        feedback_data = self._load_json(self.feedback_file)
        entries = feedback_data.get("feedback_entries", [])
        
        if not entries:
            return {"message": "No feedback data available yet"}
        
        # Analyze feedback patterns
        by_task = {}
        for entry in entries:
            task = entry["task_type"]
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(entry["rating"])
        
        insights = {
            "total_interactions": len(entries),
            "task_performance": {
                task: {
                    "average_rating": round(sum(ratings) / len(ratings), 2),
                    "total_feedback": len(ratings),
                    "trend": "improving" if len(ratings) > 1 and ratings[-1] > ratings[0] else "stable"
                }
                for task, ratings in by_task.items()
            },
            "recent_trend": "positive" if len(entries) > 2 and entries[-1]["rating"] >= 4 else "needs_attention",
            "suggestions": self._generate_improvement_suggestions(entries)
        }
        
        return insights
    
    def _generate_improvement_suggestions(self, entries: List[Dict]) -> List[str]:
        """Generate improvement suggestions based on feedback patterns"""
        if not entries:
            return ["Provide feedback to help improve responses"]
        
        recent_ratings = [e["rating"] for e in entries[-5:]]  # Last 5 ratings
        avg_recent = sum(recent_ratings) / len(recent_ratings)
        
        suggestions = []
        
        if avg_recent < 3.0:
            suggestions.append("Consider adjusting response tone or length preferences")
            suggestions.append("Provide more specific corrections to help improve responses")
        
        if avg_recent >= 4.0:
            suggestions.append("Great feedback! The system is learning your preferences well")
        
        # Task-specific suggestions
        task_counts = {}
        for entry in entries[-10:]:  # Recent 10
            task = entry["task_type"]
            task_counts[task] = task_counts.get(task, 0) + 1
        
        if task_counts.get("query", 0) > 5:
            suggestions.append("Consider adding more documents to knowledge base for better query responses")
        
        return suggestions


# Global instance for application use
local_memory = LocalMemoryManager()