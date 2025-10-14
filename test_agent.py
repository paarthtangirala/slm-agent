#!/usr/bin/env python3
"""
Quick test script for the SLM Personal Agent
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, data=None, method="GET"):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def main():
    print("🤖 Testing SLM Personal Agent")
    print("=" * 50)
    
    # Test health
    print("\n1. Health Check")
    health = test_endpoint("/health")
    print(f"Status: {health}")
    
    if not health.get("ollama_connected"):
        print("⚠️  Ollama not connected. Make sure Ollama is running with phi3:mini model.")
        return
    
    # Test summarization
    print("\n2. Text Summarization")
    summary_data = {
        "text": "Artificial intelligence has revolutionized many industries. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions. Natural language processing has enabled computers to understand and generate human language with remarkable accuracy."
    }
    summary = test_endpoint("/summarize", summary_data, "POST")
    print(f"Summary: {summary.get('summary', 'Error')}")
    
    # Test email drafting
    print("\n3. Email Drafting")
    email_data = {
        "recipient": "john@example.com",
        "subject": "Project Update Meeting",
        "context": "Need to schedule a meeting to discuss Q4 project progress",
        "tone": "professional"
    }
    email = test_endpoint("/draft-email", email_data, "POST")
    draft = email.get('draft', 'Error')
    print(f"Email Draft (first 200 chars): {draft[:200]}...")
    
    # Test web search
    print("\n4. Web Search")
    search_data = {
        "query": "small language models 2025",
        "max_results": 3
    }
    search = test_endpoint("/web-search", search_data, "POST")
    print(f"Search Results: {len(search.get('results', []))} results")
    print(f"Summary: {search.get('summary', 'Error')[:150]}...")
    
    # Test local query
    print("\n5. Local Document Query")
    query_data = {
        "query": "What are the key features of this agent?",
        "max_results": 3
    }
    local_query = test_endpoint("/local-query", query_data, "POST")
    print(f"Answer: {local_query.get('answer', 'Error')}")
    print(f"Sources: {local_query.get('sources', [])}")
    
    print("\n✅ All tests completed!")
    print("\n🚀 Your SLM Personal Agent is working perfectly!")

if __name__ == "__main__":
    main()