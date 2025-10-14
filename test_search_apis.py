#!/usr/bin/env python3
"""
Test script to demonstrate real search APIs vs mock results
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8000"

def test_search(query, description):
    """Test search endpoint"""
    print(f"\n🔍 {description}")
    print("=" * 60)
    
    data = {"query": query, "max_results": 3}
    
    try:
        response = requests.post(f"{BASE_URL}/web-search", json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print(f"Query: {result['query']}")
        print(f"Provider: {result['provider']}")
        print(f"Results found: {len(result['results'])}")
        
        print("\nTop Results:")
        for i, res in enumerate(result['results'][:2], 1):
            print(f"  {i}. {res['title']}")
            print(f"     {res['url']}")
            if 'snippet' in res:
                print(f"     {res['snippet'][:100]}...")
            print()
        
        print("AI Summary:")
        print(f"  {result['summary'][:200]}...")
        
        if 'note' in result:
            print(f"\nNote: {result['note']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🤖 SLM Personal Agent - Search API Test")
    print("=" * 60)
    
    # Check current configuration
    serpapi_key = os.getenv("SERPAPI_KEY")
    bing_key = os.getenv("BING_SEARCH_KEY")
    provider = os.getenv("SEARCH_PROVIDER", "mock")
    
    print(f"Current configuration:")
    print(f"  Provider: {provider}")
    print(f"  SerpAPI key: {'✅ Set' if serpapi_key else '❌ Not set'}")
    print(f"  Bing key: {'✅ Set' if bing_key else '❌ Not set'}")
    
    # Test queries
    test_search(
        "artificial intelligence trends 2025",
        "Testing: AI Trends Search"
    )
    
    test_search(
        "climate change solutions",
        "Testing: Climate Solutions Search"
    )
    
    test_search(
        "latest iPhone features",
        "Testing: Tech Product Search"
    )
    
    print("\n" + "=" * 60)
    print("🚀 Search Integration Complete!")
    
    if provider == "mock":
        print("\n💡 To enable real search:")
        print("1. Get API key from https://serpapi.com/ (free)")
        print("2. Add to .env: SERPAPI_KEY=your_key")
        print("3. Set: SEARCH_PROVIDER=serpapi")
        print("4. Restart server and run this test again!")
    else:
        print("\n✅ Real search API is active!")
        print(f"Using: {provider}")

if __name__ == "__main__":
    main()