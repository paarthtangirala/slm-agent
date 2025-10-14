# Real Search API Setup Guide

Your SLM Personal Agent now supports **real web search** via Google and Bing APIs! Here's how to set it up:

## 🔥 Option 1: SerpAPI (Recommended)
**Best choice** - Google search results with easy setup.

### Setup Steps:
1. **Sign up**: Go to https://serpapi.com/
2. **Free plan**: 100 searches/month free (no credit card required)
3. **Get API key**: Dashboard → API Key
4. **Configure**: Add to your `.env` file:
   ```env
   SERPAPI_KEY=your_api_key_here
   SEARCH_PROVIDER=serpapi
   ```

### Pricing:
- **Free**: 100 searches/month
- **Paid**: $50/month for 5,000 searches

## 🔵 Option 2: Bing Search API
**Alternative** - Microsoft's search API.

### Setup Steps:
1. **Azure account**: Go to https://portal.azure.com/
2. **Create resource**: Search for "Bing Search v7"
3. **Free tier**: 1,000 searches/month free
4. **Get API key**: Resource → Keys and Endpoint
5. **Configure**: Add to your `.env` file:
   ```env
   BING_SEARCH_KEY=your_api_key_here
   SEARCH_PROVIDER=bing
   ```

### Pricing:
- **Free**: 1,000 searches/month
- **Paid**: $4/1,000 searches

## ⚡ Quick Start

### 1. Choose Your Provider
Edit `.env` file:
```env
# For SerpAPI (Google)
SERPAPI_KEY=your_serpapi_key
SEARCH_PROVIDER=serpapi

# OR for Bing
BING_SEARCH_KEY=your_bing_key  
SEARCH_PROVIDER=bing

# OR stay with mock results
SEARCH_PROVIDER=mock
```

### 2. Restart Server
```bash
# Stop current server (Ctrl+C)
# Restart
uvicorn app.main:app --reload
```

### 3. Test Real Search
```bash
curl -X POST http://localhost:8000/web-search \
  -H "Content-Type: application/json" \
  -d '{"query": "latest AI research 2025", "max_results": 3}'
```

## 🎯 What You Get

### With Real APIs:
- ✅ **Current results** from Google/Bing
- ✅ **Real URLs** and snippets  
- ✅ **AI-powered summaries** of actual content
- ✅ **Fresh information** updated daily

### Response Example:
```json
{
  "query": "AI research 2025",
  "results": [
    {
      "title": "Breakthrough in Large Language Models - Nature",
      "url": "https://nature.com/articles/ai-breakthrough-2025",
      "snippet": "Researchers achieve 90% efficiency improvement..."
    }
  ],
  "summary": "Recent AI research shows significant advances in...",
  "provider": "serpapi"
}
```

## 🔧 Smart Fallbacks

The system automatically tries:
1. **Primary provider** (serpapi/bing)
2. **Secondary provider** (if first fails)
3. **Mock results** (if all APIs fail)

This ensures your agent **always works**, even without API keys!

## 💡 Recommendations

### For Personal Use:
- **SerpAPI free tier**: 100 searches/month
- **Perfect for**: Learning, prototyping, personal projects

### For Production:
- **SerpAPI Pro**: 5,000 searches/month for $50
- **Bing API**: More cost-effective for high volume

### For Development:
- **Mock mode**: No API keys needed, perfect for testing

## 🚀 Ready to Go!

1. **Get your API key** (5 minutes)
2. **Update `.env`** (30 seconds)  
3. **Restart server** (10 seconds)
4. **Enjoy real search!** 🎉

Your SLM Personal Agent will now have **real-time web search** with AI summarization!