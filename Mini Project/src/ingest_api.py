"""
Job Market Data Ingestion Module.
Fetches job listings from supported public APIs (Remotive, Adzuna, RapidAPI JSearch).
All API keys are read from environment variables (.env).
"""

import os
import logging
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_remotive_jobs(category: Optional[str] = None, search: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch remote tech jobs from Remotive Public API (No API key required).
    
    Args:
        category: Job category slug (e.g., 'software-dev', 'data')
        search: Search term (e.g., 'python', 'data engineer')
        limit: Maximum number of jobs to fetch
        
    Returns:
        List of raw job dictionaries.
    """
    url = "https://remotive.com/api/remote-jobs"
    params = {}
    if category:
        params["category"] = category
    if search:
        params["search"] = search
    if limit:
        params["limit"] = limit

    headers = {"User-Agent": "JobMarketAnalytics/1.0 (DataScienceProject)"}
    
    try:
        logger.info(f"Fetching jobs from Remotive API: {params}")
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        jobs = data.get("jobs", [])
        logger.info(f"Retrieved {len(jobs)} jobs from Remotive.")
        return jobs
    except Exception as e:
        logger.error(f"Failed to fetch Remotive jobs: {e}")
        return []


def fetch_adzuna_jobs(country: str = "us", what: str = "data scientist", where: str = "", page: int = 1) -> List[Dict[str, Any]]:
    """
    Fetch jobs from Adzuna API using ADZUNA_APP_ID and ADZUNA_APP_KEY from .env.
    """
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key or app_id == "your_adzuna_app_id_here":
        logger.warning("ADZUNA_APP_ID or ADZUNA_APP_KEY missing in .env. Skipping Adzuna API.")
        return []

    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": 50,
        "what": what,
        "where": where,
        "content-type": "application/json"
    }

    try:
        logger.info(f"Fetching Adzuna jobs for '{what}' in '{where}'...")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        logger.info(f"Retrieved {len(results)} jobs from Adzuna.")
        return results
    except Exception as e:
        logger.error(f"Failed to fetch Adzuna jobs: {e}")
        return []


def fetch_jsearch_jobs(query: str = "Data Scientist in United States", num_pages: int = 1) -> List[Dict[str, Any]]:
    """
    Fetch jobs from RapidAPI JSearch using RAPIDAPI_KEY from .env.
    """
    rapidapi_key = os.getenv("RAPIDAPI_KEY")

    if not rapidapi_key or rapidapi_key == "your_rapidapi_key_here":
        logger.warning("RAPIDAPI_KEY missing in .env. Skipping JSearch API.")
        return []

    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    
    all_jobs = []
    for page in range(1, num_pages + 1):
        params = {"query": query, "page": str(page), "num_pages": "1"}
        try:
            logger.info(f"Fetching JSearch page {page} for '{query}'...")
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            jobs = data.get("data", [])
            all_jobs.extend(jobs)
        except Exception as e:
            logger.error(f"Failed on JSearch page {page}: {e}")
            break

    logger.info(f"Retrieved {len(all_jobs)} jobs from JSearch.")
    return all_jobs
