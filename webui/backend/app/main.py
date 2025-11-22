from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
from pathlib import Path

# Add parent directory to path to import rag_system
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rag_system.workflows.rag_workflow import get_rag_workflow
from rag_system.workflows.ingest_workflow import get_ingest_workflow
from rag_system.services.qdrant_service import get_qdrant_service
from rag_system.services.elasticsearch_service import get_elasticsearch_service
from rag_system.services.redis_service import get_redis_service
from rag_system.services.embeddings import get_embedding_service

app = FastAPI(title="RAG System WebUI API", version="1.0.0")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request/Response Models
class AskRequest(BaseModel):
    query: str
    strict_local: Optional[bool] = False
    fast: Optional[bool] = False
    web_search: Optional[bool] = True

class AskResponse(BaseModel):
    query: str
    answer: str
    routing: Dict[str, Any]
    sources_used: List[str]
    tool_results: Dict[str, Any]
    failed_tools: List[str]
    context_count: int
    citations: List[str]
    latency_ms: float
    timestamp: str
    fast_path: Optional[bool] = None

class StatusResponse(BaseModel):
    qdrant: bool
    elasticsearch: bool
    redis: bool
    embeddings: bool
    overall: bool

class IngestResponse(BaseModel):
    status: str
    message: str
    files_processed: int
    chunks_created: Optional[int] = None

class KBStatsResponse(BaseModel):
    document_count: int
    chunk_count: int
    last_ingested_at: Optional[str] = None

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Ask a question to the RAG system.
    
    This endpoint wraps the RAGWorkflow.execute() method and returns
    the same JSON structure as the CLI --json mode.
    """
    try:
        rag_workflow = get_rag_workflow()
        result = rag_workflow.execute(
            query=request.query,
            strict_local=request.strict_local,
            fast_mode=request.fast,
            allow_web_search=request.web_search
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(
    files: Optional[List[UploadFile]] = File(None),
    urls: Optional[str] = Form(None)
):
    """
    Ingest documents into the knowledge base.
    
    Accepts file uploads or URLs (comma-separated) to ingest.
    """
    try:
        ingest_workflow = get_ingest_workflow()
        files_processed = 0
        chunks_created = 0
        
        # Handle file uploads
        if files:
            import tempfile
            for file in files:
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name
                
                try:
                    # Ingest the file
                    result = ingest_workflow.ingest_path(tmp_path)
                    files_processed += 1
                    if isinstance(result, dict) and 'chunks' in result:
                        chunks_created += result['chunks']
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
        
        # Handle URLs
        if urls:
            url_list = [url.strip() for url in urls.split(',') if url.strip()]
            for url in url_list:
                try:
                    result = ingest_workflow.ingest_path(url)
                    files_processed += 1
                    if isinstance(result, dict) and 'chunks' in result:
                        chunks_created += result['chunks']
                except Exception as e:
                    print(f"Error ingesting URL {url}: {e}")
        
        if files_processed == 0:
            return IngestResponse(
                status="error",
                message="No files or URLs provided",
                files_processed=0
            )
        
        return IngestResponse(
            status="success",
            message=f"Successfully ingested {files_processed} file(s)",
            files_processed=files_processed,
            chunks_created=chunks_created if chunks_created > 0 else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")

@app.get("/api/status", response_model=StatusResponse)
async def status():
    """
    Check the health status of all services.
    
    Returns the connectivity status of Qdrant, Elasticsearch, Redis, and Embeddings.
    """
    try:
        # Check Qdrant
        qdrant_ok = False
        try:
            qdrant = get_qdrant_service()
            qdrant_ok = qdrant.client is not None
        except:
            pass
        
        # Check Elasticsearch
        es_ok = False
        try:
            es = get_elasticsearch_service()
            es_ok = es.client is not None and es.client.ping()
        except:
            pass
        
        # Check Redis
        redis_ok = False
        try:
            redis = get_redis_service()
            redis_ok = redis.client is not None and redis.client.ping()
        except:
            pass
        
        # Check Embeddings
        embeddings_ok = False
        try:
            embeddings = get_embedding_service()
            embeddings_ok = embeddings is not None
        except:
            pass
        
        overall = qdrant_ok and es_ok and redis_ok and embeddings_ok
        
        return StatusResponse(
            qdrant=qdrant_ok,
            elasticsearch=es_ok,
            redis=redis_ok,
            embeddings=embeddings_ok,
            overall=overall
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking status: {str(e)}")

@app.get("/api/kb/stats", response_model=KBStatsResponse)
async def kb_stats():
    """
    Get knowledge base statistics.
    
    Returns document count, chunk count, and last ingestion timestamp.
    """
    try:
        qdrant = get_qdrant_service()
        collection_info = qdrant.client.get_collection(collection_name=qdrant.collection_name)
        
        return KBStatsResponse(
            document_count=collection_info.points_count,
            chunk_count=collection_info.points_count,
            last_ingested_at=None  # Not tracked yet, could be added later
        )
    except Exception as e:
        # Return zeros if KB is not accessible
        return KBStatsResponse(
            document_count=0,
            chunk_count=0,
            last_ingested_at=None
        )
