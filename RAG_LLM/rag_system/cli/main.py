"""Main CLI interface for RAG system"""

import warnings
import logging
import click
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rag_system.core.config import get_config
from rag_system.workflows.ingest_workflow import get_ingest_workflow
from rag_system.workflows.rag_workflow import get_rag_workflow

warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*", module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", module="yfinance")
logging.getLogger("yfinance").setLevel(logging.ERROR)

console = Console()

@click.group()
def cli():
    """RAG System - Advanced Retrieval Augmented Generation with LLM"""
    pass

@cli.command()
@click.argument('path')
def ingest(path):
    """Ingest documents from a file, directory, or URL"""
    click.echo(f"Ingesting: {path}")
    
    try:
        workflow = get_ingest_workflow()
        result = workflow.ingest_path(path)
        
        if result.get('status') == 'success':
            click.echo(f"✓ Successfully ingested {result.get('chunks', 0)} chunks")
        elif result.get('status') == 'error':
            click.echo(f"✗ Error: {result.get('error', 'Unknown error')}", err=True)
        elif 'total_files' in result:
            click.echo(f"✓ Processed {result['total_files']} files")
            click.echo(f"  Success: {result['success']}")
            click.echo(f"  Errors: {result['errors']}")
            click.echo(f"  Total chunks: {result['total_chunks']}")
        else:
            click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)

@cli.command()
@click.argument('query')
@click.option('--strict-local', is_flag=True, help='Use only local knowledge base')
@click.option('--fast', is_flag=True, help='Fast mode (skip web search)')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.option('--no-progress', is_flag=True, help='Disable progress bar')
@click.option('--file', 'files', multiple=True, type=click.Path(exists=True), help='Attach files to query (can be used multiple times)')
def ask(query, strict_local, fast, output_json, no_progress, files):
    """Ask a question and get an answer from the RAG system"""
    try:
        workflow = get_rag_workflow()
        
        file_list = list(files) if files else None
        
        if output_json or no_progress:
            result = workflow.execute(query, strict_local=strict_local, fast_mode=fast, files=file_list)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Processing query...", total=None)
                
                def update_progress(stage: str):
                    progress.update(task, description=f"[cyan]{stage}[/cyan]")
                
                result = workflow.execute(
                    query, 
                    strict_local=strict_local, 
                    fast_mode=fast,
                    files=file_list,
                    progress_callback=update_progress
                )
        
        if output_json:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\nQuery: {result['query']}")
            click.echo(f"\nAnswer:\n{result['answer']}")
            click.echo(f"\nSources used: {', '.join(result['sources_used'])}")
            click.echo(f"Context documents: {result['context_count']}")
            click.echo(f"Latency: {result['latency_ms']:.2f}ms")
            
            if result.get('citations'):
                click.echo(f"\nCitations:")
                for citation in result['citations']:
                    click.echo(f"  {citation}")
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)

@cli.group()
def config():
    """Manage configuration"""
    pass

@config.command('get')
@click.argument('key')
def config_get(key):
    """Get a configuration value"""
    try:
        cfg = get_config()
        value = cfg.get(key)
        
        if value is None:
            click.echo(f"Key not found: {key}", err=True)
        else:
            click.echo(json.dumps(value, indent=2))
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)

@config.command('set')
@click.argument('key')
@click.argument('value')
def config_set(key, value):
    """Set a configuration value"""
    try:
        cfg = get_config()
        
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        
        cfg.set(key, parsed_value)
        cfg.save()
        
        click.echo(f"✓ Set {key} = {parsed_value}")
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)

@config.command('show')
def config_show():
    """Show all configuration"""
    try:
        cfg = get_config()
        click.echo(json.dumps(cfg.config, indent=2))
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)

@cli.command()
def status():
    """Check system status"""
    click.echo("Checking system status...")
    
    try:
        from rag_system.services.qdrant_service import get_qdrant_service
        from rag_system.services.elasticsearch_service import get_elasticsearch_service
        from rag_system.services.redis_service import get_redis_service
        
        try:
            qdrant = get_qdrant_service()
            click.echo("✓ Qdrant: Connected")
        except Exception as e:
            click.echo(f"✗ Qdrant: {str(e)}")
        
        try:
            es = get_elasticsearch_service()
            click.echo("✓ Elasticsearch: Connected")
        except Exception as e:
            click.echo(f"✗ Elasticsearch: {str(e)}")
        
        try:
            redis = get_redis_service()
            redis.client.ping()
            click.echo("✓ Redis: Connected")
        except Exception as e:
            click.echo(f"✗ Redis: {str(e)}")
        
        try:
            from rag_system.services.embeddings import get_embedding_service
            embeddings = get_embedding_service()
            click.echo(f"✓ Embeddings: Loaded (dimension: {embeddings.dimension})")
        except Exception as e:
            click.echo(f"✗ Embeddings: {str(e)}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)

if __name__ == '__main__':
    cli()
