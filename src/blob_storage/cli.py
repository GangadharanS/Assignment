#!/usr/bin/env python3
"""
Local Blob Storage CLI Application

A command-line interface for uploading and managing documents in local blob storage.
Supports Azurite (Azure Storage Emulator) and local filesystem storage.
"""
import json
import sys
from pathlib import Path

import click

from blob_storage.config import config
from blob_storage.storage import get_blob_storage


@click.group()
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["azurite", "local"]),
    default=None,
    help="Storage mode (overrides config)",
)
@click.pass_context
def cli(ctx, mode):
    """
    Local Blob Storage CLI - Upload and manage documents in local blob storage.
    
    \b
    Storage Modes:
      azurite - Use Azurite (Azure Storage Emulator)
      local   - Use local filesystem storage
    """
    ctx.ensure_object(dict)
    ctx.obj["mode"] = mode


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--name", "-n", help="Custom blob name (defaults to filename)")
@click.option("--container", "-c", help="Target container name")
@click.pass_context
def upload(ctx, file_path, name, container):
    """Upload a document to blob storage."""
    try:
        storage = get_blob_storage(ctx.obj.get("mode"))
        result = storage.upload_file(file_path, blob_name=name, container_name=container)
        
        click.echo("\n✓ Upload successful!")
        click.echo(f"  Blob Name: {result['blob_name']}")
        click.echo(f"  Container: {result['container']}")
        click.echo(f"  Size: {result['size']:,} bytes")
        click.echo(f"  ETag: {result['etag']}")
        click.echo(f"  Modified: {result['last_modified']}")
        
        if "url" in result:
            click.echo(f"  URL: {result['url']}")
        elif "path" in result:
            click.echo(f"  Path: {result['path']}")
            
    except FileNotFoundError as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Upload failed: {e}", err=True)
        click.echo("\nIf using Azurite, make sure it's running:", err=True)
        click.echo("  azurite --silent --location ./azurite-data", err=True)
        sys.exit(1)


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--pattern", "-p", default="*", help="File pattern to match (e.g., '*.pdf')")
@click.option("--container", "-c", help="Target container name")
@click.option("--recursive", "-r", is_flag=True, help="Include subdirectories")
@click.pass_context
def upload_dir(ctx, directory, pattern, container, recursive):
    """Upload all documents from a directory."""
    try:
        storage = get_blob_storage(ctx.obj.get("mode"))
        dir_path = Path(directory)
        
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
        
        # Filter out directories
        files = [f for f in files if f.is_file()]
        
        if not files:
            click.echo(f"No files matching '{pattern}' found in {directory}")
            return
        
        click.echo(f"\nUploading {len(files)} file(s)...")
        
        for file_path in files:
            # Preserve relative path structure for blob name
            if recursive:
                blob_name = str(file_path.relative_to(dir_path))
            else:
                blob_name = file_path.name
            
            result = storage.upload_file(
                str(file_path),
                blob_name=blob_name,
                container_name=container,
            )
            click.echo(f"  ✓ {result['blob_name']} ({result['size']:,} bytes)")
        
        click.echo(f"\n✓ All {len(files)} file(s) uploaded successfully!")
        
    except Exception as e:
        click.echo(f"✗ Upload failed: {e}", err=True)
        sys.exit(1)


@cli.command("list")
@click.option("--container", "-c", help="Container name")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def list_blobs(ctx, container, json_output):
    """List all blobs in a container."""
    try:
        storage = get_blob_storage(ctx.obj.get("mode"))
        blobs = storage.list_blobs(container_name=container)
        
        if json_output:
            click.echo(json.dumps(blobs, indent=2))
        else:
            container_name = container or config.BLOB_CONTAINER_NAME
            click.echo(f"\nBlobs in '{container_name}':")
            click.echo("-" * 60)
            
            if not blobs:
                click.echo("  (empty)")
            else:
                for blob in blobs:
                    size_kb = blob["size"] / 1024
                    click.echo(f"  {blob['name']:<40} {size_kb:>8.1f} KB")
            
            click.echo("-" * 60)
            click.echo(f"Total: {len(blobs)} blob(s)")
            
    except Exception as e:
        click.echo(f"✗ Error listing blobs: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("blob_name")
@click.argument("destination", type=click.Path())
@click.option("--container", "-c", help="Container name")
@click.pass_context
def download(ctx, blob_name, destination, container):
    """Download a blob to a local file."""
    try:
        storage = get_blob_storage(ctx.obj.get("mode"))
        success = storage.download_file(blob_name, destination, container_name=container)
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"✗ Download failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("blob_name")
@click.option("--container", "-c", help="Container name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx, blob_name, container, yes):
    """Delete a blob from storage."""
    if not yes:
        if not click.confirm(f"Delete blob '{blob_name}'?"):
            click.echo("Cancelled.")
            return
    
    try:
        storage = get_blob_storage(ctx.obj.get("mode"))
        success = storage.delete_blob(blob_name, container_name=container)
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"✗ Delete failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("container_name")
@click.pass_context
def create_container(ctx, container_name):
    """Create a new container."""
    try:
        storage = get_blob_storage(ctx.obj.get("mode"))
        storage.create_container(container_name)
        
    except Exception as e:
        click.echo(f"✗ Failed to create container: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show storage status and configuration."""
    mode = ctx.obj.get("mode") or config.STORAGE_MODE
    
    click.echo("\n📦 Local Blob Storage Status")
    click.echo("=" * 40)
    click.echo(f"  Storage Mode: {mode}")
    click.echo(f"  Default Container: {config.BLOB_CONTAINER_NAME}")
    
    if mode == "local":
        click.echo(f"  Storage Path: {config.LOCAL_STORAGE_PATH}")
    else:
        click.echo("  Endpoint: http://127.0.0.1:10000")
    
    # Test connection
    click.echo("\nConnection Test:")
    try:
        storage = get_blob_storage(mode)
        blobs = storage.list_blobs()
        click.echo("  ✓ Connected successfully")
        click.echo(f"  ✓ {len(blobs)} blob(s) in default container")
    except Exception as e:
        click.echo(f"  ✗ Connection failed: {e}")
        if mode == "azurite":
            click.echo("\n  Tip: Make sure Azurite is running:")
            click.echo("    npm install -g azurite")
            click.echo("    azurite --silent --location ./azurite-data")


@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, type=int, help="Port to bind to")
@click.option("--reload", "-r", is_flag=True, help="Enable auto-reload for development")
@click.pass_context
def serve(ctx, host, port, reload):
    """Start the REST API server."""
    mode = ctx.obj.get("mode") or config.STORAGE_MODE
    
    click.echo(f"\n🚀 Starting Blob Storage API Server")
    click.echo("=" * 40)
    click.echo(f"  Storage Mode: {mode}")
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Reload: {'enabled' if reload else 'disabled'}")
    click.echo(f"\n  API Docs: http://{host}:{port}/docs")
    click.echo(f"  OpenAPI:  http://{host}:{port}/openapi.json")
    click.echo("=" * 40 + "\n")
    
    # Set storage mode in environment for the API
    import os
    if mode:
        os.environ["STORAGE_MODE"] = mode
    
    from blob_storage.api import run_server
    run_server(host=host, port=port, reload=reload)


@cli.command()
def info():
    """Show help information about this tool."""
    click.echo("""
╔════════════════════════════════════════════════════════════╗
║           Local Blob Storage - Document Manager            ║
╚════════════════════════════════════════════════════════════╝

This tool allows you to upload and manage documents in local blob storage.

STORAGE OPTIONS:
  
  1. Azurite (Azure Storage Emulator) - DEFAULT
     - Full Azure Blob Storage API compatibility
     - Install: npm install -g azurite
     - Start:   azurite --silent --location ./azurite-data
  
  2. Local Filesystem
     - Simple file-based storage
     - No additional setup required
     - Use: --mode local or set STORAGE_MODE=local

QUICK START (CLI):

  1. Install dependencies:
     pip install -e .

  2. Upload a document:
     blob-storage --mode local upload ./document.pdf

  3. List all documents:
     blob-storage --mode local list

  4. Download a document:
     blob-storage --mode local download document.pdf ./downloaded.pdf

QUICK START (REST API):

  1. Start the API server:
     blob-storage --mode local serve

  2. Open API docs in browser:
     http://localhost:8000/docs

  3. Upload via curl:
     curl -X POST -F "file=@document.pdf" http://localhost:8000/upload

CONFIGURATION:
  
  Copy env.example to .env and customize settings:
  - STORAGE_MODE: "azurite" or "local"
  - BLOB_CONTAINER_NAME: Default container name
  - LOCAL_STORAGE_PATH: Path for local filesystem storage
""")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()

