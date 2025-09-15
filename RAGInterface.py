"""
RAG Interface for Agent Integration

This module provides a clean interface for external agents to interact with the RAG system.
It wraps the QueryOrchestrator and DomainManager, hiding internal implementation details
like FAISS, SQLite, and HuggingFace API interactions.

Usage:
    from RAGInterface import RAGInterface
    
    rag = RAGInterface(config_path="config.toml")
    result = rag.query_llm("What is the refund policy?", domains=["Finance"])
    print(result["answer"])
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.config import AppConfig
from src.config.config_manager import ConfigManager
from src.models import Domain, Chunk
from src.query_processing.query_orchestrator import QueryOrchestrator
from src.utils.domain_manager import DomainManager
from src.utils.sqlite_manager import SQLiteManager
from src.utils.logger import get_logger


class RAGInterfaceError(Exception):
    """Base exception for RAG Interface errors."""
    pass


class RAGInterface:
    """
    Main interface for agent integration with the RAG system.
    
    Provides a clean API for querying the vector store, managing domains,
    and generating answers using the LLM.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[AppConfig] = None):
        """
        Initialize the RAG Interface.
        
        Args:
            config_path: Path to config.toml file. If None, uses default path.
            config: Pre-configured AppConfig instance. If provided, config_path is ignored.
            
        Raises:
            RAGInterfaceError: If initialization fails.
        """
        self.logger = get_logger(__name__, log_domain="RAG Interface")
        self.logger.info("Initializing RAG Interface")
        
        # Initialize components
        self.config_manager = None
        self.config = None
        self.sqlite_manager = None
        self.domain_manager = None
        self.query_orchestrator = None
        
        try:
            if config is not None:
                self.config = config
                self.logger.info("Using provided AppConfig")
            else:
                # Load configuration
                config_path = config_path or "config.toml"
                self.config_manager = ConfigManager(Path(config_path))
                self.config = self.config_manager.get_config()
                self.logger.info(f"Loaded configuration from {config_path}")
            
            # Initialize shared SQLite manager
            self.sqlite_manager = SQLiteManager(self.config.system, log_domain="RAG Interface")
            
            # Initialize domain manager
            self.domain_manager = DomainManager(
                config=self.config,
                sqlite_manager=self.sqlite_manager,
                log_domain="RAG Interface"
            )
            
            # Initialize query orchestrator
            self.query_orchestrator = QueryOrchestrator(self.config, self.sqlite_manager)
            
            self.logger.info("RAG Interface initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Interface: {e}", exc_info=True)
            raise RAGInterfaceError(f"Initialization failed: {e}") from e

    def query_llm(self, question: str, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Query the RAG system and generate an answer using the LLM.
        
        Args:
            question: The question to ask.
            domains: Optional list of domain names to search. If None, auto-selects domains.
            
        Returns:
            Dictionary containing:
                - answer: Generated answer string
                - question: Original question
                - retrieved_chunks: Number of chunks retrieved
                - embedding_model: Model used for embeddings
                - faiss_index_type: Type of FAISS index used
                - success: Boolean indicating success
                - processing_duration: Time taken for processing
                - selected_domains: List of domain names used (if auto-selected)
                
        Raises:
            RAGInterfaceError: If query fails.
            ValueError: If question is empty or invalid.
        """
        self.logger.info("Processing LLM query", question=question, domains=domains)
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            result = self.query_orchestrator.query_llm(question, domains)
            
            # Add selected domains info if auto-selected
            if domains is None and "selected_domains" not in result:
                # Extract domain names from the orchestrator's internal state if available
                result["selected_domains"] = domains or []
            
            self.logger.info("LLM query completed successfully", 
                           success=result.get("success", False),
                           duration=result.get("processing_duration"))
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}", exc_info=True)
            raise RAGInterfaceError(f"Query failed: {e}") from e

    def retrieve_chunks(self, question: str, domains: Optional[List[str]] = None, k: Optional[int] = None) -> List[Chunk]:
        """
        Retrieve relevant chunks without generating an LLM answer.
        
        Args:
            question: The question to search for.
            domains: Optional list of domain names to search. If None, auto-selects domains.
            k: Number of chunks to retrieve. If None, uses default from config.
            
        Returns:
            List of Chunk objects containing relevant content.
            
        Raises:
            RAGInterfaceError: If retrieval fails.
            ValueError: If question is empty or invalid.
        """
        self.logger.info("Retrieving chunks", question=question, domains=domains, k=k)
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            # Use the orchestrator's internal methods for retrieval
            selected_domains = self.query_orchestrator._select_domains(question, domains)
            
            all_chunks = []
            for domain in selected_domains:
                query_embedding = self.query_orchestrator._process_query(question, domain)
                domain_chunks = self.query_orchestrator._retrieve_documents(query_embedding, domain)
                all_chunks.extend(domain_chunks)
            
            # Limit to k if specified
            if k is not None and len(all_chunks) > k:
                all_chunks = all_chunks[:k]
            
            self.logger.info(f"Retrieved {len(all_chunks)} chunks successfully")
            return all_chunks
            
        except Exception as e:
            self.logger.error(f"Chunk retrieval failed: {e}", exc_info=True)
            raise RAGInterfaceError(f"Retrieval failed: {e}") from e

    def get_config(self) -> AppConfig:
        """
        Get the current configuration.
        
        Returns:
            Current AppConfig instance.
        """
        return self.config

    def update_config(self, new_config: AppConfig) -> None:
        """
        Update the configuration and propagate to all components.
        
        Args:
            new_config: New configuration to apply.
            
        Raises:
            RAGInterfaceError: If config update fails.
        """
        self.logger.info("Updating configuration")
        
        try:
            self.config = new_config
            
            # Update all components
            self.sqlite_manager.update_config(new_config.system)
            self.domain_manager.update_config(new_config)
            self.query_orchestrator.update_config(new_config)
            
            self.logger.info("Configuration updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}", exc_info=True)
            raise RAGInterfaceError(f"Config update failed: {e}") from e

    def reload_config(self) -> AppConfig:
        """
        Reload configuration from file and apply to all components.
        
        Returns:
            Reloaded AppConfig instance.
            
        Raises:
            RAGInterfaceError: If config reload fails.
        """
        self.logger.info("Reloading configuration from file")
        
        if self.config_manager is None:
            raise RAGInterfaceError("No config manager available for reload")
        
        try:
            new_config = self.config_manager.get_config()
            self.update_config(new_config)
            
            self.logger.info("Configuration reloaded successfully")
            return new_config
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}", exc_info=True)
            raise RAGInterfaceError(f"Config reload failed: {e}") from e

    def health(self) -> Dict[str, Any]:
        """
        Check the health of the RAG system.
        
        Returns:
            Dictionary containing health status and diagnostic information:
                - ok: Boolean indicating overall health
                - checks: Dictionary of individual component checks
                - timestamp: When the health check was performed
        """
        self.logger.info("Performing health check")
        
        checks = {}
        overall_ok = True
        
        try:
            # Check control database
            try:
                with self.sqlite_manager.get_connection(control=True) as conn:
                    domains = self.sqlite_manager.get_domain(conn)
                    checks["control_db"] = {
                        "ok": True,
                        "domains_count": len(domains) if domains else 0
                    }
            except Exception as e:
                checks["control_db"] = {"ok": False, "error": str(e)}
                overall_ok = False
            
            # Check storage path
            storage_path = Path(self.config.system.storage_base_path)
            checks["storage_path"] = {
                "ok": storage_path.exists(),
                "path": str(storage_path),
                "writable": storage_path.is_dir() and os.access(storage_path, os.W_OK)
            }
            if not checks["storage_path"]["ok"]:
                overall_ok = False
            
            # Check HuggingFace token
            hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
            checks["huggingface_token"] = {
                "ok": hf_token is not None,
                "present": hf_token is not None
            }
            if not checks["huggingface_token"]["ok"]:
                overall_ok = False
            
            # Check domains with populated data
            try:
                domains = self.list_domains()
                populated_domains = []
                for domain in domains:
                    if domain.db_path and os.path.exists(domain.db_path):
                        populated_domains.append(domain.name)
                
                checks["populated_domains"] = {
                    "ok": len(populated_domains) > 0,
                    "count": len(populated_domains),
                    "domains": populated_domains
                }
                if not checks["populated_domains"]["ok"]:
                    overall_ok = False
                    
            except Exception as e:
                checks["populated_domains"] = {"ok": False, "error": str(e)}
                overall_ok = False
            
            result = {
                "ok": overall_ok,
                "checks": checks,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info("Health check completed", overall_ok=overall_ok)
            return result
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "ok": False,
                "checks": {"health_check": {"ok": False, "error": str(e)}},
                "timestamp": datetime.now().isoformat()
            }

    def close(self) -> None:
        """
        Close the RAG interface and clean up resources.
        
        Currently a no-op, but can be extended for cleanup if needed.
        """
        self.logger.info("Closing RAG Interface")
        # Add any cleanup logic here if needed in the future
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
