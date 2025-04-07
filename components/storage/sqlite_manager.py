import sqlite3
import os
from typing import List
from components.models import DocumentFile, Chunk, Embedding

class SQLiteManager:

    def __init__(self, 
                 db_path: str = None,
                 schema_path: str = None
    ):
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Set default paths relative to the project root
        if db_path is None:
            self.db_path = os.path.join(project_root, "databases", "public", "public.db")
        else:
            self.db_path = db_path
            
        if schema_path is None:
            self.schema_path = os.path.join(project_root, "databases", "schemas", "schema.sql")
        else:
            self.schema_path = schema_path
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        

    def initialize_database(self) -> None:
        try:
            with open(self.schema_path, "r") as f:
                schema = f.read()

            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema)
                conn.commit()

            print(f"Database initialized successfully at {self.db_path}")

        except FileNotFoundError:
            print(f"Error: Schema file not found at {self.schema_path}")
            raise FileNotFoundError(f"Schema file not found at {self.schema_path}")
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")
            raise e
                
    def get_connection(self) -> sqlite3.Connection:
        # Check if database exists, if not initialize it
        if not os.path.exists(self.db_path):
            print(f"Database not found at {self.db_path}. Initializing...")
            self.initialize_database()
            
        return sqlite3.connect(self.db_path)
    
    def begin(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN TRANSACTION")
    

    def insert_document_file(self, file: DocumentFile, conn: sqlite3.Connection) -> None:
        try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO document_files (name, hash, path, total_pages) VALUES (?, ?, ?, ?)", 
                    (file.name, file.hash, file.path, file.total_pages)
                )
                file.id = cursor.lastrowid

                print(f"Document file {file.name} inserted successfully")
                return cursor.lastrowid
        
        except sqlite3.Error as e:
            print(f"Error inserting document file: {e}")
            raise e
        
    def insert_chunk(self, chunk: Chunk, file_id: int, conn: sqlite3.Connection) -> None:
        try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO chunks (document_id, page_number, content, chunk_page_index, chunk_start_char_position) VALUES (?, ?, ?, ?, ?)", 
                    (file_id, chunk.page_number, chunk.content, chunk.chunk_page_index, chunk.chunk_start_char_position)
                )
                chunk.id = cursor.lastrowid

                return chunk.id
        
        except sqlite3.Error as e:
            print(f"Error inserting chunks: {e}")
            raise e
        
    def insert_embedding(self, embedding: Embedding, conn: sqlite3.Connection) -> None:
        try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO embeddings (chunk_id, faiss_index_path, chunk_faiss_index, dimension) VALUES (?, ?, ?, ?)", 
                    (embedding.chunk_id, embedding.faiss_index_path, embedding.chunk_faiss_index, embedding.dimension)
                )
                embedding.id = cursor.lastrowid

                return embedding.id
        
        except sqlite3.Error as e:
            print(f"Error inserting embedding: {e}")
            raise e
                
                
    
    
    