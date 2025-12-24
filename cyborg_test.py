"""
CyborgDB Integration - Encrypted Vector Database Operations
Handles encryption-in-use storage and similarity search for fraud detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import time
import json
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import os
import pickle
import secrets
import hashlib


class TokenManager:
    """Manages access tokens for vector database operations"""
    
    def __init__(self):
        self.tokens = {}  # {token: {'scope': scope, 'expiry': datetime, 'permissions': permissions}}
    
    def generate_token(self, scope: str, expiry_minutes: int = 60, permissions: List[str] = ['read']) -> str:
        """Generate a secure access token with scope and expiry"""
        token_data = f"{scope}_{str(datetime.now())}_{secrets.token_hex(16)}"
        token = hashlib.sha256(token_data.encode()).hexdigest()
        
        expiry_time = datetime.now() + timedelta(minutes=expiry_minutes)
        
        self.tokens[token] = {
            'scope': scope,
            'expiry': expiry_time,
            'permissions': permissions,
            'created': datetime.now()
        }
        
        return token
    
    def validate_token(self, token: str, required_permission: str = 'read') -> bool:
        """Validate token for scope and expiry"""
        if token not in self.tokens:
            return False
            
        token_info = self.tokens[token]
        
        # Check expiry
        if datetime.now() > token_info['expiry']:
            del self.tokens[token]  # Clean up expired token
            return False
        
        # Check permission
        if required_permission not in token_info['permissions']:
            return False
            
        return True
    
    def get_token_scope(self, token: str) -> Optional[str]:
        """Get the scope of a token"""
        if token not in self.tokens:
            return None
            
        if datetime.now() > self.tokens[token]['expiry']:
            del self.tokens[token]
            return None
            
        return self.tokens[token]['scope']


class CyborgDBSimulator:
    """
    Simulates CyborgDB encrypted vector database functionality
    Provides encryption-in-use capabilities for secure fraud detection
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None, dimension: int = 19):
        """
        Initialize CyborgDB simulator
        
        Args:
            encryption_key (bytes): Encryption key for data protection
            dimension (int): Vector dimension for embeddings
        """
        self.dimension = dimension
        
        # Initialize encryption
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        
        self.encryption_key = encryption_key
        self.cipher = Fernet(encryption_key)
        
        # In-memory vector storage (encrypted)
        self.vector_store = {}  # {id: encrypted_vector}
        self.metadata_store = {}  # {id: encrypted_metadata}
        self.index_mapping = {}  # {id: index}
        
        # No decrypted cache - all vectors remain encrypted
        # Decryption happens only during individual query operations
        
        # Performance metrics
        self.metrics = {
            'total_insertions': 0,
            'total_queries': 0,
            'avg_insert_latency': 0.0,
            'avg_query_latency': 0.0,
            'total_vectors': 0
        }
        
        # Token manager for access control
        self.token_manager = TokenManager()
        
        print(f"CyborgDB Simulator initialized with {dimension}-dimensional vectors")
        print(f"Encryption: ENABLED (Fernet symmetric encryption)")
        print(f"Token-based access control: ENABLED")
    
    def _encrypt_vector(self, vector: np.ndarray) -> bytes:
        """Encrypt a vector using Fernet encryption"""
        vector_bytes = pickle.dumps(vector)
        encrypted = self.cipher.encrypt(vector_bytes)
        return encrypted
    
    def _decrypt_vector(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt a vector"""
        decrypted_bytes = self.cipher.decrypt(encrypted_data)
        vector = pickle.loads(decrypted_bytes)
        return vector
    
    def _encrypt_metadata(self, metadata: Dict) -> bytes:
        """Encrypt metadata dictionary"""
        metadata_json = json.dumps(metadata)
        encrypted = self.cipher.encrypt(metadata_json.encode())
        return encrypted
    
    def _decrypt_metadata(self, encrypted_data: bytes) -> Dict:
        """Decrypt metadata dictionary"""
        decrypted_bytes = self.cipher.decrypt(encrypted_data)
        metadata = json.loads(decrypted_bytes.decode())
        return metadata
    
    def generate_access_token(self, scope: str = 'vector_search', expiry_minutes: int = 60) -> str:
        """Generate a token for accessing vector operations"""
        return self.token_manager.generate_token(scope, expiry_minutes, ['read'])
    
    def insert(self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict] = None, token: Optional[str] = None):
        """
        Insert encrypted vector into database
        
        Args:
            vector_id (str): Unique identifier for the vector
            vector (np.ndarray): Feature vector to store
            metadata (Dict): Optional metadata associated with vector
            token (str): Access token for authorization (not required for insert)
        """
        start_time = time.time()
        
        # Validate vector dimension
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match expected {self.dimension}")
        
        # Encrypt and store vector
        encrypted_vector = self._encrypt_vector(vector)
        self.vector_store[vector_id] = encrypted_vector
        
        # Encrypt and store metadata
        if metadata is None:
            metadata = {}
        metadata['inserted_at'] = datetime.now().isoformat()
        encrypted_metadata = self._encrypt_metadata(metadata)
        self.metadata_store[vector_id] = encrypted_metadata
        
        # Update index
        self.index_mapping[vector_id] = len(self.index_mapping)
        
        # Update metrics
        latency = time.time() - start_time
        self.metrics['total_insertions'] += 1
        self.metrics['total_vectors'] = len(self.vector_store)
        
        # Update average latency
        prev_avg = self.metrics['avg_insert_latency']
        n = self.metrics['total_insertions']
        self.metrics['avg_insert_latency'] = (prev_avg * (n - 1) + latency) / n

    def batch_insert(self, vectors: Dict[str, np.ndarray], metadata: Optional[Dict[str, Dict]] = None, token: Optional[str] = None):
        """
        Batch insert multiple vectors efficiently
        
        Args:
            vectors (Dict[str, np.ndarray]): Dictionary of {id: vector}
            metadata (Dict[str, Dict]): Optional dictionary of {id: metadata}
            token (str): Access token for authorization (not required for insert)
        """
        print(f"Batch inserting {len(vectors)} vectors...")
        start_time = time.time()
        
        for vector_id, vector in vectors.items():
            meta = metadata.get(vector_id, {}) if metadata else {}
            self.insert(vector_id, vector, meta, token)
        
        elapsed = time.time() - start_time
        print(f"Batch insert complete: {len(vectors)} vectors in {elapsed:.3f}s")
        print(f"Throughput: {len(vectors)/elapsed:.2f} vectors/sec")

    def similarity_search(self, query_vector: np.ndarray, k: int = 5, 
                         metric: str = 'cosine', token: Optional[str] = None) -> List[Tuple[str, float, Dict]]:
        """
        Search for k most similar vectors using token-based selective decryption
        Only decrypts vectors that are being compared during the search
        
        Args:
            query_vector (np.ndarray): Query vector
            k (int): Number of nearest neighbors to return
            metric (str): Distance metric ('cosine', 'euclidean', 'manhattan')
            token (str): Access token for authorization
            
        Returns:
            List[Tuple[str, float, Dict]]: List of (id, similarity_score, metadata)
        """
        # Validate token if provided
        if token:
            if not self.token_manager.validate_token(token, 'read'):
                raise PermissionError("Invalid or expired access token")
        
        start_time = time.time()
        
        if len(self.vector_store) == 0:
            return []
        
        # Calculate similarities by decrypting vectors one by one
        vector_ids = list(self.vector_store.keys())
        distances = []
        
        for vector_id in vector_ids:
            # Decrypt only the specific vector being compared
            encrypted_vector = self.vector_store[vector_id]
            vector = self._decrypt_vector(encrypted_vector)
            
            # Calculate distance based on metric
            if metric == 'cosine':
                # Cosine similarity
                query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
                vector_norm = vector / (np.linalg.norm(vector) + 1e-10)
                similarity = np.dot(query_norm, vector_norm)
                distance = similarity  # For cosine, higher is better
            elif metric == 'euclidean':
                # Euclidean distance (lower is better)
                distance = np.linalg.norm(vector - query_vector)
            elif metric == 'manhattan':
                # Manhattan distance (lower is better)
                distance = np.sum(np.abs(vector - query_vector))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            distances.append((vector_id, distance))
        
        # Sort by distance (similarity score)
        if metric == 'cosine':
            # For cosine similarity, higher is better
            distances.sort(key=lambda x: x[1], reverse=True)
            descending = True
        else:
            # For distance metrics, lower is better
            distances.sort(key=lambda x: x[1])
            descending = False
        
        # Get top k results
        k = min(k, len(distances))
        top_k_distances = distances[:k]
        
        # Prepare results
        results = []
        for vector_id, distance in top_k_distances:
            # Decrypt metadata
            encrypted_metadata = self.metadata_store[vector_id]
            metadata = self._decrypt_metadata(encrypted_metadata)
            
            results.append((vector_id, float(distance), metadata))
        
        # Update metrics
        latency = time.time() - start_time
        self.metrics['total_queries'] += 1
        
        prev_avg = self.metrics['avg_query_latency']
        n = self.metrics['total_queries']
        self.metrics['avg_query_latency'] = (prev_avg * (n - 1) + latency) / n
        
        return results
    
    def get_vector(self, vector_id: str, token: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Retrieve a specific vector by ID
        
        Args:
            vector_id (str): Vector identifier
            token (str): Access token for authorization
            
        Returns:
            Tuple[np.ndarray, Dict]: (vector, metadata)
        """
        # Validate token if provided
        if token:
            if not self.token_manager.validate_token(token, 'read'):
                raise PermissionError("Invalid or expired access token")
        
        if vector_id not in self.vector_store:
            raise KeyError(f"Vector {vector_id} not found")
        
        encrypted_vector = self.vector_store[vector_id]
        encrypted_metadata = self.metadata_store[vector_id]
        
        vector = self._decrypt_vector(encrypted_vector)
        metadata = self._decrypt_metadata(encrypted_metadata)
        
        return vector, metadata

    def delete(self, vector_id: str, token: Optional[str] = None):
        """Delete a vector from the database"""
        # For security reasons, we may want to validate tokens for delete operations
        # In a real system, this would be required
        if vector_id in self.vector_store:
            del self.vector_store[vector_id]
            del self.metadata_store[vector_id]
            del self.index_mapping[vector_id]
            self.metrics['total_vectors'] = len(self.vector_store)

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()

    def save_database(self, filepath: str, token: Optional[str] = None):
        """Save encrypted database to disk"""
        # Validate token if provided
        if token:
            if not self.token_manager.validate_token(token, 'read'):
                raise PermissionError("Invalid or expired access token")
        
        db_data = {
            'encryption_key': self.encryption_key,
            'dimension': self.dimension,
            'vector_store': self.vector_store,
            'metadata_store': self.metadata_store,
            'index_mapping': self.index_mapping,
            'metrics': self.metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(db_data, f)
        
        print(f"Database saved to {filepath}")

    def load_database(self, filepath: str, token: Optional[str] = None):
        """Load encrypted database from disk"""
        # Validate token if provided
        if token:
            if not self.token_manager.validate_token(token, 'read'):
                raise PermissionError("Invalid or expired access token")
        
        with open(filepath, 'rb') as f:
            db_data = pickle.load(f)
        
        self.encryption_key = db_data['encryption_key']
        self.cipher = Fernet(self.encryption_key)
        self.dimension = db_data['dimension']
        self.vector_store = db_data['vector_store']
        self.metadata_store = db_data['metadata_store']
        self.index_mapping = db_data['index_mapping']
        self.metrics = db_data['metrics']
        
        print(f"Database loaded from {filepath}")
        print(f"Total vectors: {len(self.vector_store)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_vectors': len(self.vector_store),
            'dimension': self.dimension,
            'total_insertions': self.metrics['total_insertions'],
            'total_queries': self.metrics['total_queries'],
            'avg_insert_latency_ms': self.metrics['avg_insert_latency'] * 1000,
            'avg_query_latency_ms': self.metrics['avg_query_latency'] * 1000,
            'encryption_enabled': True,
            'cache_status': 'no_cache'
        }


class FraudDetectionVectorDB:
    """
    High-level interface for fraud detection using encrypted vector database
    """
    
    def __init__(self, fraud_model, dimension: int = 19):
        """
        Initialize fraud detection vector database
        
        Args:
            fraud_model: Trained fraud detection model
            dimension (int): Vector dimension
        """
        self.db = CyborgDBSimulator(dimension=dimension)
        self.fraud_model = fraud_model
        
        print("Fraud Detection Vector DB initialized")
    
    def index_transactions(self, df: pd.DataFrame):
        """
        Index transactions into encrypted vector database
        
        Args:
            df (pd.DataFrame): Transaction dataframe
        """
        print(f"\nIndexing {len(df)} transactions into CyborgDB...")
        
        # Extract features and create embeddings
        features = self.fraud_model.extract_features(df)
        embeddings = self.fraud_model.create_embeddings(features)
        
        # Prepare vectors and metadata
        vectors = {}
        metadata = {}
        
        for idx, row in df.iterrows():
            vector_id = row.get('transaction_id', f'TXN_{idx}')
            vectors[vector_id] = embeddings[idx]
            
            metadata[vector_id] = {
                'transaction_id': vector_id,
                'amount': float(row.get('amount', 0)),  # type: ignore
                'merchant_category': str(row.get('merchant_category', 'unknown')),
                'timestamp': str(row.get('timestamp', '')),
                'is_fraud': int(row.get('is_fraud', 0))  # type: ignore
            }
        
        # Batch insert
        self.db.batch_insert(vectors, metadata)
        
        print(f"Indexing complete! {len(vectors)} vectors stored securely.")
    
    def detect_fraud(self, transaction: pd.DataFrame, k: int = 5) -> Dict[str, Any]:
        """
        Detect fraud by comparing with similar historical transactions
        Uses token-based access for secure similarity search
        
        Args:
            transaction (pd.DataFrame): Transaction to analyze
            k (int): Number of similar transactions to compare
            
        Returns:
            Dict[str, Any]: Fraud detection result
        """
        start_time = time.time()
        
        # Extract features and create embedding
        features = self.fraud_model.extract_features(transaction)
        embedding = self.fraud_model.create_embeddings(features)[0]
        
        # Get fraud prediction from model (on the input transaction, not from DB)
        predictions, probabilities = self.fraud_model.predict(transaction)
        
        # Generate a temporary access token for the similarity search
        search_token = self.db.generate_access_token(scope='vector_search', expiry_minutes=5)
        
        # Search for similar transactions using token-based access
        similar = self.db.similarity_search(embedding, k=k, metric='cosine', token=search_token)
        
        # Analyze similar transactions
        fraud_count = sum(1 for _, _, meta in similar if meta.get('is_fraud', 0) == 1)
        fraud_ratio = fraud_count / len(similar) if similar else 0
        
        # Calculate final fraud score
        model_score = float(probabilities[0])
        similarity_score = fraud_ratio
        final_score = 0.7 * model_score + 0.3 * similarity_score
        
        # Determine if fraud
        is_fraud = final_score > 0.5
        
        latency = time.time() - start_time
        
        result = {
            'is_fraud': bool(is_fraud),
            'fraud_score': float(final_score),
            'model_probability': float(model_score),
            'similarity_fraud_ratio': float(similarity_score),
            'similar_transactions': len(similar),
            'similar_fraud_count': int(fraud_count),
            'latency_ms': latency * 1000,
            'transaction_id': transaction.iloc[0].get('transaction_id', 'unknown'),
            'amount': float(transaction.iloc[0].get('amount', 0))
        }
        
        return result
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.db.get_stats()
    
    def save(self, db_path: str, model_path: str):
        """Save database and model"""
        self.db.save_database(db_path)
        self.fraud_model.save_model(model_path)
    
    def load(self, db_path: str, model_path: str):
        """Load database and model"""
        self.db.load_database(db_path)
        self.fraud_model.load_model(model_path)


if __name__ == "__main__":
    print("=== CyborgDB Encryption Test ===\n")
    
    # Test encryption
    db = CyborgDBSimulator(dimension=5)
    
    # Test vector insertion
    test_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_metadata = {'type': 'test', 'value': 123}
    
    db.insert('test_001', test_vector, test_metadata)
    print("✓ Vector inserted and encrypted")
    
    # Test retrieval
    retrieved_vector, retrieved_metadata = db.get_vector('test_001')
    print(f"✓ Vector retrieved and decrypted")
    print(f"  Original:  {test_vector}")
    print(f"  Retrieved: {retrieved_vector}")
    print(f"  Match: {np.allclose(test_vector, retrieved_vector)}")
    
    # Test similarity search
    query = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    results = db.similarity_search(query, k=1)
    print(f"\n✓ Similarity search complete")
    print(f"  Query: {query}")
    print(f"  Top result: {results[0][0]} (score: {results[0][1]:.4f})")
    
    # Show stats
    stats = db.get_stats()
    print(f"\n=== Database Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
