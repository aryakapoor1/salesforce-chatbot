import json
import os
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class GetRelevantFiles:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2',
                 file_descriptions_path = "file_descriptions.json"):
        try:
            # Force CPU usage to avoid device issues
            device = 'cpu'
            self.model = SentenceTransformer(model_name, device=device)
            
            # Check if file_descriptions.json exists
            if not os.path.exists(file_descriptions_path):
                # Create a minimal default file descriptions
                self.file_descriptions = {
                    "organization-info.json": "Contains organization and company information",
                    "global-objects.json": "Contains custom and standard Salesforce objects",
                    "salesforce_complete_data.json": "Complete Salesforce data export"
                }
            else:
                with open(file_descriptions_path, 'r') as f:
                    self.file_descriptions = json.load(f)
            
            self.file_names = list(self.file_descriptions.keys())
            descriptions = list(self.file_descriptions.values())
            self.description_embeddings = self.model.encode(descriptions)
            
        except Exception as e:
            print(f"Warning: Could not initialize semantic search model: {e}")
            # Fallback to simple matching
            self.model = None
            self.file_descriptions = {
                "organization-info.json": "Contains organization and company information",
                "global-objects.json": "Contains custom and standard Salesforce objects", 
                "salesforce_complete_data.json": "Complete Salesforce data export"
            }
            self.file_names = list(self.file_descriptions.keys())
            self.description_embeddings = None


    def get_top_similar_files(self,
                              question: str,
                              top_k: int = 4) -> List[Tuple[str, float]]:
          
        if not self.file_descriptions:
            return []

        # If semantic model failed to load, use simple keyword matching
        if self.model is None or self.description_embeddings is None:
            return self._simple_keyword_matching(question, top_k)

        try:
            question_embedding = self.model.encode([question])
            similarities = cosine_similarity(question_embedding,
                                             self.description_embeddings)[0]
            file_similarities = list(zip(self.file_names, similarities))
            top_files = sorted(file_similarities,
                               key=lambda x: x[1], reverse=True)[:top_k]
            return top_files
        except Exception as e:
            print(f"Warning: Semantic search failed: {e}")
            return self._simple_keyword_matching(question, top_k)
    
    def _simple_keyword_matching(self, question: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """Fallback method using simple keyword matching"""
        question_lower = question.lower()
        file_scores = []
        
        for filename, description in self.file_descriptions.items():
            score = 0.0
            description_lower = description.lower()
            
            # Simple keyword matching
            keywords = ['organization', 'company', 'objects', 'custom', 'fields', 'data']
            for keyword in keywords:
                if keyword in question_lower and keyword in description_lower:
                    score += 0.3
            
            # Exact word matches
            question_words = question_lower.split()
            for word in question_words:
                if len(word) > 3 and word in description_lower:
                    score += 0.1
            
            file_scores.append((filename, score))
        
        # Sort by score and return top_k
        top_files = sorted(file_scores, key=lambda x: x[1], reverse=True)[:top_k]
        return top_files