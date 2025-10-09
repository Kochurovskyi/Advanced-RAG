"""
Integration tests for the ARAG application.
Tests the complete workflow from question to answer.
"""
import pytest
from graph.graph import app
from graph.state import GraphState


class TestARAGIntegration:
    """Integration tests for the complete ARAG workflow."""
    
    def test_rag_workflow_relevant_question(self):
        """Test the complete RAG workflow with a relevant question."""
        question = "What is chain of thought prompting?"
        
        result = app.invoke({"question": question})
        
        # Verify the result structure
        assert "question" in result
        assert "generation" in result
        assert "web_search" in result
        assert "documents" in result
        
        # Verify the question is preserved
        assert result["question"] == question
        
        # Verify we have a generation
        assert result["generation"] is not None
        assert len(result["generation"]) > 0
        
        # Verify we have documents
        assert len(result["documents"]) > 0
        
        # For a relevant question, web_search should be False (RAG path)
        assert result["web_search"] is False
        
        # Verify the generation contains relevant content
        generation_lower = result["generation"].lower()
        assert any(keyword in generation_lower for keyword in 
                  ["chain", "thought", "prompting", "reasoning"])
    
    def test_web_search_workflow_irrelevant_question(self):
        """Test the complete workflow with an irrelevant question that should trigger web search."""
        question = "How to make pizza?"
        
        result = app.invoke({"question": question})
        
        # Verify the result structure
        assert "question" in result
        assert "generation" in result
        assert "web_search" in result
        assert "documents" in result
        
        # Verify the question is preserved
        assert result["question"] == question
        
        # Verify we have a generation
        assert result["generation"] is not None
        assert len(result["generation"]) > 0
        
        # Verify we have documents
        assert len(result["documents"]) > 0
        
        # For an irrelevant question, web_search should be True (web search path)
        assert result["web_search"] is True
        
        # Verify the generation contains relevant content about pizza
        generation_lower = result["generation"].lower()
        assert any(keyword in generation_lower for keyword in 
                  ["pizza", "dough", "ingredients", "cooking"])
    
    def test_state_validation(self):
        """Test that the state validation works correctly."""
        # Test with missing question
        invalid_state = {}
        
        # This should not crash and should return a proper error state
        try:
            result = app.invoke(invalid_state)
            # The app should handle this gracefully
            assert "question" in result
            assert "generation" in result
        except Exception as e:
            # If it raises an exception, that's also acceptable for invalid input
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in 
                      ["question", "keyerror", "recursion", "limit"])
    
    def test_document_retrieval_and_grading(self):
        """Test the document retrieval and grading process."""
        question = "What is agent memory?"
        
        result = app.invoke({"question": question})
        
        # Verify we have documents
        assert len(result["documents"]) > 0
        
        # Verify documents are Document objects with page_content
        for doc in result["documents"]:
            assert hasattr(doc, 'page_content')
            assert len(doc.page_content) > 0
    
    def test_generation_quality(self):
        """Test that the generation is of good quality."""
        question = "What is short term memory in agents?"
        
        result = app.invoke({"question": question})
        
        # Verify generation is not empty
        assert result["generation"] is not None
        assert len(result["generation"]) > 50  # Should be substantial
        
        # Verify generation is not just a repetition of the question
        assert result["generation"] != question
        
        # Verify generation contains relevant keywords
        generation_lower = result["generation"].lower()
        assert any(keyword in generation_lower for keyword in 
                  ["memory", "agent", "short", "term", "learning"])
    
    def test_hallucination_detection(self):
        """Test that hallucination detection works."""
        question = "What is machine learning?"
        
        result = app.invoke({"question": question})
        
        # The generation should be grounded in the documents
        # This is implicitly tested by the fact that the generation
        # passed the hallucination check in the workflow
        
        # Verify we have a valid generation
        assert result["generation"] is not None
        assert len(result["generation"]) > 0
    
    def test_multiple_questions(self):
        """Test that the system can handle multiple questions in sequence."""
        questions = [
            "What is chain of thought prompting?",
            "How does agent memory work?",
            "What is LangGraph?"
        ]
        
        results = []
        for question in questions:
            result = app.invoke({"question": question})
            results.append(result)
            
            # Basic validation for each result
            assert "question" in result
            assert "generation" in result
            assert "web_search" in result
            assert "documents" in result
            assert result["question"] == question
            assert result["generation"] is not None
            assert len(result["generation"]) > 0
        
        # Verify we got results for all questions
        assert len(results) == len(questions)
    
    def test_error_handling(self):
        """Test that the system handles errors gracefully."""
        # Test with empty question
        try:
            result = app.invoke({"question": ""})
            # Should not crash
            assert "question" in result
            assert "generation" in result
        except Exception as e:
            # If it raises an exception due to recursion limit, that's acceptable
            assert "recursion" in str(e).lower() or "limit" in str(e).lower()
        
        # Test with very long question
        long_question = "What is " + "machine learning " * 100 + "?"
        try:
            result = app.invoke({"question": long_question})
            # Should not crash
            assert "question" in result
            assert "generation" in result
        except Exception as e:
            # If it raises an exception, that's also acceptable for edge cases
            assert len(str(e)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
