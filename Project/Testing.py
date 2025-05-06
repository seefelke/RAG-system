import unittest
import Extraction


class RetrievalTesting(unittest.TestCase):

    def setUp(self):
        self.vectorstore = Extraction.get_vectorstore()
        self.retriever = self.vectorstore.as_retriever()

    def run_similarity_score(self, query):
        results = self.vectorstore.similarity_search_with_score(query)
        doc, score = results[0]
        return score

    def run_similarity_score_min(self, query, min_score):
        score = self.run_similarity_score(query)
        self.assertTrue(score >= min_score, "Score is less than min_score.")

    def run_lexical_search(self, query, keywords):
        results = self.retriever.invoke(query)
        contents = [doc.page_content for doc in results]
        for word in keywords:
            self.assertTrue(any(word in c.lower() for c in contents),
                            "Lexical search failed to find exact match.")

    def test_similarity_score(self):
        self.run_similarity_score_min("Welche Überlebensstrategien verwenden Tiere?", 0.1)

    def test_lexical_search(self):
        self.run_lexical_search("Welche Überlebensstrategien verwenden Tiere?", {"mimese", "mimikry"})