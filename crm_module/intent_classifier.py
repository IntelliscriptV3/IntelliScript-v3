from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define your system prompt for intent identification
system_prompt = """You are an intent classification AI for an education institution management system.

    Your main task is to identify the intent of user queries and classify them into two categories:

    1. **STRUCTURED_RAG** - For queries about:
    - Student information and enrollment
    - Teacher details and courses
    - Attendance records
    - Fee information
    - Assessment results
    - Academic performance data
    - Class schedules
    - Student grades

    2. **UNSTRUCTURED_RAG** - For queries about:
    - Institution rules and policies
    - Procedures and processes
    - General guidelines
    - Code of conduct
    - Disciplinary actions
    - Administrative procedures
    - Campus facilities
    - General institutional information

    You must respond with ONLY one of these two classifications: "STRUCTURED_RAG" or "UNSTRUCTURED_RAG"

    Do not provide explanations or additional text - just the classification."""

class IntentClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        self.system_prompt = system_prompt

    def identify_intent(self, user_query):
        """
        Identify if the query requires structured or unstructured RAG

        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_query)
        ]
        response = self.llm.invoke(messages)

        # Clean and validate response
        intent = response.content.strip().upper()
        print(f"Identified Intent: {intent}")

        # Ensure we get a valid response
        if intent in ["STRUCTURED_RAG", "UNSTRUCTURED_RAG"]:
            return intent
        else:
            # Fallback logic based on keywords
            structured_keywords = [
                "student", "teacher", "attendance", "fee", "grade", "enrollment",
                "course", "assessment", "score", "marks", "schedule", "class"
            ]

            unstructured_keywords = [
                "rule", "policy", "procedure", "guideline", "conduct", "discipline",
                "punishment", "regulation", "process", "facility", "campus"
            ]

            query_lower = user_query.lower()

            structured_score = sum(1 for keyword in structured_keywords if keyword in query_lower)
            unstructured_score = sum(1 for keyword in unstructured_keywords if keyword in query_lower)

            return "STRUCTURED_RAG" if structured_score >= unstructured_score else "UNSTRUCTURED_RAG"

    def process_query_with_intent(self, user_query):
        """
        Process a query by first identifying intent, then providing the classification

        Returns:
            dict: Contains intent and recommended action
        """
        intent = self.identify_intent(user_query)

        result = {"intent": intent}
        if intent == "STRUCTURED_RAG":
            # TODO: Call structured RAG processing function here
            pass
        elif intent == "UNSTRUCTURED_RAG":
            # TODO: Call unstructured RAG processing function here
            pass
        else:
            result["recommendation"] = "Unable to determine intent."

        return result


# Example usage:
if __name__ == "__main__":
    classifier = IntentClassifier()
    
    # Test individual intent identification
    print("\n=== Direct Intent Testing ===")

    test_query = "What information do you need to enroll a new student?"
    intent = classifier.identify_intent(test_query)

    print(f"Query: {test_query}")
    print(f"Intent: {intent}")
