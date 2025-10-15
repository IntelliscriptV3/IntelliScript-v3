from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import argparse
from report_generation.app.main import ReportGeneration
from crm_module.intent_classifier import IntentClassifier
# Load environment variables from .env file

load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define your system prompt for intent identification
system_prompt = """You are an intent classification AI for an education institution management system.

    Your main task is to identify the intent of user queries and classify them into three categories:

    1. **CRM_MODULE** - For queries about:
    - Student information and enrollment
    - Teacher details and courses
    - Attendance records
    - Fee information
    - Assessment results
    - Academic performance data
    - Class schedules
    - Student grades
    - Institution rules and policies
    - Procedures and processes
    - General guidelines
    - Code of conduct
    - Disciplinary actions
    - Administrative procedures
    - Campus facilities
    - General institutional information

    2. **REPORT_GENERATION** - For queries requesting visualization or analysis of data such as:
    - Charts, graphs, or visual representations of data
    - Statistical analysis of student performance
    - Attendance trends and patterns
    - Fee collection reports
    - Grade distribution analysis
    - Performance comparisons
    - Data dashboards
    - Analytical reports
    - Visual summaries of academic data

    3. **OTHER** - For queries that don't fit into the above categories:
    - General conversations
    - Technical support questions
    - Non-education related queries
    - Unclear or ambiguous requests

    You must respond with ONLY one of these three classifications: "CRM_MODULE", "REPORT_GENERATION", or "OTHER"

    Do not provide explanations or additional text - just the classification."""

class Main:
    def __init__(self, user_id=1):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        self.system_prompt = system_prompt
        self.crm_module = IntentClassifier(user_id=user_id)
        self.report_module = ReportGeneration(user_id=user_id)
           

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
        if intent in ["CRM_MODULE", "REPORT_GENERATION", "OTHER"]:
            return intent
        else:
            # Fallback logic based on keywords
            crm_keywords = [
                "student", "teacher", "attendance", "fee", "grade", "enrollment",
                "course", "assessment", "score", "marks", "schedule", "class",
                "rule", "policy", "procedure", "guideline", "conduct", "discipline",
                "punishment", "regulation", "process", "facility", "campus"
            ]

            report_keywords = [
                "chart", "graph", "visualize", "plot", "dashboard", "report",
                "analysis", "trend", "pattern", "statistics", "distribution",
                "comparison", "analytics", "summary", "show data", "display"
            ]

            query_lower = user_query.lower()

            crm_score = sum(1 for keyword in crm_keywords if keyword in query_lower)
            report_score = sum(1 for keyword in report_keywords if keyword in query_lower)

            if report_score > 0:
                return "REPORT_GENERATION"
            elif crm_score > 0:
                return "CRM_MODULE"
            else:
                return "OTHER"


    def chat(self, user_query):
        """
        Process a query by first identifying intent, then providing the classification

        Returns:
            dict: Contains intent and recommended action
        """
        intent = self.identify_intent(user_query)

        print(f"Final Intent: {intent}")

        if intent == "CRM_MODULE":
            # For CRM queries, use both structured and unstructured RAG
            # First try structured (database) queries
            response = self.crm_module.chat(user_query)
            print("crm response: ", response)
                
        elif intent == "REPORT_GENERATION":
            # TODO: Call report generation module here
            print("\n📊 Running Report Generation...\n")
            response = self.report_module.chat_endpoint(user_query)
            print("report response: ", response)
            
        elif intent == "OTHER":
            response = "I can help with student information, academic data, institutional policies, or generate reports. Please ask a more specific question related to the education management system."
            
        else:
            response = "Error: Unable to determine intent." 


            # Set response
        return response


# Example usage:
if __name__ == "__main__":
    classifier = Main(user_id=1)
    
    # Test individual intent identification
    print("\n=== Direct Intent Testing ===")

    test_query = "visualize the attendance of aiden adams for the last month"
    intent = classifier.chat(test_query)

    print(f"Query: {test_query}")
    print(f"Intent_this model: {intent}")
