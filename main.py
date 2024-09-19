import sys
import os
from venv import logger
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QListWidget, QStackedWidget,
                             QProgressBar)
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import json
from langchain.schema import AIMessage

from dotenv import load_dotenv

# Import from previous steps
from Data_Ingestion_and_Knowledge_Graph_Creation import Neo4jConnector
from Query_Engine_and_Multi_Agent_Architecture import Orchestrator
from LLM_Integration_and_Response_Generation import LLMIntegration, ResponseGenerator
from System_Robustness_and_Performance import LoadBalancer, cached_query_execution

# Load environment variables
load_dotenv()

class QueryWorker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, load_balancer, query):
        super().__init__()
        self.load_balancer = load_balancer
        self.query = query

    def run(self):
        try:
            result = cached_query_execution(self.query, self.load_balancer)
        except Exception as e:
            result = {
                "error": f"An error occurred while processing the query: {str(e)}",
                "enhanced_response": {
                    "summary": "Error in query processing",
                    "detailed_answer": "The system encountered an error while processing your query.",
                    "confidence": "low",
                    "follow_up_questions": []
                }
            }
        self.finished.emit(result)

class KnowledgeGraphUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Knowledge Graph Assistant")
        self.setGeometry(100, 100, 1200, 800)
        
        self.load_balancer = self.initialize_system()
        
        if self.load_balancer is None:
            print("Failed to initialize LoadBalancer")
            # Handle the error appropriately, e.g., show an error message to the user
        else:
            self.init_ui()

    def initialize_system(self):
        print("Starting system initialization")
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        print(f"Neo4j URI: {neo4j_uri}")

        try:
            load_balancer = LoadBalancer(neo4j_uri, neo4j_user, neo4j_password)
            load_balancer.initialize()
            print("LoadBalancer initialized")
            return load_balancer
        except Exception as e:
            print(f"Failed to initialize system: {str(e)}")
            return None

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel for query input and history
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Query input
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Enter your query here...")
        self.query_input.setFont(QFont("Arial", 12))
        left_layout.addWidget(self.query_input)

        # Submit button and loader
        button_loader_layout = QHBoxLayout()
        self.submit_button = QPushButton("Submit Query")
        self.submit_button.clicked.connect(self.submit_query)
        button_loader_layout.addWidget(self.submit_button)

        self.loader = QProgressBar()
        self.loader.setRange(0, 0)  # Makes it an "infinite" progress bar
        self.loader.setVisible(False)
        button_loader_layout.addWidget(self.loader)

        left_layout.addLayout(button_loader_layout)

        # Query history
        history_label = QLabel("Query History")
        history_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        left_layout.addWidget(history_label)

        self.query_history = QListWidget()
        self.query_history.itemClicked.connect(self.load_query)
        left_layout.addWidget(self.query_history)

        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel, 1)

        # Right panel for results
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Tabs for different result views
        self.result_tabs = QStackedWidget()

        # Summary view
        summary_widget = QWidget()
        summary_layout = QVBoxLayout()
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Arial", 14))  # Increased font size
        summary_layout.addWidget(self.summary_text)
        summary_widget.setLayout(summary_layout)
        self.result_tabs.addWidget(summary_widget)

        # Detailed view
        detailed_widget = QWidget()
        detailed_layout = QVBoxLayout()
        self.detailed_text = QTextEdit()
        self.detailed_text.setReadOnly(True)
        self.detailed_text.setFont(QFont("Arial", 14))  # Increased font size
        detailed_layout.addWidget(self.detailed_text)
        detailed_widget.setLayout(detailed_layout)
        self.result_tabs.addWidget(detailed_widget)

        # Raw JSON view
        json_widget = QWidget()
        json_layout = QVBoxLayout()
        self.json_text = QTextEdit()
        self.json_text.setReadOnly(True)
        self.json_text.setFont(QFont("Courier", 12))  # Monospaced font for JSON, slightly increased size
        json_layout.addWidget(self.json_text)
        json_widget.setLayout(json_layout)
        self.result_tabs.addWidget(json_widget)

        right_layout.addWidget(self.result_tabs)

        # Buttons to switch views
        view_buttons = QHBoxLayout()
        self.summary_btn = QPushButton("Summary")
        self.summary_btn.clicked.connect(lambda: self.change_view(0))
        self.detailed_btn = QPushButton("Detailed")
        self.detailed_btn.clicked.connect(lambda: self.change_view(1))
        self.json_btn = QPushButton("Raw JSON")
        self.json_btn.clicked.connect(lambda: self.change_view(2))
        view_buttons.addWidget(self.summary_btn)
        view_buttons.addWidget(self.detailed_btn)
        view_buttons.addWidget(self.json_btn)
        right_layout.addLayout(view_buttons)

        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel, 2)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.set_color_scheme()
        self.change_view(0)  # Set initial view to Summary

    def set_color_scheme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)

        # Set stylesheet for buttons
        button_style = """
        QPushButton {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            margin: 4px 2px;
            border-radius: 8px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        """
        self.submit_button.setStyleSheet(button_style)
        self.summary_btn.setStyleSheet(button_style)
        self.detailed_btn.setStyleSheet(button_style)
        self.json_btn.setStyleSheet(button_style)

    def submit_query(self):
        query = self.query_input.toPlainText()
        if query:
            self.query_history.addItem(query)
            self.submit_button.setEnabled(False)
            self.loader.setVisible(True)

            self.worker = QueryWorker(self.load_balancer, query)
            self.worker.finished.connect(self.display_result)
            self.worker.start()

    def display_result(self, result):
        if 'error' in result:
            self.summary_text.setPlainText(result['error'])
            self.detailed_text.setPlainText(result['enhanced_response']['detailed_answer'])
        else:
            self.summary_text.setPlainText(result['enhanced_response']['summary'])
            self.detailed_text.setPlainText(result['enhanced_response']['detailed_answer'])
        
        # Convert result to JSON-serializable format
        json_result = self.convert_to_json_serializable(result)
        self.json_text.setPlainText(json.dumps(json_result, indent=2))
        
        self.submit_button.setEnabled(True)
        self.loader.setVisible(False)
    
    def convert_to_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, AIMessage):
            return str(obj)  # Convert AIMessage to string
        elif hasattr(obj, '__dict__'):
            return self.convert_to_json_serializable(obj.__dict__)
        else:
            return obj

    def load_query(self, item):
        self.query_input.setPlainText(item.text())

    def change_view(self, index):
        self.result_tabs.setCurrentIndex(index)
        buttons = [self.summary_btn, self.detailed_btn, self.json_btn]
        for i, button in enumerate(buttons):
            if i == index:
                button.setStyleSheet(button.styleSheet() + "background-color: #45a049;")
            else:
                button.setStyleSheet(button.styleSheet().replace("background-color: #45a049;", ""))

    def closeEvent(self, event):
        try:
            if self.load_balancer:
                self.load_balancer.shutdown()
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = KnowledgeGraphUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()