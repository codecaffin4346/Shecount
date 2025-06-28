from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import random
import google.generativeai as genai
from google.generativeai.types.generation_types import StopCandidateException
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure AI services
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize sentence transformer
sentence_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Pinecone configuration
pinecone_api_key = os.getenv('PINECONE_API_KEY')
if pinecone_api_key:
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "fidhacks"
else:
    pc = None
    print("Warning: PINECONE_API_KEY not found. Resource lookup will be disabled.")

try:
    if pc:
        index = pc.Index(index_name)
    else:
        index = None
except Exception:
    print("Warning: Could not connect to Pinecone index. Resource lookup will be disabled.")
    index = None

safety_instruction = "Please ensure that the content you generate is safe, appropriate, and free from explicit or harmful language."

class ChatbotAPI:
    def __init__(self):
        self.chat = model.start_chat(history=[])
        self.current_question = ""
        self.correct_answer = ""

    def generate_question(self, topic="financial literacy"):
        question_instruction = f"Generate a specific multiple choice or short answer question about {topic}. Make it educational and practical. Only provide the question, nothing else."
        question = ""

        try:
            question_response = self.chat.send_message(safety_instruction + " " + question_instruction)
            question = question_response.text.strip()
            if question.startswith("Question:"):
                question = question.replace("Question:", "").strip()
        except StopCandidateException:
            question = "What is the recommended amount for an emergency fund?"
        except Exception:
            question = "What factors should you consider when choosing a credit card?"

        self.current_question = question
        return question

    def generate_answer(self):
        if not self.current_question:
            return "No question available."

        answer_instruction = f"Provide a clear, concise answer to this question: {self.current_question}. Give a direct answer without extra formatting."
        correct_answer = ""

        try:
            answer_response = self.chat.send_message(safety_instruction + " " + answer_instruction)
            correct_answer = answer_response.text.strip()
            if correct_answer.startswith("Answer:"):
                correct_answer = correct_answer.replace("Answer:", "").strip()
        except StopCandidateException:
            correct_answer = "Please refer to financial literacy resources for the correct answer."
        except Exception as e:
            print(f"Error generating answer: {e}")
            correct_answer = "Unable to generate answer at this time."

        self.correct_answer = correct_answer
        return correct_answer

    def evaluate_answer(self, user_answer):
        if not self.current_question:
            return {"is_correct": False, "message": "No question available."}

        if not self.correct_answer:
            self.generate_answer()

        evaluation_instruction = f"""
        Question: {self.current_question}
        User's Answer: {user_answer}
        Correct Answer: {self.correct_answer}

        Compare the user's answer to the correct answer. If they are similar in meaning or the user's answer contains the key correct information, respond with "CORRECT". Otherwise respond with "INCORRECT" followed by a brief explanation.
        """

        try:
            evaluation_response = self.chat.send_message(safety_instruction + " " + evaluation_instruction)
            evaluation_text = evaluation_response.text.strip()

            is_correct = "CORRECT" in evaluation_text.upper() and "INCORRECT" not in evaluation_text.upper()

            result = {
                "is_correct": is_correct,
                "message": evaluation_text,
                "correct_answer": self.correct_answer
            }

            if not is_correct and index:
                resource = self.get_relevant_resource(self.correct_answer)
                if resource:
                    result["resource"] = resource

            return result

        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {
                "is_correct": False,
                "message": f"Error evaluating answer: {str(e)}",
                "correct_answer": self.correct_answer or "Answer not available"
            }

    def get_relevant_resource(self, query_text):
        if not index:
            return None

        try:
            temp_emb = sentence_model.encode(query_text).tolist()
            query_results = index.query(
                namespace="auto_loan_resources",
                vector=temp_emb,
                top_k=1,
                include_metadata=True
            )

            if query_results.matches:
                top_match_metadata = query_results.matches[0].metadata
                return {
                    "title": top_match_metadata.get('title', 'Financial Resource'),
                    "link": top_match_metadata.get('link', '#'),
                    "description": top_match_metadata.get('description', '')
                }
        except Exception as e:
            print(f"Error getting resource: {e}")

        return None

    def handle_general_question(self, question):
        instruction = f"Answer this financial literacy question: {question}. Provide a helpful, educational response."

        try:
            response = self.chat.send_message(safety_instruction + " " + instruction)
            return response.text
        except Exception:
            return "I apologize, but I'm having trouble answering that question right now. Please try again."

    def generate_question_with_answer(self, topic="financial literacy"):
        combined_instruction = f"""
        Create a financial literacy question about {topic} and provide its answer.
        Format your response exactly like this:

        QUESTION: [Your question here]
        ANSWER: [The correct answer here]

        Make the question practical and educational.
        """

        try:
            response = self.chat.send_message(safety_instruction + " " + combined_instruction)
            response_text = response.text.strip()

            if "QUESTION:" in response_text and "ANSWER:" in response_text:
                parts = response_text.split("ANSWER:")
                question_part = parts[0].replace("QUESTION:", "").strip()
                answer_part = parts[1].strip()

                self.current_question = question_part
                self.correct_answer = answer_part

                return question_part
            else:
                self.current_question = "What factors should you consider when choosing a credit card?"
                self.correct_answer = "When choosing a credit card, consider: annual fees, interest rates (APR), rewards programs, credit limit, accepted locations, customer service, and any special benefits or perks."
                return self.current_question

        except Exception as e:
            print(f"Error generating question with answer: {e}")
            self.current_question = "What is an emergency fund and why is it important?"
            self.correct_answer = "An emergency fund is money set aside to cover unexpected expenses like medical bills, car repairs, or job loss. It's important because it provides financial security and prevents you from going into debt during emergencies."
            return self.current_question

# Initialize chatbot
chatbot = ChatbotAPI()

@app.route('/')
def index():
    return render_template('chatbot_frontend.html')

@app.route('/api/generate_question', methods=['POST', 'OPTIONS'])
def api_generate_question():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.get_json() or {}
        topic = data.get('topic', 'financial literacy')
        question = chatbot.generate_question_with_answer(topic)
        return jsonify({'success': True, 'question': question})
    except Exception as e:
        print(f"Error generating question: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/submit_answer', methods=['POST', 'OPTIONS'])
def api_submit_answer():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.get_json() or {}
        user_answer = data.get('answer', '')

        if not user_answer.strip():
            return jsonify({'success': False, 'error': 'Answer cannot be empty'}), 400

        evaluation = chatbot.evaluate_answer(user_answer)
        return jsonify({'success': True, 'evaluation': evaluation})
    except Exception as e:
        print(f"Error submitting answer: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ask_question', methods=['POST', 'OPTIONS'])
def api_ask_question():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.get_json() or {}
        question = data.get('question', '')
        response = chatbot.handle_general_question(question)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        print(f"Error handling question: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reset_chat', methods=['POST', 'OPTIONS'])
def api_reset_chat():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        global chatbot
        chatbot = ChatbotAPI()
        return jsonify({'success': True, 'message': 'Chat reset successfully'})
    except Exception as e:
        print(f"Error resetting chat: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
