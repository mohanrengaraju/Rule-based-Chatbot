from flask import Flask, render_template, request, jsonify
from reg_exp import TextPreprocessing, AIPoweredChatbot, Chat, pairs, reflections

app = Flask(__name__)
preprocessor = TextPreprocessing()
ai_bot = AIPoweredChatbot()
rule_bot = Chat(pairs, reflections)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    cleaned = preprocessor.preprocess(user_input)

    response = rule_bot.respond(cleaned)
    if not response:
        response = ai_bot.chatbot_response(cleaned)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
