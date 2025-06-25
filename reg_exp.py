from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import string
import nltk

# Downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Text Preprocessing Class
class TextPreprocessing:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text):
        return word_tokenize(text)

    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token.lower() not in self.stopwords]

    def preprocess(self, text):
        text = self.remove_punctuation(text)
        text = text.lower()
        tokens = self.tokenize(text)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)

# AI Chatbot Class
class AIPoweredChatbot:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.pipeline = make_pipeline(self.vectorizer, self.model)
        self.train_data()

    def train_data(self):
        training_sentences = [
            "hello", "hi", "hey", "goodbye", "bye", "how are you", "I am fine", "thanks", "thank you",
            "what is your name", "who created you", "tell me a joke", "I love you"
        ]
        responses = [
            "Hi there!", "Hi there!", "Hello!", "Goodbye!", "Bye!", "I'm doing great, thanks!", "That's good to hear!",
            "You're welcome!", "You're welcome!",
            "I'm a chatbot!", "I was created by a developer!",
            "Why don't scientists trust atoms? Because they make up everything!", "I love you too!"
        ]
        self.pipeline.fit(training_sentences, responses)

    def chatbot_response(self, user_input):
        return self.pipeline.predict([user_input])[0]

# Rule-Based Patterns
pairs = [
    [
        r"(hi|hello|hey)",
        ["Hello, how can I help you?", "Hey there! What's up?", "Hi! How's it going?"]
    ],
[
    r"tell me a joke",
    ["Why don't scientists trust atoms? Because they make up everything!",
     "Why did the computer go to the doctor? It had a virus!"]
],

    [
        r"what is your name\??",
        ["I am a simple chatbot created using Python and NLTK."]
    ],
        ## Bot identity
    [r"(what is|what's) your name\??", ["My name is Chat Bot. What's yours?"]],
    [r"(who|what) are you\??", ["I'm your friendly chatbot built using Python."]],
    
    ## What the bot does
    [r"what do you do\??", ["I help answer your questions, share jokes, and chat with you!"]],
    [r"how can you help me\??", ["I can assist you with questions, information, or just chat!"]],

    ## Jokes
    [r"tell me a joke", [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why did the computer go to the doctor? Because it had a virus!",
        "Why was the math book sad? It had too many problems."
    ]],

    ## Help or assistance
    [r"(can you help me|i need help)", ["Sure, tell me what you need help with."]],
    [r"(.*)help(.*)", ["I'm here to help. What exactly do you need?"]],

    ## Gratitude
    [r"(thank you|thanks)", ["You're welcome!", "Anytime!", "Glad to help."]],
    
    ## Farewell
    [r"(bye|goodbye|see you)", ["Goodbye! Have a great day!", "See you later!"]],
    [
        r"how are you\??",
        ["I'm just a bot, but I'm doing great! How about you?", "I'm good! Thanks for asking."]
    ],
    [
        r"(.*) your name\??",
        ["My name is Chat Bot. What's yours?", "I am Chat Bot, your virtual assistant."]
    ],
    [
        r"(.*) help (.*)",
        ["I can help you with general queries. What do you need help with?", "Sure! Let me know your question."]
    ],
    [
        r"(bye|goodbye)",
        ["Goodbye! Have a great day!", "See you later! Take care."]
    ],
    [
        r"(.*)",
        ["I'm not sure how to respond to that. Could you rephrase?", "That's interesting! Tell me more."]
    ]
]

# Initialize bots
preprocessor = TextPreprocessing()
ai_bot = AIPoweredChatbot()
rule_bot = Chat(pairs, reflections)

# Start chat loop
# print("Chatbot: Hello, how can I help you? (Type 'exit' to quit)\n")
# while True:
#     user_input = input("User: ")
#     if user_input.lower() == 'exit':
#         print("Chatbot: Goodbye! Have a great day!")
#         break
