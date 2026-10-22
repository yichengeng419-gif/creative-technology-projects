import re
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

class DebatePartner:
    def __init__(self):
        self.conversation_is_active = True
        self.role = None
        self.emotion_history = []
        self.topic_history = []
        self.last_user_statements = []
        self.sid = SentimentIntensityAnalyzer()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        try:
            _ = self.sid.polarity_scores("test")
        except LookupError:
            import nltk
            print("VADER lexicon not found. Downloading now...")
            nltk.download('vader_lexicon', quiet=True)
            self.sid = SentimentIntensityAnalyzer()

        self.intent_examples = {
            "greeting": ["hi", "hello", "hey", "nice to meet"],
            "farewell": ["bye", "see you", "goodbye", "got to go"],
            "planning": ["future", "goal", "plan", "next step", "should i"],
            "help": ["what can you do", "help me", "use this"],
            "joke": ["joke", "laugh"],
            "time_enquiry": ["what time", "current time", "time now", "date"]
        }

        self.humor = {
            "planning_opt": [
                "Your plans show ambition! Future-you is giving high-fives.",
                "Planning mode activated. May your goals not crash like my updates.",
                "A strong plan! Even if reality disagrees, you still get points.",
                "Future-you is already cheering somewhere."
            ],
            "planning_ske": [
                "Plans are cute. Life usually says 'no'.",
                "Big plans, huh? Bold of you to challenge the universe.",
                "Future-you might not approve.",
                "Let's hope reality cooperates for once."
            ],
            "positive_opt": [
                "Love the energy—you're glowing like 100% battery.",
                "Great vibes detected. Keep going!",
                "You're basically sunshine today.",
                "That enthusiasm is contagious."
            ],
            "positive_ske": [
                "Positive… suspicious.",
                "Good for you. Probably.",
                "Let’s not celebrate too early.",
                "Hope you're right this time."
            ],
            "negative_opt": [
                "You'll grow through this. Even chaos grows flowers.",
                "It’s okay. Everyone glitches sometimes.",
                "Tough moment, but you're tougher.",
                "Storms pass eventually."
            ],
            "negative_ske": [
                "Yep. Sounds about right.",
                "Your worry is valid. Sadly.",
                "Chaos detected. Expected.",
                "Life strikes again, I see."
            ],
            "general_opt": [
                "Interesting—tell me more about",
                "Go on, I’m listening. So you mean",
                "That sounds deep. Continue with",
                "I see—explain more about"
            ],
            "general_ske": [
                "Hmm. Questionable. Why do you say",
                "Go on. I’m judging silently. You feel that",
                "Interesting. Probably wrong. But tell me more about",
                "I'm curious—explain why you think"
            ]
        }

        self.humor_embeddings = {
            key: self.encoder.encode(responses, convert_to_tensor=True)
            for key, responses in self.humor.items()
        }

    def choose_role(self):
        print("\nWelcome to the Inner Duel.")
        print("1. Optimistic\n2. Skeptical\nType exit to leave.\n")
        while True:
            c = input("Your choice: ").lower()
            if c in ["1", "optimistic"]:
                self.role = "optimistic"
                print("\nOptimistic: Life is full of possibilities!")
                break
            elif c in ["2", "skeptical"]:
                self.role = "skeptical"
                print("\nSkeptical: I doubt that. Let's talk.")
                break
            elif c == "exit":
                self.conversation_is_active = False
                break
            else:
                print("Invalid choice. Try again.")

    def get_intent(self, text):
        user_text = text.lower()
        for intent, keywords in self.intent_examples.items():
            for kw in keywords:
                if kw in user_text:
                    return intent
        return "general"

    def analyze_sentiment(self, text):
        c = self.sid.polarity_scores(text)["compound"]
        if c >= 0.3:
            label = "positive"
        elif c <= -0.3:
            label = "negative"
        else:
            label = "neutral"
        self.emotion_history.append(label)
        return label

    def update_topics(self, text):
        t = text.lower()
        if any(k in t for k in ["work", "job", "career"]):
            self.topic_history.append("work")
        if any(k in t for k in ["study", "exam", "university", "school"]):
            self.topic_history.append("study")
        if any(k in t for k in ["relationship", "friend", "partner", "family"]):
            self.topic_history.append("relationship")
        if any(k in t for k in ["future", "plan", "goal"]):
            self.topic_history.append("future")

    def reflect_text(self, text):
        reflections = {
            "i": "you", "my": "your", "me": "you", "am": "are", "was": "were",
            "i'm": "you're", "i've": "you've", "mine": "yours"
        }
        words = re.findall(r'\b\w+\b', text.lower())
        out = [reflections.get(w, w) for w in words]
        if out:
            out[0] = out[0].capitalize()
        return " ".join(out)

    def select_non_repetitive_response(self, user_input, key):
        user_emb = self.encoder.encode(user_input, convert_to_tensor=True)
        emb = self.humor_embeddings[key]
        scores = util.cos_sim(user_emb, emb).squeeze(0)
        best = scores.argmin().item()
        return self.humor[key][best]
    
    def generate_summary(self):
        recent_topics = self.topic_history[-5:] if self.topic_history else []
        recent_emotions = self.emotion_history[-5:] if self.emotion_history else []
        recent_texts = self.last_user_statements[-5:] if self.last_user_statements else []
        return (
            "Summary:\n"
            f"- Recent topics: {recent_topics}\n"
            f"- Recent emotions: {recent_emotions}\n"
            f"- Recent thoughts: {recent_texts}"
        )

    def analyze_trend(self):
        last = self.emotion_history[-10:]
        pos = last.count("positive")
        neg = last.count("negative")
        neu = last.count("neutral")
        if pos > neg:
            trend = "More positive lately."
        elif neg > pos:
            trend = "More negative lately."
        else:
            trend = "Balanced emotional pattern."
        return f"Trend:\nPositive: {pos}  Negative: {neg}  Neutral: {neu}\n{trend}"

    def show_memory(self):
        return (
            "Memory:\n"
            f"- Emotions: {self.emotion_history}\n"
            f"- Topics: {self.topic_history}\n"
            f"- Said: {self.last_user_statements}"
        )

    def respond(self, user_input):
        
        cmd = user_input.strip().lower()
        if cmd == "memory":
            return self.show_memory()
        if cmd == "summary":
            return self.generate_summary()
        if cmd == "trend":
            return self.analyze_trend()

        # 记录用户输入（在特殊命令处理后）
        self.last_user_statements.append(user_input)

        intent_raw = self.get_intent(user_input)
        emotion = self.analyze_sentiment(user_input)
        self.update_topics(user_input)

        if intent_raw == "farewell":
            self.conversation_is_active = False
            return "Goodbye. Try not to overthink too much."
        if intent_raw == "greeting":
            return "Hello! Ready for some introspective chaos?"
        if intent_raw == "time_enquiry":
            return f"It's {datetime.now().strftime('%H:%M:%S')}."
        if intent_raw == "joke":
            return random.choice([
                "I told my computer I needed a break—and it froze.",
                "Parallel lines have so much in common but never meet.",
                "I used to be a lampshade. It was a light job."
            ])
        
        if intent_raw == "help":
            return "You can talk about feelings, plans, or worries. You can also type 'summary', 'trend', or 'memory' to view our conversation data. Use 'exit' to quit."

        if intent_raw == "planning":
            key = "planning_opt" if self.role == "optimistic" else "planning_ske"
            return self.select_non_repetitive_response(user_input, key)

        if emotion == "positive":
            key = "positive_opt" if self.role == "optimistic" else "positive_ske"
            return self.select_non_repetitive_response(user_input, key)

        if emotion == "negative":
            key = "negative_opt" if self.role == "optimistic" else "negative_ske"
            return self.select_non_repetitive_response(user_input, key)

        key = "general_opt" if self.role == "optimistic" else "general_ske"
        starter = self.select_non_repetitive_response(user_input, key)
        phrased = self.reflect_text(user_input)
        return f"{starter} '{phrased}'?"

    def chat(self):
        self.choose_role()
        while self.conversation_is_active and self.role:
            u = input("\nYou: ")
            if u.lower() in ["exit", "quit"]:
                print("Goodbye. Try not to overthink too much.")
                break
            print(f"{self.role.capitalize()}: {self.respond(u)}")

if __name__ == "__main__":
    DebatePartner().chat()