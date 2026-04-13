"""
Spam Message Detector
Uses TF-IDF + Naive Bayes classifier trained on SMS spam dataset.
"""

import re
import pickle
import os
from collections import defaultdict
import math



TRAINING_DATA = [
    
    ("Free entry in 2 a weekly competition to win FA Cup final tkts! Text FA to 87121", "spam"),
    ("WINNER!! You have been selected to receive a $1000 gift card. Call now!", "spam"),
    ("Congratulations! You've won a free iPhone. Click here to claim now!", "spam"),
    ("URGENT: Your account has been compromised. Verify now at http://scam.com", "spam"),
    ("You have been pre-approved for a $5000 loan. No credit check needed!", "spam"),
    ("FREE ringtones! Text RING to 82468. 3 msgs/week. £1.50/msg.", "spam"),
    ("Win a brand new car! Reply YES to enter. Limited time offer!", "spam"),
    ("Your mobile number has won £500,000 in our lottery. Call to claim!", "spam"),
    ("SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 to 87575", "spam"),
    ("PRIVATE! Your 2003 Account Statement shows 800 un-redeemed points. Call now!", "spam"),
    ("Had your mobile 11 months or more? You are entitled to update to the latest colour mobiles", "spam"),
    ("IMPORTANT. Please call us free on 0800 169 6031 immediately.", "spam"),
    ("Did you hear about the new weight loss pill? Lose 30 lbs in 30 days!", "spam"),
    ("Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry", "spam"),
    ("Free msg: Txt 2 claim ur prize of £100 to be paid direct 2 ur bank account!", "spam"),
    ("Click here to get your FREE gift now! Limited offer expires tonight!", "spam"),
    ("You have been chosen to receive a £200 Tesco voucher. Text INFO to 86888", "spam"),
    ("Make money fast from home! No experience needed. Earn £500/day!", "spam"),
    ("Lose weight fast with our miracle pill! 100% guaranteed results!", "spam"),
    ("CLAIM YOUR PRIZE NOW! You are the winner of our monthly draw!", "spam"),
    ("Get cheap meds online! No prescription needed. Best prices guaranteed!", "spam"),
    ("Your account will be suspended. Verify your details immediately.", "spam"),
    ("Investment opportunity! Guaranteed 200% returns in just 7 days!", "spam"),
    ("Hi babe I want to meet you. Text MEET to 69911 to find local singles.", "spam"),
    ("SMS: We are trying to contact you. Last chance to claim your £150 award", "spam"),

    
    ("Hey, are you coming to the party tonight?", "ham"),
    ("I'll be there in 10 minutes, just finishing up at work.", "ham"),
    ("Can you pick up some milk on your way home?", "ham"),
    ("Meeting has been moved to 3pm tomorrow. Let me know if that works.", "ham"),
    ("Happy birthday! Hope you have a great day!", "ham"),
    ("Did you watch the game last night? What a match!", "ham"),
    ("I'm running a bit late, traffic is terrible today.", "ham"),
    ("Thanks for dinner last night, it was delicious!", "ham"),
    ("Can we reschedule our call to Thursday? I have a conflict.", "ham"),
    ("Just wanted to check in and see how you're doing.", "ham"),
    ("The report is ready. I've sent it to your email.", "ham"),
    ("Are you free this weekend? We should catch up!", "ham"),
    ("I left my keys at your place. Can I come by and get them?", "ham"),
    ("Great work on the presentation today! Everyone loved it.", "ham"),
    ("Mom says dinner is at 7. Don't be late!", "ham"),
    ("I need to talk to you about something important. Call me when you can.", "ham"),
    ("The project deadline has been extended by a week.", "ham"),
    ("Can you send me the address for the venue?", "ham"),
    ("I'm feeling much better now, thanks for asking.", "ham"),
    ("Flight is delayed by 2 hours. Will keep you updated.", "ham"),
    ("Your package has been delivered to the front door.", "ham"),
    ("Don't forget we have a team lunch tomorrow at noon.", "ham"),
    ("I'll call you when I land. Should be around 6pm.", "ham"),
    ("The kids had a great time at the park today.", "ham"),
    ("Sorry for the late reply, been super busy at work.", "ham"),
    ("Can you review the document I sent? No rush.", "ham"),
    ("We're out of coffee. Can you grab some?", "ham"),
    ("Let me know when you're ready and I'll come pick you up.", "ham"),
    ("The doctor says everything looks fine. Big relief!", "ham"),
    ("Dinner tonight was amazing. We should go back!", "ham"),
]




class SpamDetector:
    def __init__(self):
        self.spam_word_counts = defaultdict(int)
        self.ham_word_counts = defaultdict(int)
        self.spam_total = 0
        self.ham_total = 0
        self.spam_messages = 0
        self.ham_messages = 0
        self.vocabulary = set()

    def preprocess(self, text: str) -> list:
        """Clean and tokenize text."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', 'URL', text)
        text = re.sub(r'\b\d+\b', 'NUMBER', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Remove very short words
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    def train(self, data: list):
        """Train on list of (message, label) tuples."""
        for message, label in data:
            tokens = self.preprocess(message)
            if label == "spam":
                self.spam_messages += 1
                for token in tokens:
                    self.spam_word_counts[token] += 1
                    self.spam_total += 1
            else:
                self.ham_messages += 1
                for token in tokens:
                    self.ham_word_counts[token] += 1
                    self.ham_total += 1
            self.vocabulary.update(tokens)

        print(f"Trained on {self.spam_messages} spam + {self.ham_messages} ham messages")
        print(f"Vocabulary size: {len(self.vocabulary)} words\n")

    def predict(self, message: str) -> dict:
        """Classify a message as spam or ham with confidence score."""
        tokens = self.preprocess(message)
        vocab_size = len(self.vocabulary)

        total_messages = self.spam_messages + self.ham_messages
        log_prob_spam = math.log(self.spam_messages / total_messages)
        log_prob_ham = math.log(self.ham_messages / total_messages)

        for token in tokens:
            # Laplace smoothing
            spam_likelihood = (self.spam_word_counts[token] + 1) / (self.spam_total + vocab_size)
            ham_likelihood = (self.ham_word_counts[token] + 1) / (self.ham_total + vocab_size)
            log_prob_spam += math.log(spam_likelihood)
            log_prob_ham += math.log(ham_likelihood)

        # Convert to probabilities
        max_log = max(log_prob_spam, log_prob_ham)
        prob_spam = math.exp(log_prob_spam - max_log)
        prob_ham = math.exp(log_prob_ham - max_log)
        total = prob_spam + prob_ham

        spam_confidence = prob_spam / total
        ham_confidence = prob_ham / total

        label = "SPAM" if spam_confidence > 0.5 else "HAM"

        # Find top suspicious words
        suspicious_words = []
        for token in tokens:
            spam_count = self.spam_word_counts.get(token, 0)
            ham_count = self.ham_word_counts.get(token, 0)
            if spam_count > ham_count:
                suspicious_words.append(token)

        return {
            "label": label,
            "confidence": round(max(spam_confidence, ham_confidence) * 100, 1),
            "spam_probability": round(spam_confidence * 100, 1),
            "ham_probability": round(ham_confidence * 100, 1),
            "suspicious_words": suspicious_words[:5],
        }

    def get_stats(self) -> dict:
        """Return model stats."""
        return {
            "spam_messages_trained": self.spam_messages,
            "ham_messages_trained": self.ham_messages,
            "vocabulary_size": len(self.vocabulary),
            "top_spam_words": sorted(
                self.spam_word_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "top_ham_words": sorted(
                self.ham_word_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }


# ── Main Program ───────────────────────────────────────────────────────────

def print_result(message: str, result: dict):
    label = result["label"]
    confidence = result["confidence"]
    icon = "🚨" if label == "SPAM" else "✅"

    print(f"\n{'─'*55}")
    print(f"Message   : {message[:60]}{'...' if len(message) > 60 else ''}")
    print(f"Result    : {icon}  {label}")
    print(f"Confidence: {confidence}%")
    print(f"Spam prob : {result['spam_probability']}%")
    print(f"Ham prob  : {result['ham_probability']}%")
    if result["suspicious_words"] and label == "SPAM":
        print(f"Red flags : {', '.join(result['suspicious_words'])}")
    print(f"{'─'*55}")


def main():
    print("=" * 55)
    print("   SPAM MESSAGE DETECTOR")
    print("   Naive Bayes Classifier")
    print("=" * 55)
    print()

    # Train the model
    print("Training model...")
    detector = SpamDetector()
    detector.train(TRAINING_DATA)

    # Test on sample messages
    test_messages = [
        "Congratulations! You've won a $1000 prize. Claim now!",
        "Hey, are you free for lunch tomorrow?",
        "URGENT: Your bank account has been locked. Verify now!",
        "Can you send me the meeting notes from today?",
        "FREE iPhone! Click this link to claim your reward!",
        "I'll pick you up at 7, see you then.",
        "You have been pre-selected for a £5000 loan!",
        "Don't forget mom's birthday is next week.",
        "Win cash prizes by texting WIN to 80488 now!",
        "The package you ordered has been shipped.",
    ]

    print("Testing on sample messages:")
    print("=" * 55)

    for msg in test_messages:
        result = detector.predict(msg)
        print_result(msg, result)

    # Interactive mode
    print("\n" + "=" * 55)
    print("INTERACTIVE MODE — Type your own messages")
    print("Type 'quit' to exit | 'stats' for model info")
    print("=" * 55)

    while True:
        print()
        user_input = input("Enter message: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "stats":
            stats = detector.get_stats()
            print(f"\nModel Statistics:")
            print(f"  Spam messages trained: {stats['spam_messages_trained']}")
            print(f"  Ham messages trained : {stats['ham_messages_trained']}")
            print(f"  Vocabulary size      : {stats['vocabulary_size']}")
            print(f"  Top spam words: {[w for w, _ in stats['top_spam_words'][:5]]}")
            continue

        result = detector.predict(user_input)
        print_result(user_input, result)


if __name__ == "__main__":
    main()