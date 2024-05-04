from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np

# Expanded dataset for training
training_data = {
    "Greet": ["Hi", "How are you?", "Hello", "Hey there!", "Greetings", "Good to see you",
              "Hi, how's it going?", "Hey, what's up?", "Good morning!", "Good afternoon!",
              "What's happening?", "Howdy!", "Nice to meet you", "Hey, how have you been?",
              "Hi there!", "What's up?", "Yo!", "Hiya!", "Hi, nice to see you!", "Hey, good to see you!",
              "Howdy-do!", "Well, hello!", "Hi, howdy, hey!", "Hiya, how's it going?", "Hi, how are things?",
              "Hey, how's your day?", "Hey, what's going on?", "Hey, how are you doing?", "Hello, how have you been?",
              "Hi, what's new?", "Hey, long time no see!", "Hello, how's everything?", "Hi, how's life?"],

    "Farewell": ["Goodbye", "See you later", "Take care", "Until next time", "Bye for now", "Have a good day",
                 "Farewell, my friend", "Take it easy", "Catch you later", "Have a great day ahead",
                 "So long!", "See you soon", "Bye-bye", "Adios!", "Talk to you later", "Until we meet again",
                 "Take care, bye!", "Bye, have a great day!", "Goodbye, see you tomorrow!", "Bye for now, take care!",
                 "Catch you later, bye!", "Until next time, take care!", "Farewell, until we meet again!",
                 "Goodbye, it was nice talking to you!", "Take care, talk to you later!", "See you soon, bye!",
                 "Bye-bye, have a good one!", "Adios, take care!", "Until next time, farewell!",
                 "So long, have a great day!", "Goodbye, see you next time!", "Take care, goodbye!"],

    "Inquiry": ["What's the weather like today?", "Can you tell me the time?", "Where is the nearest restaurant?",
                "How's the weather?", "What time is it?", "Is there a good place to eat nearby?",
                "Any recommendations for lunch?", "What's the temperature outside?", "What's the forecast?",
                "Do you know any good restaurants around here?", "Where can I find a good coffee shop?",
                "How far is the nearest grocery store?", "What's the traffic like right now?",
                "Any movie recommendations?",
                "What's the news today?", "Can you suggest a good book to read?",
                "Do you have any advice?", "What's your opinion on this?", "What do you think about that?",
                "Could you help me with something?", "I need your assistance with this.",
                "What's your favorite movie?", "What's your favorite food?", "Tell me about yourself.",
                "What's the meaning of life?", "How do I solve this problem?", "Can you teach me something?",
                "What's the best way to learn?", "What's the purpose of existence?", "What's your favorite color?",
                "What's the secret to happiness?", "What's the key to success?", "What's the origin of the universe?",
                "What's your favorite hobby?", "What's your favorite music genre?", "What's your dream vacation?",
                "What's the most interesting thing you've learned?", "What's the craziest thing you've ever heard?",
                "What's your favorite quote?", "What's your idea of a perfect day?", "What's the best way to relax?",
                "What's the most important thing in life?", "What's your favorite animal?",
                "What's your favorite book?",
                "What's your favorite TV show?", "What's the best advice you've ever received?",
                "What's the weirdest dream you've ever had?", "What's the meaning of love?",
                "What's the greatest invention of all time?", "What's the most beautiful place you've ever seen?",
                "What's your favorite season?", "What's your favorite holiday?", "What's your favorite memory?",
                "What's the best thing that's ever happened to you?", "What's your biggest fear?",
                "What's your favorite thing to do on weekends?", "What's your favorite childhood memory?",
                "What's the best thing about being you?", "What's your favorite thing about yourself?",
                "What's your favorite thing about life?", "What's your favorite thing to do in your free time?",
                "What's the best way to start the day?", "What's the best way to end the day?",
                "What's the best thing that's ever happened to you?", "What's the best thing you've ever done?",
                "What's your favorite thing about nature?", "What's your favorite thing about humanity?",
                "What's the most beautiful thing you've ever seen?", "What's the most amazing thing you've ever seen?",
                "What's the best thing about living in the present?", "What's the best thing about being alive?",
                "What's the most important lesson you've learned?", "What's the most valuable thing you own?",
                "What's the most important thing in your life?", "What's the best thing about being you?",
                "What's the best thing about being alive?", "What's the most important thing in the world?",
                "What's the most important thing you've ever done?", "What's the most important thing in your life?",
                "What's the most important thing you've learned?", "What's the most important thing you can do?",
                "What's the most important thing in life?", "What's the most important thing in the universe?",
                "What's the most important thing in your opinion?", "What's the most important thing to you?",
                "What's the most important thing in your opinion?", "What's the most important thing in the world?",
                "What's the most important thing you've ever done?", "What's the most important thing you can do?",
                "What's the most important thing in life?", "What's the most important thing in the universe?",
                "What's the most important thing in your opinion?", "What's the most important thing to you?",
                "What's the most important thing in your opinion?"],

    "Confirmation": ["Yes", "No", "Maybe", "I'm not sure", "Absolutely", "Definitely", "Of course",
                     "Certainly", "I guess so", "Not really", "I don't think so", "Nope", "Sure thing",
                     "Absolutely not", "You bet", "Absolutely yes", "Yes, definitely", "Yes, of course",
                     "Yes, certainly", "Yes, absolutely", "No, definitely not", "No, absolutely not",
                     "Maybe, I'm not sure", "Maybe, it's possible", "I'm not sure, let me check",
                     "I'm not sure, can you repeat that?", "I guess so, but I'm not certain",
                     "I'm not sure, could you clarify?", "Yes, without a doubt", "No, without a doubt",
                     "Yes, for sure", "No, for sure", "Maybe, let me think about it", "Maybe, it's a possibility",
                     "I'm not sure, I'll get back to you on that", "Yes, that sounds right",
                     "No, that doesn't seem right",
                     "Yes, I agree", "No, I disagree", "I'm not sure, what do you think?",
                     "Yes, I think so", "No, I don't think so", "Yes, it's possible", "No, it's unlikely",
                     "Yes, it's probable", "No, it's improbable", "Yes, it's likely", "No, it's unlikely"],

    "Apology": ["Sorry", "My apologies", "I apologize", "Excuse me", "Forgive me", "I'm sorry about that",
                "Pardon me", "I beg your pardon", "I regret that", "I owe you an apology", "Please forgive me",
                "I'm deeply sorry", "It's my fault", "I'm really sorry", "I apologize for the inconvenience",
                "I'm so sorry", "Sorry about that", "Sorry, I didn't mean to", "Sorry, I didn't catch that",
                "Sorry, could you repeat that?", "Sorry, what was that?", "Sorry, I didn't understand",
                "Sorry, my mistake", "I'm sorry for any confusion", "I'm sorry for any inconvenience",
                "Sorry, that wasn't clear", "I'm sorry for any misunderstanding", "I'm sorry if I offended you",
                "I'm sorry, I'll try to do better", "Sorry, let me correct that", "I'm sorry for the trouble",
                "Sorry, that's not what I meant", "I'm sorry, I misspoke", "I'm sorry, I wasn't thinking",
                "I'm sorry, that was thoughtless", "I'm sorry, I'll be more careful",
                "I'm sorry, I'll make it up to you",
                "I'm sorry, I didn't realize", "I'm sorry, I didn't mean to upset you",
                "I'm sorry, I'll be more mindful",
                "I'm sorry, I'll try to be more considerate", "I'm sorry, I'll try to be more understanding",
                "I'm sorry, I'll try to be more empathetic", "I'm sorry, I'll try to be more compassionate",
                "I'm sorry, I'll try to be more patient", "I'm sorry, I'll try to be more tolerant",
                "I'm sorry, I'll try to be more respectful", "I'm sorry, I'll try to be more polite",
                "I'm sorry, I'll try to be more courteous", "I'm sorry, I'll try to be more gracious",
                "I'm sorry, I'll try to be more humble", "I'm sorry, I'll try to be more modest",
                "I'm sorry, I'll try to be more modest", "I'm sorry, I'll try to be more modest",
                "I'm sorry, I'll try to be more modest", "I'm sorry, I'll try to be more modest",
                "I'm sorry, I'll try to be more modest", "I'm sorry, I'll try to be more modest"],

    "Thanks": ["Thank you", "Thanks a lot", "Appreciate it", "Thanks so much", "Thank you very much",
               "Thanks a bunch", "You're awesome", "I'm grateful", "Much obliged", "Thanks a million",
               "You're the best", "Thank you kindly", "I owe you one", "I really appreciate it", "Thanks heaps",
               "Many thanks", "Thanks for everything", "Thanks for your help", "Thanks for your time",
               "Thanks for being there", "Thanks for listening", "Thanks for understanding",
               "Thanks for your support", "Thanks for the advice", "Thanks for the information",
               "Thanks for the update", "Thanks for the reminder", "Thanks for the clarification",
               "Thanks for the heads-up", "Thanks for the tip", "Thanks for the feedback",
               "Thanks for your input", "Thanks for the suggestion", "Thanks for your cooperation",
               "Thanks for your understanding", "Thanks for your patience", "Thanks for waiting",
               "Thanks for being patient", "Thanks for being understanding", "Thanks for being supportive",
               "Thanks for being there for me", "Thanks for everything you do", "Thanks for always being there",
               "Thanks for your kindness", "Thanks for your generosity", "Thanks for your hospitality",
               "Thanks for your thoughtfulness", "Thanks for your consideration", "Thanks for your compassion",
               "Thanks for your empathy", "Thanks for your support and encouragement",
               "Thanks for being so helpful", "Thanks for being so understanding", "Thanks for being so supportive",
               "Thanks for being so kind", "Thanks for being so generous", "Thanks for being so thoughtful",
               "Thanks for being so considerate", "Thanks for being so compassionate", "Thanks for being so empathetic",
               "Thanks for being so supportive and encouraging"]
}

# Expand the dataset by tripling each example
expanded_data = {intent: examples * 3 for intent, examples in training_data.items()}

# Prepare training data
X_train = []
y_train = []
for intent, examples in expanded_data.items():
    X_train.extend(examples)
    y_train.extend([intent] * len(examples))

# Define and train the model
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(probability=True))
])
model.fit(X_train, y_train)

# Intent classification function
def classify_intent(text, threshold=0.7):
    confidence_scores = model.predict_proba([text])[0]
    predicted_intent = model.predict([text])[0]
    max_confidence = np.max(confidence_scores)
    
    if max_confidence >= threshold:
        return predicted_intent, max_confidence
    else:
        return "NLU fallback: Intent could not be confidently determined",

# Test the classification function with user inputs
while True:
    user_input = input("Enter your message (or type 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    
    result = classify_intent(user_input)
    if len(result) == 1:
        print(result[0])  # Fallback response
    else:
        intent, confidence = result
        print(f"Predicted Intent: {intent}, Confidence: {confidence}")
