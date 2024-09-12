import re
import speech_recognition as sr
import pyttsx3
import os

from better_profanity import profanity
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from od import ObjectDetector
import threading


obj_detector = ObjectDetector()
obj_detector_thread = threading.Thread(target=obj_detector.start_detecting)
obj_detector_thread.start()

# Initialize recognizer and text-to-speech engine
r = sr.Recognizer()
engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def speak(text):
    engine.say(text)
    engine.runAndWait()


# Define preprocessing function
def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip() 

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    processed_text = ' '.join(stemmed_tokens)
    return processed_text

# Function to check for profanity and sentiment in the text
def check_profanity_and_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    contains_profanity = profanity.contains_profanity(text)
    
    if contains_profanity:
        if sentiment < 0:
            speak("Profanity detected with a negative sentiment. Please watch your language.")
            print("Profanity detected. Negative sentiment detected.")
        else:
            speak("Profanity detected. Please watch your language.")
            print("Profanity detected. Positive or neutral sentiment detected.")
    else:
        if sentiment < 0:
            speak("Negative sentiment detected. I hope everything is okay.")
            print("Negative sentiment detected.")
        else:
            speak("Your message has been received.")
            print("No profanity detected. Positive or neutral sentiment detected.")
            
    # Provide additional sentiment feedback if no profanity is detected
    if not contains_profanity:
        if sentiment > 0:
            speak("I'm glad you're feeling positive!")
            print("Positive sentiment detected.")
        elif sentiment == 0:
            speak("I detect a neutral sentiment. Can I assist you with something?")
            print("Neutral sentiment detected.")

def listen():
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                print(f"User said: {text}")
                # preprocessed_text = preprocess_text(text)
                # check_profanity_and_sentiment(preprocessed_text)
                return text.lower() #return preprocessed_text.lower()
            except sr.UnknownValueError:
                print("Sorry, I did not get that.")
                speak("Sorry, I did not get that.")
                return None
            except sr.RequestError:
                print("Sorry, my speech recognition service is currently down.")
                speak("Sorry, my speech recognition service is currently down.")
                return None


def get_name():
    speak("My name is FindO.")
    speak("What is your name?")
    name = listen()
    if name:
        speak(f"Hello, {name}.")

   
def handle_find_object(obj_detector):
    while True:
        speak("What object are you looking for?")
        object_name = listen()

        if object_name is None:
            speak("I'm sorry, I didn't catch that. Could you please repeat?")
            continue

        object_name = object_name.lower()
        if object_name not in [name.lower() for name in obj_detector.classNames]:
            speak("I'm sorry, but I don't recognize that object. Going back to command mode.")
            return

        speak(f"Searching for {object_name}...")
        break       
     
    while True:
        instructions = obj_detector.get_near_instructions(object_name)
        if instructions == "Object not found.":
            speak(instructions)
            break
        else:
            speak(instructions)

def main():
    
    exit_flag = False
            
    while True:
        speak("Hello! What can I do for you today?")
        
        while True:
            command = listen()
            if command:
                if "hello" in command:
                    speak("Hi there!")
                elif "how are you" in command:
                    speak("I am good, thank you.")
                elif "what's your name" in command:
                    get_name()
                elif "locate" in command:
                    speak("Locate Function..")
                    handle_find_object(obj_detector)
                    
                elif "bye" in command:
                    speak("Goodbye!")
                    exit_flag = True
                    break
         
            if exit_flag:
                    break

if __name__ == "__main__":
    main()
