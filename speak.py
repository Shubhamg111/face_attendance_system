# pip install gTTS
from gtts import gTTS
import os

# text = "Hello, how are you?"
# tts = gTTS(text=text, lang='en')
# tts.save("output.mp3")
# os.system("start output.mp3")  

# from win32com.client import Dispatch

# def speak():
#     text = "Hello, how are you?"
#     # tts = gTTS(text=text, lang='en')
#     speak = Dispatch(gTTS(text=text, lang='en'))
#     speak.Speak(text)





# pip install pyttsx3
# import pyttsx3

# engine = pyttsx3.init()
# engine.say("Hello, how are you?")
# engine.runAndWait()

# pip install pyttsx4
# import pyttsx4

# engine = pyttsx4.init()
# engine.say("Hello, how are you?")
# engine.runAndWait()


# pip install SpeechRecognition
# pip install pyaudio

# import speech_recognition as sr

# recognizer = sr.Recognizer()
# with sr.Microphone() as source:
#     print("Say something!")
#     audio = recognizer.listen(source)

# try:
#     text = recognizer.recognize_google(audio)
#     print("You said: " + text)
# except sr.UnknownValueError:
#     print("Google Speech Recognition could not understand audio")
# except sr.RequestError as e:
#     print("Could not request results from Google Speech Recognition service; {0}".format(e))


