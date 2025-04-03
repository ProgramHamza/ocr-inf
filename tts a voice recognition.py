import speech_recognition as sr
import pyttsx3

import datetime

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def talk(text):
    engine.say(text)
    engine.runAndWait()


def take_command():
    try:
        with sr.Microphone() as source:
            print('listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'alexa' in command:
                command = command.replace('alexa', '')
                print(command)
    except:
        pass
    return command

a = False
def run_alexa():
    command = take_command()
    print(command)
    if 'time' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time)
    elif 'thanks' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('You are welcome')
    elif 'off' in command:
            talk('I am sorry, I will not disturb you again')
            a = True
    else:
        talk('Please say the command again.')


while True:
    run_alexa()
    if a == True:
        break