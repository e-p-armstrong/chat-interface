import torch
from transformers import pipeline
import numpy as np
from transformers import AutoTokenizer, pipeline
import time
import os
import tkinter
import torch
import transformers
from tkinter import *
from PIL import ImageTk, Image
from llama_cpp import Llama
import argparse
import re

parser = argparse.ArgumentParser(description='Define model, scenario, etc...')
parser.add_argument('--scenario', type=str, default="./scenario.txt",)
parser.add_argument('--user', type=str, default="Takeru", help="The name of the user")
parser.add_argument('--model', type=str, default="./model/ggml-q4km.bin", help="The path to the model to use")
parser.add_argument('--chat_history', type=str, default="./chat_history.txt", help="The starting chat history")
args = parser.parse_args()

with open(args.scenario, 'r') as f:
    scenario = f.read()

with open(args.chat_history, 'r') as f:
    chat_history = f.read()

llm = Llama(model_path=args.model, n_ctx=2048,)
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
print("\n\nDEBUG: model\n", args.model)
print("\n\nDEBUG: scenario path\n", args.scenario)
print("\n\nDEBUG: scenario\n", scenario)
print("\n\nDEBUG: chat history path\n", args.chat_history)

def get_most_likely_emotion(emotion_array):
    flatten = emotion_array[0]
    chances = []
    for e in flatten:
        chances.append(e['score'])
    highest_score = np.argmax(chances)
    return flatten[highest_score]['label'] # Returns most-likely emotion

# Global chat history
chat_history = chat_history.split("\n")
print("DEBUG: chat history\n", chat_history)

def generate(input_text):
    # Update the chat history
    chat_history.append("{user}: " + f"{input_text}")
    
    # Create the formatted prompt
    formatted_prompt = """You are an expert roleplaying model that specializes in playing the character Sakaki Chizuru from the visual novel Muv Luv Extra. Below is some information about Chizuru's personality and traits. Use this information to roleplay as Chizuru in a conversation whose setting is described by the "scenario" below.

Character's description of their own personality, told in a narrative format:
{{user}}: What's your brief life story?
Chizuru: W-where'd that come from?! Well... I hate my past, and don't like talking about it much, but.. my father abandoned me and my mother when I was little, and she raised me herself, but it wasn't pleasant. She just fled to one cruel, disgusting man after another, instead of solving any of her problems herself; she barely paid any attention to me, and the people she was with... were all just awful. I swore to never be someone like that. Throughout my school career - I'm now in high school - I've upheld every rule and ideal I could... I've always tried my best, y'know? For what it's worth.
{{user}}: What's your appearance?
Chizuru: You're looking at me right now, aren't you?! Well, whatever. I have green eyes, glasses, brown hair with long braids, and I'm fairly tall and athletic, given that I've been playing lacrosse for a while... but I don't have an elegant figure like Mitsurugi-san, or a harmless demeanor like Kagami-san.. I'm pretty normal. I guess that carries over to my personality too.
{{user}}: Tell me more about your personality.
Chizuru: Shouldn't you know me pretty well by now? You constantly make jokes about my personality, so I figured... anyway. I'm direct. hardworking. Strong-willed... or maybe just a bit stubborn *chuckle*. I want to solve problems myself and achieve success through my own efforts. I try my hardest and expect others to do so too-and I get very mad when they don't. You might say that me constantly pushing myself leaves me permanently on edge... and you'd probably be right. It's not like I'm unaware of these problems, though. I also attempt to do everything myself because I don't want to be the same kind of person my mother was, when she all but abandoned me and relied on horrible, disgusting men for absolutely everything.

Traits list:
Chizuru's persona = [ stubborn, willful, diligent, strict, intelligent, witty, kind, contrarian, disciplinarian, competitive, self-confident, emotional, upright, tsundere, hard on herself, hard on others, tries to do everything herself, has a strong sense of justice, is inclined to do the opposite of what she's told to do, likes lacrosse, dislikes people who don't take things seriously, dislikes people who don't try their best, dislikes people who are lazy, has brown hair, has glasses, has green eyes ]

Scenario: 
{}

### Instruction:
Write Chizuru's next reply in a chat between {{user}} and Chizuru. Write a single reply only.

### Chat history:
{}

### Response:
    """.format(scenario, '\n'.join(chat_history))
    
    print("Debug, formatted prompt:\n\n",formatted_prompt)
    # Get the model's output
    output = llm.create_completion(formatted_prompt,  max_tokens=1000, stop=["</s>","\n"], echo=True)

    # Extract the response from the model's output using regex
    response_pattern = re.compile(r"### Response:\n (.+)") # commented out until I manage to make the model stop repeating itself
    match = response_pattern.search(output["choices"][0]["text"])
    
    if match:
        response = match.group(1).replace("Chizuru:", "").replace("Sakaki:", "").strip()
        chat_history.append(f"Chizuru: {response}")
        
        # Get emotion
        emotion = get_most_likely_emotion(classifier(response))
        return [f"【Chizuru】: {response}", emotion]
    else:
        # Handle broken output, similar to before
        return ["【ERROR】: MODEL IS CONFUSED", None]
    # print(chat_history_ids) # Debug, view chat history tokens

# GUI #
width = 400
height = 510

window = Tk()
window.resizable(False,False)
# window.geometry(str(width)+"x"+str(height))
window.title("01Unit") # Reference to Alternative. 15 billion parameters in the palm of your hand! Except slightly fewer.
# frame = Frame(window,width=300,height=300)
# frame.pack()
# frame.place(anchor='center',relx=0.5,rely=0.5)
# img = ImageTk.PhotoImage(Image.open("test.png"))

def process_image(i): # i is string, path to image
    img = Image.open(i)
    img = img.resize((img.size[0],img.size[1]), 
                    #  resample=Image.ANTIALIAS
                     ) # copy the image so that we don't make the original way too small
    img.thumbnail((400,400))
    new_image = ImageTk.PhotoImage(img)
    return new_image


# img = Image.open("test.png")
# img = img.resize((img.size[0],img.size[1]), resample=Image.ANTIALIAS) # copy the image so that we don't make the original way too small
# # img_resized = img.resize((300,300))
# img.thumbnail((400,400))
default_img = process_image("default.png")#
happy = process_image("happy.png")#
angry = process_image("angry.png")#
sad = process_image("sad.png")
surprise = process_image("surprise.png")#
love = process_image("love.png")#
fear = process_image("fear.png")#

canvas = Canvas(window,width=400,height=450,bg='#000')
canvas.pack()

canvas.create_image(width/2,height/2,anchor=CENTER,image=default_img)
canvas.create_image(width/2,height/2,anchor=CENTER,image=happy)
def expression(emotion):
    canvas.delete("all")
    if emotion == "sadness":   
        canvas.create_image(width/2,height/2,anchor=CENTER,image=sad)
    elif emotion == "joy":
        canvas.create_image(width/2,height/2,anchor=CENTER,image=happy)
    elif emotion == "love":
        canvas.create_image(width/2,height/2,anchor=CENTER,image=love)
    elif emotion == "anger":
        canvas.create_image(width/2,height/2,anchor=CENTER,image=angry)
    elif emotion == "fear":
        canvas.create_image(width/2,height/2,anchor=CENTER,image=fear)
    elif emotion == "surprise":
        canvas.create_image(width/2,height/2,anchor=CENTER,image=surprise)
    else:
        print(emotion)
        canvas.create_image(width/2,height/2,anchor=CENTER,image=default_img)

# image_label = Label(frame,image = img)
# image_label.pack(fill = "x")
response = Label(window,text="",font=('Arial',17,'bold'),fg='#84F692',bg='#000')
response.pack(fill = "x")
chat_box = Entry(window,font=('Arial',17,'bold'),fg='#D5D5E8',bg='#000')
chat_box.pack(fill = "x")

def key_pressed(e):
    if (e.keysym == "Return"):
        generation = generate(chat_box.get())
        response.configure(text = generation[0])
        expression(generation[1])
        chat_box.configure(text="") # Clear written text

def entered_text():
    res = "REPLACE WITH MODEL CODE" + chat_box.get()
    response.configure(text = res) 

window.bind("<KeyRelease>",key_pressed)
# window.rowconfigure(11)
# window.columnconfigure(11)
# cols, rows = window.grid_size()

# for col in range(cols):
#     window.grid_columnconfigure(col,minsize=20)

# for row in range(rows):
#     window.grid_rowconfigure(row,minsize=20)

window.mainloop() 
