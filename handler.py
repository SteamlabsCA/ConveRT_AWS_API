#Import the required dependencies
import tensorflow_hub as tfhub
import tensorflow as tf
import tensorflow_text

import numpy as np
from scipy import special
import pandas as pd

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import boto3

import mysql.connector
import json
import datetime
import hashlib

import login

class Handler:
    def __init__(self, config):
        self.sess = tf.compat.v1.InteractiveSession(graph=tf.Graph()) #Tensorflow session used in interactive contexts

        module = tfhub.Module("https://github.com/AndyForest/PolyAI-model/raw/master/models/model.tar.gz") #Draw conveRT AI model from repository 

        self.text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None]) #Placeholder for a tensor that will always be fed 
        self.context_encoding_tensor = module(self.text_placeholder, signature="encode_context") #Context(For AI response) Encoding tensor 
        self.response_encoding_tensor = module(self.text_placeholder, signature="encode_response") #Response Encoding tensor

        self.sess.run(tf.compat.v1.tables_initializer()) #Initialize tables within tensorflow moddels
        self.sess.run(tf.compat.v1.global_variables_initializer()) #Initialize globale variables within tensorflow moddels

        #Setup GPT-2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {self.device}")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)

        #Setup Amazon translate
        self.translate = boto3.client(
            service_name=login.service_name,
            region_name=login.region_name,
            aws_access_key_id=login.aws_access_key_id,
            aws_secret_access_key=login.aws_secret_access_key
        )



    def handle_post(self, payload):

        responses = payload["responseList"] #Store payload list (potential responses)
        prompt = payload["inputPrompt"] #Store payload prompt
        response_encodings = [] #List where response encodings will be stored
        
        #Setup mySQL database connector
        host = login.host
        user = login.user
        password = login.password
        database = login.database
        self.cnx = mysql.connector.connect(user=user, password=password,
                              host=host,
                              database=database)
        self.cursor = self.cnx.cursor()
        self.add_apicall = ("INSERT INTO Response "
               "(SHA256, ResponseList, Parameters, TimeStamp) "
               "VALUES (%(SHA256)s, %(ResponseList)s, %(Parameters)s, %(TimeStamp)s)")
               
        #Read SQL database 
        self.cursor.execute("SELECT * FROM Response")
        #Store SQL database in a variable
        SQLdb = self.cursor.fetchall()
        SQLDecision=0 #Variable for storage decisions 

        #Check if responseList payload sent was a Response List or a SHA256 Hash
        if (type(responses)==list): #If a Response List is sent hash it and store data in mySQL database
            HashedList=""
            for x in responses: #Prepared list for hashing by combining each response together into one big string
                HashedList=HashedList+x+" "
            HashedList=HashedList[:-1]
            HashedList = hashlib.sha256(str(HashedList).encode()).hexdigest() #Hash Response List
            listJSON=json.dumps(responses) #Turn Response list into json so mySQL can store it
            Parameters=json.dumps(payload["language"]) #Send parameters 
            #dt stores current date/time
            dt = datetime.datetime.now()
            dt = str(dt)
            #Prepare data to enter into mySQL database
            data_apicall = {'SHA256' : HashedList, 'ResponseList' : listJSON, 'Parameters' : Parameters, 'TimeStamp' : dt }
            #Check if this list is already in mySQL database, if so skip adding it
            for x in SQLdb:
                if(x[0] == HashedList):
                    SQLDecision=1 #Decision turns to 1 if this hash is already in database
            if (SQLDecision == 0): #Only add in new row into mySQL if data does not exist 
                #Insert new data
                self.cursor.execute(self.add_apicall, data_apicall)
                #Ensure data is committed to the database
                self.cnx.commit()
        
        else: #If a response list is not sent and a SHA256 Hash is sent
            for x in SQLdb:
                if (x[0] == responses): #If the Hash is in the mySQL database
                    SQLDecision=1 #Decision turns to 1 if this hash is already in database
                    responses = x[1] #Load the response list corresponding to the hash into 'responses' variable
                    #Process the responses back into a list (it exists as a text blob in mySQL)
                    responses=responses[1:] #Drop first character (Extra square bracket)
                    responses=responses[:-1] #Drop last character (Extra square bracket)
                    responses=str(responses).replace('"','') #Replace the double quotes from each response
                    responses=responses.split(',') #Split by comma (Original responses)
            if (SQLDecision == 0): #If Hash is not inside database
                #We return a -1 to let the front end know that it needs to send a response list
                return -1
        #Close mySQL cursor & connection(cnx)
        #self.cursor.close()
        #self.cnx.close()
        #Check if french mode is enabled - if so translate prompt(french) and responses(french) to english for processing
        if (payload["language"] == 'FR'):

            #Translate french prompt to english 
            result = self.translate.translate_text(Text=prompt, 
            SourceLanguageCode="fr", TargetLanguageCode="en")
            prompt = result.get('TranslatedText')

            #Translate french list of responses to english
            for i in range(len(responses)):
                result = self.translate.translate_text(Text=responses[i], 
                SourceLanguageCode="fr", TargetLanguageCode="en")
                responses[i] = result.get('TranslatedText')

        # Encode the responses in batches of 64.
        batch_size = 64
        response_encodings = []
        for i in range(0, len(responses), batch_size): #Loop through the list of responses and encode each one
            batch = responses[i:i + batch_size]
            response_encodings.append(self.sess.run(self.response_encoding_tensor, feed_dict={self.text_placeholder: (batch)})) #This line encodes responses
        response_encodings = np.concatenate(response_encodings)

        #Consider top 5 responses 
        responseChoices=payload["consider"] 
        # Score responses using ConveRT
        context_encoding = self.sess.run(self.context_encoding_tensor, feed_dict={self.text_placeholder: [prompt]}) #This line encodes context (inputPrompt)
        scores = np.dot(response_encodings, context_encoding.T)

        # Pick top 1 response
        top_index = np.argmax(scores)
        top_score = float(scores[top_index])
        response = f"[{top_score:.3f}] {responses[top_index]}"

        # Find top {responseChoices} responses, and randomly pick one. Use score squared to prefer the higher ranked responses
        # Note: scores is an array of single item arrays. Not sure why.
        scoresArray = np.asarray(scores).reshape(-1)
        top_scores = np.argsort(scoresArray)
    
        top_n_scores = top_scores[-responseChoices:]
  
        #Softmax has squaring the probabilities built in
        top_n_scores_p = special.softmax([(scoresArray[item]) for item in top_n_scores])

        responseChoice = np.random.choice(top_scores[-responseChoices:], p=top_n_scores_p)
        top_score = scores[responseChoice]
        response = responses[responseChoice]
        
        #GPT-2 SECTION
        #Check if Response has a word in curly brackets ("{..}"), if so drop it 
        if ('{' in response):
            sep='{'
            response = response.split(sep,1)[0]
            
            #Generate new text to replace the dropped word
            input_length = len(response.split())
            tokens = self.tokenizer.encode(response, return_tensors="pt").to(self.device)
            prediction = self.model.generate(tokens, max_length=input_length + 50, do_sample=True)
            response = self.tokenizer.decode(prediction[0])

            #If statement to end sentance at a period or comma so text generation does not end on a cliff hanger.
            #if ('.' in response):
            #    sep = '.'
            #    response = response.split(sep, 1)[0]

        #Check if french mode is enabled - if so translate selected response back to french 
        if (payload["language"] == 'FR'):
            result = self.translate.translate_text(Text=response, 
            SourceLanguageCode="en", TargetLanguageCode="fr")
            response = result.get('TranslatedText')

        #Return selected response
        return response
