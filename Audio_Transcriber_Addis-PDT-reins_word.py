# -*- coding: utf-8 -*-

#author          :Alireza Tajadod in Dr.Jennifer Ryan's lab
#date            :02/05/18
#usage           :python3 pyscript.py
#documentation   :Audio_Transcriber.pdf
#python_version  :3.6 
#author-contact  :atajadod@research.baycrest.org

import os
import csv
import sys
import argparse 
import google.cloud.speech	
import google.cloud.storage	

parser = argparse.ArgumentParser(description='Audio transcriber')

parser.add_argument('--gui', type=str, nargs='?', default = 'on',
                   help='Gui needs to be on or off')

parser.add_argument('--folders', type=str, nargs='*', default = [],
                   help='Data folders to transcribe')

args = parser.parse_args()
if args.gui:
	gui_mode = args.gui
else:
	gui_mode = 'on'

if gui_mode == 'off':
	try: 
		Data_folders = args.folders
		print( 'Data Folders set : ' + ' '.join(str(e) for e in Data_folders))

	except:
		print ('You must provide folders if --gui is off. Use -h for more information')
		quit()


## =================  IMPORT STATEMENTS  =================

try:

	print(sys.version) 							

	from google.cloud import storage
	from google.cloud import speech_v1p1beta1 as speech
##	from google.cloud.speech import enums
##	from google.cloud.speech import types

except (ImportError) as error:
	# Output expected ImportErrors.
	print(error)
except (Exception) as exception:
	# Output unexpected Exceptions.
	print(exception, False)

print(' Initializing... ')

if gui_mode == 'on':
	# Gui imports 
	from tkinter import Tk
	from tkinter import messagebox
	from tkinter import filedialog

	root = Tk()
	root.withdraw() # we don't want a full GUI, so keep the root window from appearing
	root.update()

# =================  GOOGLE CREDENTIAL =================

# put your credentials here
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

if gui_mode == 'off': ## Can be turned on/off depending on preference. Currently turned 'off'
	change_credential = messagebox.askyesno("Audio_Transcriber","Would you like to change the default Google Credential?")
	if change_credential:
		filename = askopenfilename(title = 'Please Select your credential File') # show an "Open" dialog box and return the path to the selected file

		os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = filename
 
# ====  Setting up the storage  ====		
try: 
	storage_client = storage.Client()
	print('Google account access: Success')
except Exception as error1:
	print('Can not access Service account')
	print(error1)
	print(os.getcwd())

# ====  Setting up the bucket  ====
# put your bucket name here
try: 
	bucket_name = ''
	bucket = storage_client.get_bucket(bucket_name)
	print('Bucket account access: Success')
except Exception as error2:
	print("Bucket access error")
	print(error2)

# ====  Setting up the speech recognizer  ====  
try: 
	speech_client = speech.SpeechClient()
	print('Speech account access: Success')
except Exception as error3:
	print ("Speech_recognition access error")
	print(error3)


# =================  AUDIO SET UP  =================

uri = bucket_name + '/audio_data'

# AF221130 rewrote structure
config = {
    "language_code": 'en-US',
    "use_enhanced": True,
    "enable_word_time_offsets":True,
    "enable_automatic_punctuation": True,
    "encoding":'LINEAR16',
    "model": 'video'}
# put your audio file here in form of an uri
audio = {}

# =================  INPUT FOLDERS  =================
if gui_mode == 'on':
	Data_folders = []
	title = 'Please select your input folder or cancel if finished'
	while True:
		dir = filedialog.askdirectory(title=title)
		if not dir:	
			break
		Data_folders.append(dir)
		title = 'got %s. Next dir' % dir

	root.update()

# =================  CREATING OUTPUTS  =================
for Data_folder in Data_folders:
	Output_folder = Data_folder + '/Transcript_auto'
	try:
		os.mkdir(Output_folder)
#		os.mkdir(Output_folder + 'Transcript')		
	except OSError as exception:
		if not os.path.isdir(Output_folder):
			raise

	print('Getting All files from: ' + Data_folder)
	print('Outputfolder set to: ' + Output_folder)

	for file in os.listdir(Data_folder):

		filename_base = os.path.splitext(file)[0]
		filename_ext = os.path.splitext(file)[-1]
        
		filename,word,start,end,confidence = ['filename'],['word'],['start'],['end'],['confidence']

		if filename_ext in ['.wav', '.mp3']  :   # more audio extensions can be added here. eg:  if filename_ext in ['.wav', '.mp3', '.extension'] ## The script only works with WAV 16-bit PCM
			trial = os.path.join(Data_folder , file)
			blob = bucket.blob('audio_data')
			blob.upload_from_filename(trial)
			operation = speech_client.long_running_recognize(config=config, audio=audio) ###

			
			print('Now trying : ' + filename_base)
			print('Waiting for operation to complete...')
			result = operation.result(timeout = 40000)
			print(filename_base + ': Successful')
 
			for myresult in result.results:
				alternative = myresult.alternatives[0]
				for word_info in alternative.words:
					filename.append(filename_base)
					word.append(word_info.word)
					start.append(word_info.start_time.total_seconds())
					end.append(word_info.end_time.total_seconds())
					confidence.append(word_info.confidence)
				rows = zip(filename, word, start, end, confidence)
				with open(f'{Output_folder}/{filename_base}_auto-transcript.csv', "w", newline='') as f:
					writer = csv.writer(f)
					for row in rows:
						writer.writerow(row)
				print( 'Successfully created: ' + f'{Output_folder}/{filename_base}_auto-transcript.csv')
try:
   blob.delete()
except NameError:
	print ('No audio_files were transcribed')
