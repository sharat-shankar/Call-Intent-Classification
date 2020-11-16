##Replace the Google speech to text api code in the classification.py file to azure 

def speech_to_text(audio1):
    speech_key, service_region = " b032c7c5b3e344b09c049f96e096fb6d", "westus"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    audio_config = speechsdk.audio.AudioConfig(filename=audio1)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,audio_config=audio_config)

    done = False
    def stop_cb(evt):
        """callback that stops continuous recognition upon receiving an event evt"""
        print('CLOSING on {}'.format(evt))
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True
    all_results = []
    def handle_final_result(evt):
        all_results.append(evt.result.text)
    speech_recognizer.recognized.connect(handle_final_result)
    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)
    result_data = ' '.join(all_results)
    return(result_data)
