from pathlib import Path
import sys
import os
os.chdir(Path(__file__).parents[1].as_posix())
sys.path.append(Path(__file__).parents[1].as_posix())

import threading
import multiprocessing
import logging
import numpy as np
from stream.ffmpeg_stream import get_video_info, start_ffmpeg_process2, write_video_frame, write_audio_frame
import subprocess
import wave

logger = logging.getLogger(__name__)

####################################################################
###################### Streaming Functions ##########################
####################################################################

face_path = "/home/ubuntu/hungbnt/SadTalker/examples/0709.mp4"
audio_file_path = "/home/ubuntu/hungbnt/SadTalker/examples/Buddhism/Buddhism_Domi.mp3"
output_port = 7863
video_output_to="socket"
video_output_path=None
fps = 25
# # checkpoint_path = None

BYTE_WIDTH = 2
audio_sr = 16000
NUM_AUDIO_SAMPLES_PER_STEP = np.ceil(audio_sr * 0.2).astype('int')  # 200 ms audio for 16000 Hz

#%%
# mel_step_size = None
# wav2lip_batch_size = None
# device = None
# face = None
# resize_factor, = None
# rotate, = None
# crop = None
# face_det_batch_size = None
# pads = None
# nosmooth = None
# box, 
# static = None
# img_siz = 256

import cv2

def get_frames_from_video(video_path, num_frames):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    frames = []
    frame_count = 0

    # Read frames until reaching the desired number or end of video
    while frame_count < num_frames and video.isOpened():
        # Read the current frame
        ret, frame = video.read()

        if ret:
            # Append the frame to the list
            frames.append(frame)
            frame_count += 1
        else:
            # Break the loop if there are no more frames
            break

    # Release the video file
    video.release()

    return frames

def video_inference(fifo_filename_video, audio_inqueue, fps,
                      BYTE_WIDTH, NUM_AUDIO_SAMPLES_PER_STEP, audio_sr):
    # Setup video streaming pipe:
    fifo_video_out = open(fifo_filename_video, "wb")
    frames_done = 0
    audio_received = 0.0
    audio_data = audio_inqueue.get()
    while len(audio_data) == NUM_AUDIO_SAMPLES_PER_STEP * BYTE_WIDTH:
        # break when exactly desired length not received (so very last packet might be lost)
        audio_received += NUM_AUDIO_SAMPLES_PER_STEP / audio_sr
        frames = get_frames_from_video(face_path, 1/audio_received)
        for frame in frames:
            frames_done += 1
            # write to pipe
            write_video_frame(fifo_video_out, frame)

        print('Generated', frames_done, 'frames from', '{:.1f}'.format(audio_received), 's of received audio')

        audio_data = audio_inqueue.get()

        if audio_data == 'BREAK':
            print("=" * 50)
            print('Closing Fifo Video')
            print("=" * 50)
            fifo_video_out.close()
            break

def audio_thread_handler(fifo_filename_audio, audio_inqueue):
    """
    receive audio from audio_inqueue and write to fifo_filename_audio pipe in chunks
    """
    fifo_audio_out = open(fifo_filename_audio, "wb")
    # this blocks until the read for the fifo opens so we run in separate thread

    # read frame one by one, process and write to fifo pipe
    while True:
        in_audio_frame = audio_inqueue.get()
        # if len(in_audio_frame) == 0:
        #     break
        if in_audio_frame == 'BREAK':
            break
        write_audio_frame(fifo_audio_out, in_audio_frame)
    fifo_audio_out.close()
    
def audio_input_thread_handler(outqueues,audio_file_path, BYTE_WIDTH, NUM_AUDIO_SAMPLES_PER_STEP, audio_sr):
    """
    function to take input audio connection as is, and write output to queues
    """
    # audio file case
    print('Extracting raw audio...')
    temp_audio_file = 'temp.wav'
    # convert to 16 kHz wav file
    command = 'ffmpeg -y -i {} -strict -2 -ar {} {}'.format(audio_file_path, audio_sr, temp_audio_file)
    print(command)
    subprocess.call(command, shell=True)
    wf = wave.open(temp_audio_file, 'rb')
    # we send wav audio in 200 ms chunks, with gap between sending chunks to simulate real-time transmission
    assert wf.getsampwidth() == BYTE_WIDTH
    assert wf.getframerate() == audio_sr
    assert wf.getnchannels() == 1  # Only supports mono channel audio (not stereo)
    print("start sending wav file")
    audio_sent = 0.0  # in s
    while True:
        audio_bytes_to_write = wf.readframes(NUM_AUDIO_SAMPLES_PER_STEP)
        if len(audio_bytes_to_write) == 0:
            break
        for q in outqueues:
            q.put(audio_bytes_to_write)
    for q in outqueues:
        q.put('BREAK')
    os.remove(temp_audio_file)
    
def stream():
    """
    Handles all the threads
    """
    width, height = get_video_info(face_path)

    # fifo pipes (remove file name if already exists)
    fifo_filename_video = '/tmp/fifovideo'
    fifo_filename_audio = '/tmp/fifoaudio'
    fifo_resemble_tts = '/tmp/fiforesembletts'

    if os.path.exists(fifo_filename_video):
        os.remove(fifo_filename_video)
    if os.path.exists(fifo_filename_audio):
        os.remove(fifo_filename_audio)
    if os.path.exists(fifo_resemble_tts):
        os.remove(fifo_resemble_tts)

    os.mkfifo(fifo_filename_video)
    os.mkfifo(fifo_filename_audio)
    os.mkfifo(fifo_resemble_tts)
    logger.info('fifo exists now')

    process2 = start_ffmpeg_process2(fifo_filename_video, fifo_filename_audio, width, height, fps,
                                                   output_port, video_output_to, video_output_path)
    logger.info('Output pipe set')

    # queues for sending audio packets from T1 (audio receiving) to T2 (audio generation) and T3
    # (video generation) at unlimited capacity
    audio_packet_queue_T2 = multiprocessing.Queue()
    audio_packet_queue_T3 = multiprocessing.Queue()

    # we run audio and video in separate threads otherwise the fifo opening blocks
    outqueue_list = [audio_packet_queue_T2, audio_packet_queue_T3]

    audio_input_thread = multiprocessing.Process(target=audio_input_thread_handler,args=(outqueue_list, audio_file_path,BYTE_WIDTH, NUM_AUDIO_SAMPLES_PER_STEP,audio_sr))
    logger.info('T4: Audio input thread launched -- Audio Input')
    video_thread = threading.Thread(target=video_inference, args=(fifo_filename_video, audio_packet_queue_T2, fps, BYTE_WIDTH, NUM_AUDIO_SAMPLES_PER_STEP, audio_sr))
    logger.info('T2: Video thread launched')
    audio_thread = multiprocessing.Process(target=audio_thread_handler, args=(fifo_filename_audio, audio_packet_queue_T3))
    logger.info('T3: Audio thread launched')

    
    audio_input_thread.start()
    video_thread.start()
    audio_thread.start()
    audio_input_thread.join()
    video_thread.join()
    audio_thread.join()

    logger.info('Waiting for ffmpeg process2')
    process2.wait()

    os.remove(fifo_filename_video)
    os.remove(fifo_filename_audio)
    os.remove(fifo_resemble_tts)
    logger.info('Done')


def main():
    stream()


if __name__ == '__main__':
    main()
