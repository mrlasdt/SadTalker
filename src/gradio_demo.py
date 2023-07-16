import torch, uuid
import os, sys, shutil
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data

from src.utils.init_path import init_path
from src.utils.timer import Timer
from pydub import AudioSegment
import gc

# import multiprocessing
# import threading
# from stream.inference_streaming_pipeline_dummy import (
#     audio_input_thread_handler,
#     audio_thread_handler,
#     write_video_frame,
#     write_audio_frame,
#     get_video_info,
#     start_ffmpeg_process2,
#     fps,
#     output_port,
#     video_output_to,
#     video_output_path,
#     BYTE_WIDTH,
#     NUM_AUDIO_SAMPLES_PER_STEP,
#     audio_sr,
# )
# import logging

# logger = logging.getLogger(__name__)


def mp3_to_wav(mp3_filename, wav_filename, frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename, format="wav")


class SadTalker:
    def __init__(
        self, checkpoint_path="checkpoints", config_path="src/config", lazy_load=False
    ):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.device = device

        os.environ["TORCH_HOME"] = checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.dcfg_models = {}
        self.init()
        self.warm_up()

    def init(self):
        for size in [256, 512]:
            self.sadtalker_paths = init_path(
                self.checkpoint_path, self.config_path, size, False, preprocess="crop"
            )
            audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
            preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
            self.dcfg_models[size] = (audio_to_coeff, preprocess_model)

        for preprocess in ["full", "crop"]:
            self.sadtalker_paths = init_path(
                self.checkpoint_path,
                self.config_path,
                size=256,
                old_version=False,
                preprocess=preprocess,
            )
            animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)
            self.dcfg_models[preprocess] = animate_from_coeff

    def warm_up(self, warm_up_steps=3, tmp_save_path="./results/tmp/"):
        # with Timer("Warming up"):
        for _ in range(warm_up_steps):
            for size, preprocess in zip([256, 512], ["crop", "full"]):
                self.test(
                    source_image="examples/source_image/art_2.png",
                    driven_audio="examples/driven_audio/RD_Radio31_000.wav",
                    size=size,
                    preprocess=preprocess,
                    result_dir=tmp_save_path,
                    is_warming_up=True,
                )
        shutil.rmtree(tmp_save_path)

    def test(
        self,
        source_image,
        driven_audio,
        preprocess="crop",
        still_mode=False,
        use_enhancer=False,
        size=256,
        pose_style=0,
        # batch_size=1,
        exp_scale=1.0,
        use_ref_video=False,
        ref_video=None,
        ref_info=None,
        use_idle_mode=False,
        length_of_audio=0,
        use_blink=True,
        result_dir="./results/",
        is_warming_up=False,
        is_stream=False,
    ):
        # with Timer("Inference time"):
        print("[INFO]: Inferencing...")
        batch_size = 1  # TODO: fix this hardcode
        audio_to_coeff, preprocess_model = self.dcfg_models[size]
        animate_from_coeff = self.dcfg_models[preprocess]
        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        print(source_image)
        pic_path = os.path.join(input_dir, os.path.basename(source_image))
        if is_warming_up:
            shutil.copy2(source_image, input_dir)
        else:
            shutil.move(source_image, input_dir)

        if driven_audio is not None and os.path.isfile(driven_audio):
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))

            #### mp3 to wav
            if ".mp3" in audio_path:
                mp3_to_wav(driven_audio, audio_path.replace(".mp3", ".wav"), 16000)
                audio_path = audio_path.replace(".mp3", ".wav")
            else:
                if is_warming_up:
                    shutil.copy2(driven_audio, input_dir)
                else:
                    shutil.move(driven_audio, input_dir)

        elif use_idle_mode:
            audio_path = os.path.join(
                input_dir, "idlemode_" + str(length_of_audio) + ".wav"
            )  ## generate audio from this new audio_path
            from pydub import AudioSegment

            one_sec_segment = AudioSegment.silent(
                duration=1000 * length_of_audio
            )  # duration in milliseconds
            one_sec_segment.export(audio_path, format="wav")
        else:
            print(use_ref_video, ref_info)
            assert use_ref_video == True and ref_info == "all"

        if use_ref_video and ref_info == "all":  # full ref mode
            ref_video_videoname = os.path.basename(ref_video)
            audio_path = os.path.join(save_dir, ref_video_videoname + ".wav")
            print("new audiopath:", audio_path)
            # if ref_video contains audio, set the audio from ref_video.
            cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s" % (
                ref_video,
                audio_path,
            )
            os.system(cmd)

        os.makedirs(save_dir, exist_ok=True)

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            pic_path, first_frame_dir, preprocess, True, size
        )

        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        if use_ref_video:
            print("using ref video for genreation")
            ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
            ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
            os.makedirs(ref_video_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing pose")
            ref_video_coeff_path, _, _ = preprocess_model.generate(
                ref_video, ref_video_frame_dir, preprocess, source_image_flag=False
            )
        else:
            ref_video_coeff_path = None

        if use_ref_video:
            if ref_info == "pose":
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = None
            elif ref_info == "blink":
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == "pose+blink":
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == "all":
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = None
            else:
                raise ("error in refinfo")
        else:
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = None

        # audio2ceoff
        if use_ref_video and ref_info == "all":
            coeff_path = ref_video_coeff_path  # self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
        else:
            batch = get_data(
                first_coeff_path,
                audio_path,
                self.device,
                ref_eyeblink_coeff_path=ref_eyeblink_coeff_path,
                still=still_mode,
                idlemode=use_idle_mode,
                length_of_audio=length_of_audio,
                use_blink=use_blink,
            )  # longer audio?
            coeff_path = audio_to_coeff.generate(
                batch, save_dir, pose_style, ref_pose_coeff_path
            )

        # coeff2video
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            still_mode=still_mode,
            preprocess=preprocess,
            size=size,
            expression_scale=exp_scale,
        )
        return_path_or_frames = animate_from_coeff.generate(
            data,
            save_dir,
            pic_path,
            crop_info,
            enhancer="gfpgan" if use_enhancer else None,
            preprocess=preprocess,
            img_size=size,
        )
        if is_stream:
            return return_path_or_frames
        video_name = data["video_name"]
        print(f"The generated video is named {video_name} in {save_dir}")
        if is_warming_up:
            os.remove(os.path.join(input_dir, os.path.basename(source_image)))
            os.remove(os.path.join(input_dir, os.path.basename(driven_audio)))
        # del self.preprocess_model
        # del self.audio_to_coeff
        # del self.animate_from_coeff
        if torch.cuda.is_available():
            print("[INFO]: Clearing cache...")

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
        return return_path_or_frames

    # def video_inference(
    #     self,
    #     fifo_filename_video,
    #     audio_inqueue,
    #     BYTE_WIDTH,
    #     NUM_AUDIO_SAMPLES_PER_STEP,
    #     audio_sr,
    #     source_image,
    #     preprocess="crop",
    #     still_mode=False,
    #     use_enhancer=False,
    #     size=256,
    #     pose_style=0,
    #     exp_scale=1.0,
    #     result_dir="./results/",
    #     is_warming_up=False,
    # ):
    #     # Setup video streaming pipe:
    #     fifo_video_out = open(fifo_filename_video, "wb")
    #     frames_done = 0
    #     audio_received = 0.0
    #     audio_data = audio_inqueue.get()
    #     while len(audio_data) == NUM_AUDIO_SAMPLES_PER_STEP * BYTE_WIDTH:
    #         # break when exactly desired length not received (so very last packet might be lost)
    #         audio_received += NUM_AUDIO_SAMPLES_PER_STEP / audio_sr
    #         tmp_audio_path = "/tmp/audio_sadtalker.wav"
    #         write_audio_frame(tmp_audio_path, audio_data)

    #         frames = self.test(
    #             source_image,
    #             tmp_audio_path,
    #             preprocess,
    #             still_mode,
    #             use_enhancer,
    #             size,
    #             pose_style,
    #             exp_scale,
    #             result_dir,
    #             is_warming_up,
    #             is_stream=True,
    #         )

    #         for frame in frames:
    #             frames_done += 1
    #             # write to pipe
    #             write_video_frame(fifo_video_out, frame)

    #         print(
    #             "Generated",
    #             frames_done,
    #             "frames from",
    #             "{:.1f}".format(audio_received),
    #             "s of received audio",
    #         )

    #         audio_data = audio_inqueue.get()

    #         if audio_data == "BREAK":
    #             print("=" * 50)
    #             print("Closing Fifo Video")
    #             print("=" * 50)
    #             fifo_video_out.close()
    #             break

    # def stream(
    #     self,
    #     source_image,
    #     driven_audio,
    #     preprocess="crop",
    #     still_mode=False,
    #     use_enhancer=False,
    #     size=256,
    #     pose_style=0,
    #     exp_scale=1.0,
    #     result_dir="./results/",
    #     is_warming_up=False,
    # ):
    #     """
    #     Handles all the threads
    #     """
    #     width, height = get_video_info(source_image)

    #     # fifo pipes (remove file name if already exists)
    #     fifo_filename_video = "/tmp/fifovideo"
    #     fifo_filename_audio = "/tmp/fifoaudio"
    #     fifo_resemble_tts = "/tmp/fiforesembletts"

    #     if os.path.exists(fifo_filename_video):
    #         os.remove(fifo_filename_video)
    #     if os.path.exists(fifo_filename_audio):
    #         os.remove(fifo_filename_audio)
    #     if os.path.exists(fifo_resemble_tts):
    #         os.remove(fifo_resemble_tts)

    #     os.mkfifo(fifo_filename_video)
    #     os.mkfifo(fifo_filename_audio)
    #     os.mkfifo(fifo_resemble_tts)
    #     print("fifo exists now")

    #     process2 = start_ffmpeg_process2(
    #         fifo_filename_video,
    #         fifo_filename_audio,
    #         width,
    #         height,
    #         fps,
    #         output_port,
    #         video_output_to,
    #         video_output_path,
    #     )
    #     print("Output pipe set")

    #     # queues for sending audio packets from T1 (audio receiving) to T2 (audio generation) and T3
    #     # (video generation) at unlimited capacity
    #     audio_packet_queue_T2 = multiprocessing.Queue()
    #     audio_packet_queue_T3 = multiprocessing.Queue()

    #     # we run audio and video in separate threads otherwise the fifo opening blocks
    #     outqueue_list = [audio_packet_queue_T2, audio_packet_queue_T3]

    #     audio_input_thread = multiprocessing.Process(
    #         target=audio_input_thread_handler,
    #         args=(
    #             outqueue_list,
    #             driven_audio,
    #             BYTE_WIDTH,
    #             NUM_AUDIO_SAMPLES_PER_STEP,
    #             audio_sr,
    #         ),
    #     )
    #     print("T4: Audio input thread launched -- Audio Input")
    #     video_thread = threading.Thread(
    #         target=self.video_inference,
    #         args=(
    #             fifo_filename_video,
    #             audio_packet_queue_T2,
    #             BYTE_WIDTH,
    #             NUM_AUDIO_SAMPLES_PER_STEP,
    #             audio_sr,
    #             source_image,
    #             preprocess,
    #             still_mode,
    #             use_enhancer,
    #             size,
    #             pose_style,
    #             exp_scale,
    #             result_dir,
    #             is_warming_up,
    #         ),
    #     )
    #     print("T2: Video thread launched")
    #     audio_thread = multiprocessing.Process(
    #         target=audio_thread_handler,
    #         args=(fifo_filename_audio, audio_packet_queue_T3),
    #     )
    #     print("T3: Audio thread launched")

    #     audio_input_thread.start()
    #     video_thread.start()
    #     audio_thread.start()
    #     audio_input_thread.join()
    #     video_thread.join()
    #     audio_thread.join()

    #     print("Waiting for ffmpeg process2")
    #     process2.wait()

    #     os.remove(fifo_filename_video)
    #     os.remove(fifo_filename_audio)
    #     os.remove(fifo_resemble_tts)

    #     if torch.cuda.is_available():
    #         print("[INFO]: Clearing cache...")

    #         torch.cuda.empty_cache()
    #         torch.cuda.synchronize()

    #     gc.collect()
    #     print("Done")
