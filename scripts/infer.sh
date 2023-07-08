img="examples/YuiHatano.mp4"
audio="examples/Buddhism_Domi.mp3"
output_dir="examples/output"
ref_video="examples/WDA_KatieHill_000.mp4"

cmd="python3.8 inference.py --driven_audio $audio
           --source_image $img
           --result_dir $output_dir 
           --still
           --preprocess full 
           --enhancer gfpgan"
echo $cmd
exec $cmd
