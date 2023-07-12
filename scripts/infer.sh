img="examples/Buddhism/YuiHatano.mp4"
audio="examples/Buddhism/Buddhism_Domi.mp3"
output_dir="examples/Buddhism/output"
ref_video="examples/Buddhism/WDA_KatieHill_000.mp4"

cmd="python inference.py --driven_audio $audio
           --source_image $img
           --result_dir $output_dir 
           --still
           --preprocess full 
           --enhancer gfpgan"
echo $cmd
exec $cmd
