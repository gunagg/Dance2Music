#!/bin/bash
inp_dir=$1
output_dir=$2
out_images_extracted_dir="images_extracted"
out_imgs_dir="output_imgs_AMT"
out_json_dir="output_jsons_AMT"
out_feats_dir="output_feats/"

rm -r ${out_images_extracted_dir}
rm -r ${out_feats_dir}

mkdir -p ${out_feats_dir}
mkdir -p ${out_json_dir}
mkdir -p ${out_imgs_dir}
mkdir -p ${out_images_extracted_dir}

#Extract frames from the videos
python extract_frames.py --inp_dir ${inp_dir} --out_dir ${out_images_extracted_dir}

size=$(ls ${out_images_extracted_dir} | wc -l)

echo "the size is "${size}

#Run openpose over all the videos

for idx in 0 .. $((size-1))
do

./build/examples/openpose/openpose.bin --image_dir ${out_images_extracted_dir}"/"${idx}  --keypoint_scale 4 --write_video output.mp4  --write_video_fps 30 --display 0 --disable_blending  --write_json ${out_json_dir}"/"${idx} --write_images ${out_imgs_dir}"/"${idx}

done

#Save poses 
python save_feats_only.py --input_json_dir ${out_json_dir} --output_feats_dir ${out_feats_dir}

rm -r ${out_json_dir}
rm -r ${out_imgs_dir}

#Generate music
python generate_music_BS.py --dancepath ${out_feats_dir} --imagespath ${out_images_extracted_dir}  --visfolder ${output_dir}

