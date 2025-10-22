#download the laion-400m data using the img2dataset command

wget -l1 -r --no-parent https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/
mv the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/ .

img2dataset --url_list laion400m-meta --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion400m-data --processes_count 16 --thread_count 128 --image_size 224\
             --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb True


#select some tars as training data and some for validation data
