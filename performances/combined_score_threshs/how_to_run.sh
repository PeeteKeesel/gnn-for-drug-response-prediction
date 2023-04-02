
# TODO
# For combined_score_thresh of 700.
# logging file: logfile_model_graphgraph_v1_gdsc2_700_42_GatGin700Max150
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphGraph \
    --conv_type=GATConv \
    --conv_layers=3 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=combined_score_threshs/ \
    --combined_score_thresh=700 \
    --file_ending=GatGin700Max150

# For combined_score_thresh of 800.
# logging file: logfile_model_graphgraph_v1_gdsc2_800_42_GatGin800Max150
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphGraph \
    --conv_type=GATConv \
    --conv_layers=3 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=combined_score_threshs/ \
    --combined_score_thresh=800 \
    --file_ending=GatGin800Max150
    
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphGraph \
    --conv_type=GATConv \
    --conv_layers=3 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=combined_score_threshs/ \
    --combined_score_thresh=900 \
    --file_ending=GatGin900Max150    