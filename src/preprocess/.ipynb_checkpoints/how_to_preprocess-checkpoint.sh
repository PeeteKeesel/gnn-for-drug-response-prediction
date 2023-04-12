
$ pip install rdkit-pypi


# TODO
python3 main_80_20.py \
    --seed=42 \
    --download=n \
    --process=y \
    --raw_path=../data/raw/ \
    --processed_path=../data/processed/ \
    --combined_score_thresh=700 \
    --build_model=n

# DONE
python3 main_80_20.py \
    --seed=42 \
    --download=n \
    --process=y \
    --raw_path=../data/raw/ \
    --processed_path=../data/processed/ \
    --combined_score_thresh=800 \
    --build_model=n
    
# used env2803.yml
# DONE
python3 main_80_20.py \
    --seed=42 \
    --download=n \
    --process=y \
    --raw_path=../data/raw/ \
    --processed_path=../data/processed/ \
    --combined_score_thresh=850 \
    --build_model=n

# used env2803.yml
# DONE
python3 main_80_20.py \
    --seed=42 \
    --download=n \
    --process=y \
    --raw_path=../data/raw/ \
    --processed_path=../data/processed/ \
    --combined_score_thresh=950 \
    --build_model=n
        