# env 2803
# --------------------------------------------------------------------------- #
# Combined Score Experiments
# --------------------------------------------------------------------------- #
# DONE
# For combined_score_thresh of 700.
# log file: logfile_model_graphgraph_v1_gdsc2_700_42_GatGin700Max150
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

# DONE
# For combined_score_thresh of 800.
# log file: logfile_model_graphgraph_v1_gdsc2_800_42_GatGin800Max150
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

# DONE
# For combined_score_thresh of 850.
# log file: logfile_model_graphgraph_v1_gdsc2_850_42_GatGin800Max150
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
    --combined_score_thresh=850 \
    --file_ending=GatGin850Max150

# DONE
# For combined_score_thresh of 900.
# log file: logfile_model_graphgraph_v1_gdsc2_800_42_GatGin800Max150
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
    
# DONE
# For combined_score_thresh of 950.
# log file: logfile_model_graphgraph_v1_gdsc2_950_42_GatGin950Max150
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
    --combined_score_thresh=950 \
    --file_ending=GatGin950Max150    
    
    
# --------------------------------------------------------------------------- #
# Grid Experiments
# --------------------------------------------------------------------------- # 
# ----------------------------------- #
# GraphTab
# ----------------------------------- #
# DONE
# GT--GCN-2-GIN-max    log file: 
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphTab \
    --conv_type=GCNConv \
    --conv_layers=2 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=grid_experiments/ \
    --combined_score_thresh=990 \
    --file_ending=Gcn2Gin990Max150 
  
# DONE
# GT--GCN-2-GIN-mean    log file: 
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphTab \
    --conv_type=GCNConv \
    --conv_layers=2 \
    --global_pooling=mean \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=grid_experiments/ \
    --combined_score_thresh=990 \
    --file_ending=Gcn2Gin990Mean150 
    
# DONE
# GT--GCN-3-GIN-max    log file: 
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphTab \
    --conv_type=GCNConv \
    --conv_layers=3 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=grid_experiments/ \
    --combined_score_thresh=990 \
    --file_ending=Gcn3Gin990Max150

# DONE
# GT--GCN-3-GIN-mean    log file: 
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphTab \
    --conv_type=GCNConv \
    --conv_layers=3 \
    --global_pooling=mean \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=grid_experiments/ \
    --combined_score_thresh=990 \
    --file_ending=Gcn3Gin990Mean150

# GT--GAT-2-GIN-max    log file: 
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphTab \
    --conv_type=GATConv \
    --conv_layers=2 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=grid_experiments/ \
    --combined_score_thresh=990 \
    --file_ending=Gat2Gin990Max150

# GT--GAT-2-GIN-mean    log file: 

# GT--GAT-3-GIN-max    log file: 

# GT--GAT-3-GIN-mean    log file: 


python3 main_80_20_temp.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphTab \
    --conv_type=GATConv \
    --conv_layers=2 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=grid_experiments/ \
    --combined_score_thresh=990 \
    --file_ending=tempMultiple
    
    
python3 main_80_20_temp.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphGraph \
    --conv_type=GATConv \
    --conv_layers=2 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=grid_experiments/ \
    --combined_score_thresh=990 \
    --file_ending=allGraphGraph
    
# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- # 
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=TabTab \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=grid_experiments/ \
    --combined_score_thresh=990 \
    --file_ending=TabTab
    
# --------------------------------------------------------------------------- #
# Ablation Studies
# --------------------------------------------------------------------------- # 
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphGraph \
    --conv_type=GCNConv \
    --conv_layers=2 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=ablation_studies/ \
    --combined_score_thresh=990 \
    --ablation_subset=gexpr \
    --nr_node_features=1 \
    --file_ending=AblationStudy_GExpr
    
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphGraph \
    --conv_type=GCNConv \
    --conv_layers=2 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=ablation_studies/ \
    --combined_score_thresh=990 \
    --ablation_subset=cnvg \
    --nr_node_features=1 \
    --file_ending=AblationStudy_CnvGistic    
    
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphGraph \
    --conv_type=GCNConv \
    --conv_layers=2 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=ablation_studies/ \
    --combined_score_thresh=990 \
    --ablation_subset=cnvp \
    --nr_node_features=1 \
    --file_ending=AblationStudy_CnvPicnic  
    
python3 main_80_20.py \
    --seed=42 \
    --num_epochs=150 \
    --batch_size=128 \
    --num_workers=8 \
    --test_ratio=0.2 \
    --dropout=0.1 \
    --model=GraphGraph \
    --conv_type=GCNConv \
    --conv_layers=2 \
    --global_pooling=max \
    --version=v1 \
    --download=n \
    --process=n \
    --processed_path=../data/processed/ \
    --logging_path=ablation_studies/ \
    --combined_score_thresh=990 \
    --ablation_subset=mut \
    --nr_node_features=1 \
    --file_ending=AblationStudy_Mut    