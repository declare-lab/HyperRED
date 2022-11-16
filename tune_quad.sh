DATA=rest15
DATA=rest16

for BS in 4 8 16 32
do
  for LR in 3e-5 5e-5 1e-4
  do
    for EPOCHS in 30 50 100
    do
      save_dir="ckpt/quad_$DATA/epochs_${EPOCHS}_bs_${BS}_lr_${LR}"
      echo $save_dir
      python q_main.py \
      --epochs $EPOCHS \
      --train_batch_size $BS \
      --bert_learning_rate $LR \
      --learning_rate $LR \
      --save_dir $save_dir \
      --data_dir data/$DATA \
      --no_value_mlp \
      --prune_topk 20 \
      --use_pair2_mlp \
      --fix_q_loss \
      --config_file q_config.yml
    done
  done
done