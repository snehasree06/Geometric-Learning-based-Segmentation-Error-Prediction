MODEL='CNN_GNN_MLP' 
TASK='regression'
MODEL_PATCH='CGM_smoothL1_withtanh_new'
BASE_PATH='/basepath'
BATCH_SIZE=1

NUM_EPOCHS=100
DEVICE='cuda'
DATASET_TYPE='bony_labyrinth_dataset'
# NC=2

MESH_DIR=${BASE_PATH}'/GL_preprocessed_data/graph_objects/'
SIGNED_DISTANCES_DIR=${BASE_PATH}'/GL_preprocessed_data/signed_distances/'
CT_PATCHES_DIR=${BASE_PATH}'/GL_preprocessed_data/ct_patches/'
TRIANGLES_DIR=${BASE_PATH}'/GL_preprocessed_data/triangles_smooth/'


USE_PRETRAIN_ENCODER=false
ENCODER='CNN'
PRETRAINED_MODEL_DIR=${BASE_PATH}'/GL_experiments/experiments/bony_labyrinth_dataset/Pretraining/'${ENCODER}'/lr_0.001_lr_sched_cosS_optim_Adadelta_bs_128/best_model.pt'

LR_SCHEDULER='cosS'  # noS, expS, cosS, StepLR
PROCESSOR='nodeformer'   #spline, GAT
LR=0.001
OPTIMIZER='AdamW'



EXP_DIR=${BASE_PATH}'/sneha_research/GL_experiments_parotid/experiments/'${DATASET_TYPE}'/'${MODEL}'/'${TASK}'/'${MODEL_PATCH}'/lr_'${LR}'_lr_sched_'${LR_SCHEDULER}'_optim_'${OPTIMIZER}'_bs_'${BATCH_SIZE}'_'${ENCODER}'_'${PROCESSOR}

if $USE_PRETRAIN_ENCODER
then
    echo python train_regression.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --mesh-dir ${MESH_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --use_pretrain_encoder ${USE_PRETRAIN_ENCODER} --pretrained-model-dir ${PRETRAINED_MODEL_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR}  --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER} 
    python train_regression.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --mesh-dir ${MESH_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --use_pretrain_encoder ${USE_PRETRAIN_ENCODER} --pretrained-model-dir ${PRETRAINED_MODEL_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR} --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER} 
else
    echo python train_regression.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --mesh-dir ${MESH_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR}  --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER} 
    python train_regression.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --mesh-dir ${MESH_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR} --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER} 
fi

