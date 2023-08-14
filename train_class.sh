MODEL='CNN_GNN_MLP' 
TASK='classification'
MODEL_PATCH='CGM'
BASE_PATH='/basepath'
BATCH_SIZE=1

NUM_EPOCHS=100
DEVICE='cuda'
DATASET_TYPE='bony_labyrinth_dataset'
# NC=2

MESH_DIR=${BASE_PATH}'/GL_preprocessed_data_parotid/graph_objects/'
GS_CLASSES_DIR=${BASE_PATH}'/GL_preprocessed_data_parotid/signed_classes/'
SIGNED_DISTANCES_DIR=${BASE_PATH}'/GL_preprocessed_data_parotid/signed_distances/'
CT_PATCHES_DIR=${BASE_PATH}'/GL_preprocessed_data_parotid/ct_patches/'
TRIANGLES_DIR=${BASE_PATH}'/GL_preprocessed_data_parotid/triangles_smooth/'

# ALL_SIGNED_CLASSES_DIR=${BASE_PATH}'/GL_preprocessed_data_parotid/all_signed_classes/all_signed_classes.pt'


USE_PRETRAIN_ENCODER=false
ENCODER='CNN'
PRETRAINED_MODEL_DIR=${BASE_PATH}'/GL_experiments_parotid/experiments/bony_labyrinth_dataset/Pretraining/'${ENCODER}'/lr_0.001_lr_sched_cosS_optim_Adadelta_bs_128/best_model.pt'

LR_SCHEDULER='cosS'  # noS, expS, cosS, StepLR
PROCESSOR='graphsage'   #spline, GAT
LR=0.001
OPTIMIZER='AdamW'


EXP_DIR=${BASE_PATH}'/GL_experiments_parotid/experiments/'${DATASET_TYPE}'/'${MODEL}'/'${TASK}'/'${MODEL_PATCH}'/lr_'${LR}'_lr_sched_'${LR_SCHEDULER}'_optim_'${OPTIMIZER}'_bs_'${BATCH_SIZE}'_'${ENCODER}'_'${PROCESSOR}

if $USE_PRETRAIN_ENCODER
then
    echo python train_class.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --mesh-dir ${MESH_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --use_pretrain_encoder ${USE_PRETRAIN_ENCODER} --pretrained-model-dir ${PRETRAINED_MODEL_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR}  --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER} 
    python train_class.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --mesh-dir ${MESH_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --use_pretrain_encoder ${USE_PRETRAIN_ENCODER} --pretrained-model-dir ${PRETRAINED_MODEL_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR} --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER} 
else
    echo python train_class.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --mesh-dir ${MESH_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR}  --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER} 
    python train_class.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --mesh-dir ${MESH_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR} --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER} 
fi



