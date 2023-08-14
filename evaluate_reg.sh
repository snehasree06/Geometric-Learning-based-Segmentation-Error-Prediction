MODEL='CNN_GNN_MLP' 
TASK='regression'
MODEL_PATCH='CGM_smoothL1_withtanh_new'
BASE_PATH='/base_path/'
BATCH_SIZE=1

NUM_EPOCHS=1
DEVICE='cuda'
DATASET_TYPE='bony_labyrinth_dataset'
# NC=2

MESH_DIR=${BASE_PATH}'/GL_preprocessed_data/graph_objects/'
GS_CLASSES_DIR=${BASE_PATH}'/GL_preprocessed_data/signed_classes/'
SIGNED_DISTANCES_DIR=${BASE_PATH}'/GL_preprocessed_data/signed_distances/'
CT_PATCHES_DIR=${BASE_PATH}'/GL_preprocessed_data/ct_patches/'
TRIANGLES_DIR=${BASE_PATH}'/GL_preprocessed_data/triangles_smooth/'

ALL_SIGNED_CLASSES_DIR=${BASE_PATH}'/GL_preprocessed_data/all_signed_classes/all_signed_classes.pt'


USE_PRETRAIN_ENCODER=false
ENCODER='CNN'
PRETRAINED_MODEL_DIR=${BASE_PATH}'/GL_experiments/experiments/bony_labyrinth_dataset/Pretraining/'${ENCODER}'/lr_0.001_lr_sched_cosS_optim_Adadelta_bs_128/best_model.pt'

LR_SCHEDULER='cosS'  # noS, expS, cosS, StepLR
PROCESSOR='nodeformer'   #spline, GAT
LR=0.001
OPTIMIZER='AdamW'


EXP_DIR=${BASE_PATH}'/GL_experiments/experiments/'${DATASET_TYPE}'/'${MODEL}'/'${TASK}'/'${MODEL_PATCH}'/lr_'${LR}'_lr_sched_'${LR_SCHEDULER}'_optim_'${OPTIMIZER}'_bs_'${BATCH_SIZE}'_'${ENCODER}'_'${PROCESSOR}
CHECKPOINT=${EXP_DIR}'/best_model.pt'


if $USE_PRETRAIN_ENCODER
then

    echo python evaluate_reg.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --checkpoint ${CHECKPOINT} --mesh-dir ${MESH_DIR} --gs-classes-dir ${GS_CLASSES_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --all-signed-classes-dir ${ALL_SIGNED_CLASSES_DIR} --pretrained-model-dir ${PRETRAINED_MODEL_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR}  --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER}  --use_pretrain_encoder ${USE_PRETRAIN_ENCODER}
    python evaluate_reg.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --checkpoint ${CHECKPOINT} --mesh-dir ${MESH_DIR} --gs-classes-dir ${GS_CLASSES_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --all-signed-classes-dir ${ALL_SIGNED_CLASSES_DIR}  --pretrained-model-dir ${PRETRAINED_MODEL_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR} --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER} --use_pretrain_encoder ${USE_PRETRAIN_ENCODER}


else
    echo python evaluate_reg.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --checkpoint ${CHECKPOINT} --mesh-dir ${MESH_DIR} --gs-classes-dir ${GS_CLASSES_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --all-signed-classes-dir ${ALL_SIGNED_CLASSES_DIR}  --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR}  --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER}  --use_pretrain_encoder ${USE_PRETRAIN_ENCODER}
    python evaluate_reg.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --checkpoint ${CHECKPOINT} --mesh-dir ${MESH_DIR} --gs-classes-dir ${GS_CLASSES_DIR} --signed-distances-dir ${SIGNED_DISTANCES_DIR} --ct-patches-dir ${CT_PATCHES_DIR} --triangles-dir ${TRIANGLES_DIR} --all-signed-classes-dir ${ALL_SIGNED_CLASSES_DIR} --lr_sched ${LR_SCHEDULER} --lr $LR --optimizer ${OPTIMIZER} --processor ${PROCESSOR} --preds-output-dir ${EXP_DIR}'/soft_preds' --encoder ${ENCODER} --use_pretrain_encoder ${USE_PRETRAIN_ENCODER}

fi



