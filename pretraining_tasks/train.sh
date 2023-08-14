MODEL='Pretraining' 
TASK='VAE'
BASE_PATH='basepath/'
BATCH_SIZE=128
NUM_EPOCHS=1000
DEVICE='cuda'
DATASET_TYPE='bony_labyrinth_dataset'
# NC=2

MESH_DIR=${BASE_PATH}'/GL_preprocessed_data/graph_objects/GS'
CT_VOLUMES_DIR=${BASE_PATH}'/GL_preprocessed_data/CT_vol'
VERTEX_NORMALS_DIR=${BASE_PATH}'/GL_preprocessed_data/vertex_normals_smooth/GS'
LR=0.001
LR_SCHEDULER='cosS'
OPTIMIZER='AdamW'
PRETRAIN_FLAG='vae_recon'  #mask_ae, vn

EXP_DIR=${BASE_PATH}'/GL_experiments/experiments/'${DATASET_TYPE}'/'${MODEL}'/'${TASK}'/lr_'${LR}'_lr_sched_'${LR_SCHEDULER}'_optim_'${OPTIMIZER}'_bs_'${BATCH_SIZE}

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --mesh-dir ${MESH_DIR} --ct-volumes-dir ${CT_VOLUMES_DIR} --vertex-normals-dir ${VERTEX_NORMALS_DIR}  --lr ${LR}  --lr_sched ${LR_SCHEDULER} --pretrain_flag ${PRETRAIN_FLAG} --optimizer ${OPTIMIZER}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --mesh-dir ${MESH_DIR} --ct-volumes-dir ${CT_VOLUMES_DIR} --vertex-normals-dir ${VERTEX_NORMALS_DIR} --lr ${LR}   --lr_sched ${LR_SCHEDULER}  --pretrain_flag ${PRETRAIN_FLAG} --optimizer ${OPTIMIZER}
