import argparse
import os,sys
import shutil
import torch
import time
import numpy as np

from test_3D_util_mirror import test_all_case_without_score
# from test_3D_util_rt import test_all_case_without_score

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_path',  type=str,
    default='/input/images/cbct',
    help='(必选) CBCT 输入目录，容器里应挂载到 /input/images/cbct'
)
parser.add_argument(
    '--output_path', type=str,
    default='/output/images/oral-pharyngeal-segmentation',
    help='(必选) 分割结果输出目录，容器里应挂载到 /output/images/oral-pharyngeal-segmentation'
)
parser.add_argument(
    '--metadata_path', type=str,
    default='/output/metadata',
    help='(可选) 实例级元数据目录，容器里应挂载到 /output/metadata'
)
# parser.add_argument('--exp', type=str,
#                     default='BraTS2019/Interpolation_Consistency_Training_25', help='experiment_name'ValidationSe)
parser.add_argument('--model', type=str,
                    default='nnUNetv2', help='model_name')
parser.add_argument('--TTA', type=bool,
                    default=False, help='TTA')

def Inference(FLAGS):
    num_classes = 10 ############ 2 ##################
    # # test_save_path = "/output/images/oral-pharyngeal-segmentation"      # Path 2/4
    # # json_path = r'/output/results.json'                                 # Path 3/4
    # # 把分割结果存到刚建好的 output/images 文件夹
    # test_save_path = "/media/ps/PS10T/changkai/MICCAI2025ToothFairtTask1/output/images"
    # # 把指标 JSON 存到 output 根目录
    # json_path      = "/media/ps/PS10T/changkai/MICCAI2025ToothFairtTask1/output/results.json"
    input_dir = FLAGS.input_path
    test_save_path = FLAGS.output_path
    metadata_dir = FLAGS.metadata_path

    if os.path.exists(test_save_path):
        # 只删除目录内容，不删除目录本身
        for filename in os.listdir(test_save_path):
            file_path = os.path.join(test_save_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(test_save_path, exist_ok=True)
    print(f"Files in {FLAGS.input_path}: {os.listdir(FLAGS.input_path)}")
    # net = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=43).cuda()
    
    ################### 1 ###############
    # /workspace

    # save_model_list = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']        # Path 4/4
    # model_weights = [1, 1, 1, 1, 1]
    #
    # model_weights = np.array(model_weights) / np.sum(model_weights)
    # save_model_list = [os.path.join('/workspace/weight', fold, 'checkpoint_best.pth') for fold in save_model_list]
    # save_model_list = [
    #     '/media/ps/PS10T/changkai/v2nnunet/nnUNet-master/nnUnetFrame'
    #     '/nnUNet_results/Dataset300_ToothFairy'
    #     '/nnUNetTrainerNoDA__nnUNetPlans__3d_fullres'
    #     '/fold_all/checkpoint_best.pth'
    #     ]
    save_model_list = [
            os.path.join('/opt/app/weights', 'checkpoint_best.pth')
    ]
    model_weights = np.array([1.0])

    # net.load_state_dict(torch.load(save_mode_path_0)['network_weights'])
    # net.eval()
    print("init weight from {}".format(save_model_list))
    print("init weight by weight {}".format(model_weights))
    ################# 2 #################
    avg_metric = test_all_case_without_score(save_model_list, model_weights, model_name=FLAGS.model,base_dir=input_dir, num_classes=num_classes,
                               patch_size=(160, 112, 128), json_path=None,test_save_path=test_save_path, TTA_flag=FLAGS.TTA)  ############ 3 preprocess json #########
    print(f"Files in {test_save_path}: {os.path.exists(test_save_path)}, {os.listdir(test_save_path)}")
    return avg_metric


if __name__ == '__main__':
    T1 = time.time()
    FLAGS = parser.parse_args()
    torch.set_num_threads(8)

    print('Whether GPU?',torch.cuda.is_available(),torch.cuda.get_device_name(0))

    metric = Inference(FLAGS)
    T2 = time.time()
    print('Inference Time: %s s' % (T2 - T1))
    print(metric)
