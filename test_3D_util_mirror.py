import math
import os
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from torch.cuda.amp import autocast
from skimage import filters,measure
import json
from scipy.ndimage import gaussian_filter
from networks.net_factory_3d import net_factory_3d

def compute_steps_for_sliding_window(patch_size, image_size, step_size=0.5):
    """
    从 nnUNetv2 抄过来的滑动窗口计算方式
    """
    steps = []
    for dim in range(len(patch_size)):
        target_step = patch_size[dim] * step_size
        if image_size[dim] <= patch_size[dim]:
            steps_here = [0]
        else:
            num_steps = int(np.ceil((image_size[dim] - patch_size[dim]) / target_step)) + 1
            actual_step_size = (image_size[dim] - patch_size[dim]) / max(num_steps - 1, 1)
            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps)]
        steps.append(steps_here)
    return steps


def resample_sitk_to_spacing(itk_img, out_spacing=(0.3, 0.3, 0.3),
                             is_label=False, force_size=None):
    """
    重采样到指定 spacing。
    如果 force_size 不为 None，则强制输出尺寸一致。
    """
    original_spacing = np.array(itk_img.GetSpacing(), dtype=np.float64)
    original_size = np.array(itk_img.GetSize(), dtype=np.int64)

    out_spacing = np.array(out_spacing, dtype=np.float64)
    if force_size is not None:
        # 🚨 转成 python list[int]，避免 VectorUInt32 报错
        out_size = [int(x) for x in force_size]
    else:
        out_size = np.round(original_size * (original_spacing / out_spacing)).astype(np.int64).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(out_spacing.tolist()))
    resampler.SetSize(out_size)
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    resampler.SetOutputDirection(itk_img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    # 影像用线性插值，标签用最近邻插值
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

    return resampler.Execute(itk_img)


def test_all_case_without_score(save_model_list, model_weights, model_name, base_dir,
                                num_classes=10, patch_size=(160, 112, 128),
                                json_path=None, test_save_path=None, TTA_flag=False):
    """
    - 输入影像先重采样到 plans.json 的 fullres spacing
    - 预测
    - 预测结果重采样回原始 spacing，并强制和原图 size 一致
    - 保存保证和原图尺寸完全一致
    """
    print("Testing begin")
    path = os.listdir(base_dir)
    _case_results = []

    # 来自 312 plans.json 的目标 spacing
    target_spacing = (0.30000001192092896, 0.2999992370605469, 0.30000001192092896)
    transpose_forward = (1, 0, 2)
    transpose_backward = (1, 0, 2)

    for image_path in path:
        print('Processing', image_path)
        itk_img = sitk.ReadImage(os.path.join(base_dir, image_path))

        # 1) 重采样到 fullres spacing
        itk_img_rs = resample_sitk_to_spacing(itk_img, out_spacing=target_spacing, is_label=False)
        img_np = sitk.GetArrayFromImage(itk_img_rs).astype(np.float32)

        # 2) 转置（如训练时有 transpose）
        img_np = np.transpose(img_np, transpose_forward)

        # 3) 归一化
        img_np = normalized(img_np)

        # 4) 预测
        prediction_all = test_single_case_v2(
            save_model_list, model_weights, img_np, patch_size,
            num_classes=num_classes, do_mirroring=TTA_flag)
        prediction_all = postprocessing(prediction_all)

        # 5) 转置回来
        prediction_all = np.transpose(prediction_all, transpose_backward)

        # 6) numpy → sitk，几何信息先用重采样后的
        pred_itk_rs = sitk.GetImageFromArray(prediction_all.astype(np.uint8))
        pred_itk_rs.CopyInformation(itk_img_rs)

        # 7) 把预测结果重采样回原图 spacing + 原图 size
        pred_itk_final = resample_sitk_to_spacing(
            pred_itk_rs,
            out_spacing=itk_img.GetSpacing(),
            is_label=True,
            force_size=itk_img.GetSize()   # 🚨 强制和原图 size 一致
        )
        pred_itk_final.CopyInformation(itk_img)

        # 8) 保存结果
        output_filename = os.path.basename(image_path).replace('.nii.gz', '_Mask.nii.gz')
        sitk.WriteImage(pred_itk_final, os.path.join(test_save_path, output_filename))

        _case_results.append(process_case(image_path))

    save(json_path, _case_results)
    return "Testing end"



def save(_output_file,_case_results):
    with open(str(_output_file), "w") as f:
        json.dump(_case_results, f)

def process_case(case_name):
    # Load and test the image for this case


    # Write segmentation file path to result.json for this case
    return {
            "outputs": [
                dict(type="metaio_image", filename=case_name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=case_name)
            ],
            "error_messages": [],
        }

def compute_gaussian(tile_size, sigma_scale = 1. / 8, dtype=np.float16):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def normalized(image):
    """
    nnUNetv2 风格：
    1) 先按照 dataset 统计的百分位裁剪（plans.json）
    2) 然后对整幅图做 z-score（use_mask_for_norm = false）
    """
    # 312 Dataset 统计（来自 plans.json -> foreground_intensity_properties_per_channel["0"]）
    p005 = 191.0
    p995 = 3095.0

    # 百分位裁剪
    image = np.clip(image, p005, p995).astype(np.float32)

    # 全图 z-score
    mean = image.mean()
    std = image.std() + 1e-8
    image = (image - mean) / std
    return image.astype(np.float32)

def normalized_v2(image):
    mean = image.mean()
    std = image.std()
    image = (image - mean) / std
    return image.astype(np.float32)

# def normalized_v2(image):
#     mean = 1283.30859375
#     std = 503.14630126953125
#     return (image - mean) / std

def test_single_case_v2(save_model_list,model_weights, image, patch_size, num_classes=1, do_mirroring = False, use_gaussian = True):
    print(f"using TTA: {do_mirroring}")
    print("Accelerated version.")
    w, h, d = image.shape
    # image = normalized_v2(image)
    with autocast():
        with torch.no_grad():
            # if the size of image is less than patch_size, then padding it
            add_pad = False
            if w < patch_size[0]:
                w_pad = patch_size[0]-w
                add_pad = True
            else:
                w_pad = 0
            if h < patch_size[1]:
                h_pad = patch_size[1]-h
                add_pad = True
            else:
                h_pad = 0
            if d < patch_size[2]:
                d_pad = patch_size[2]-d
                add_pad = True
            else:
                d_pad = 0
            wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
            hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
            dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
            if add_pad:
                image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                       (dl_pad, dr_pad)], mode='constant', constant_values=0)
            ww, hh, dd = image.shape

            steps = compute_steps_for_sliding_window(patch_size, (ww, hh, dd), step_size=0.5)
            sx, sy, sz = [len(s) for s in steps]

            print("{}, {}, {}".format(sx, sy, sz))

            score_map_torch = torch.zeros((num_classes, ) + image.shape, dtype=torch.float16).cuda()
            cnt_torch = torch.zeros(image.shape, dtype=torch.float16).cuda()
            # print(score_map_torch.shape, score_map_torch.dtype)

            if use_gaussian:
                gaussian = compute_gaussian(patch_size)
                # make sure nothing is rounded to zero or we get division by zero :-(
                mn = gaussian.min()
                if mn == 0:
                    gaussian.clip(min=mn)
                gaussian_torch = torch.from_numpy(gaussian).cuda()

            image_torch = torch.from_numpy(image.astype(np.float16)).cuda()

            for j, save_model_path in enumerate(save_model_list):
                net = net_factory_3d(net_type='nnUNetv2', in_chns=1, class_num=10).cuda()
                # net.load_state_dict(torch.load(save_model_path)['network_weights'])
                state = torch.load(save_model_path, weights_only=False)  # 允许完整反序列化
                net.load_state_dict(state['network_weights'])

                net.eval()
                # print(f"load from {save_model_path}")
                for x in range(0, sx):
                    xs = steps[0][x]
                    for y in range(0, sy):
                        ys = steps[1][y]
                        for z in range(0, sz):
                            zs = steps[2][z]
                            # test_patch = image[xs:xs+patch_size[0],
                            #                 ys:ys+patch_size[1], zs:zs+patch_size[2]]
                            # test_patch = np.expand_dims(np.expand_dims(
                            #     test_patch, axis=0), axis=0).astype(np.float32)

                            # # y = torch.zeros([1, num_classes] + list(test_patch.shape[2:]),
                            # #                             dtype=torch.float16).cuda()
                            # test_patch = torch.from_numpy(test_patch).cuda()

                            test_patch = image_torch[xs:xs+patch_size[0],
                                            ys:ys+patch_size[1], zs:zs+patch_size[2]]
                            test_patch = test_patch[None, None, :, :, :]

                            # y = net(test_patch)[0]
                            # # y = torch.softmax(y, dim=1, dtype=torch.float16)
                            # y = y[0, :, :, :, :]
                            # y = net(test_patch)
                            # if isinstance(y, list):
                            #     y = y[0]
                            # y = y[0, :, :, :, :]

                            y = net(test_patch)
                            if isinstance(y, list):
                                y = y[0]
                            # y: [B=1, C, Dx, Dy, Dz]
                            y = torch.softmax(y, dim=1)  # 与 nnUNetv2 融合一致
                            y = y[0, :, :, :, :]
                            if use_gaussian:
                                score_map_torch[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y * gaussian_torch
                                cnt_torch[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += gaussian_torch
                            else:
                                score_map_torch[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                                cnt_torch[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1


            # score_map = score_map_torch.cpu().data.numpy()
            # cnt = cnt_torch.cpu().data.numpy()
            # score_map = score_map/np.expand_dims(cnt, axis=0)
            # label_map = np.argmax(score_map, axis=0)
            # print(score_map.shape, score_map.dtype)
            # print(label_map.shape, label_map.dtype)
            score_map_torch = score_map_torch / cnt_torch.unsqueeze(0)
            label_map_torch = torch.argmax(score_map_torch, dim=0)

            # 只传输最终的小结果到CPU
            label_map = label_map_torch.cpu().data.numpy().astype(np.uint8)

            print(f"Final result shape: {label_map.shape}, dtype: {label_map.dtype}")

            # 清理GPU内存
            del score_map_torch, cnt_torch, label_map_torch
            torch.cuda.empty_cache()

            if add_pad:
                label_map = label_map[wl_pad:wl_pad+w,
                                      hl_pad:hl_pad+h, dl_pad:dl_pad+d]

    return label_map.astype(np.uint8)


def remove_small_connected_object(npy_mask, area_least=10):
    from skimage import measure
    from skimage.morphology import label

    npy_mask[npy_mask != 0] = 1
    labeled_mask, num = label(npy_mask, return_num=True)
    print('Num of Connected Objects',num)
    if num == 2:
        print('No Postprocessing...')
        return npy_mask
    else:
        print('Postprocessing...')
        region_props = measure.regionprops(labeled_mask)

        res_mask = np.zeros_like(npy_mask)
        for i in range(1, num + 1):
            t_area = region_props[i - 1].area
            if t_area > area_least:
                res_mask[labeled_mask == i] = 1

    return res_mask

def connected_component(image):
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return image

	# 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(1, num+1)]
    area_list = [region[i-1].area for i in num_list]
    print(num_list,area_list)
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    print(num_list_sorted)
	# 去除面积较小的连通域
    if len(num_list_sorted) > 2:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[2:]:
            # label[label==i] = 0
            label[region[i-1].slice][region[i-1].image] = 0
        # num_list_sorted = num_list_sorted[:1]
    return label



def postprocessing(prediction):
    """
    标签值映射后处理：3→4, 4→5, 5→6, 6→7, 7→8, 8→9, 9→12
    使用查找表方法实现高效映射
    """
    # 标签映射
    label_map = {3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 12}

    # 创建查找表
    max_label = max(max(label_map.keys()), prediction.max())
    lookup_table = np.arange(max_label + 1)
    for old_val, new_val in label_map.items():
        lookup_table[old_val] = new_val

    # 应用映射
    result = lookup_table[prediction]

    return result.astype(prediction.dtype)


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    # ravd = abs(metric.binary.ravd(pred, gt))
    # hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # return np.array([dice, ravd, hd, asd])
    return np.array([dice])
