import os
import warnings
import os.path as osp
import gin
import tensorflow as tf
from absl import app, flags, logging
import numpy as np
from matplotlib import pyplot as pl
import torchvision.transforms as tvf
from scipy.spatial.transform import Rotation as R

from dycheck import core, geometry

from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images, _resize_pil_image
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

# dycheck configs
flags.DEFINE_multi_string(
    "gin_configs",
    "dycheck_test/iphone_dataset.gin",
    "Gin config files.",
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
FLAGS = flags.FLAGS

DATA_ROOT = "/data/dycheck"

# dust3r configs
model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
device = 'cuda'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def process_torch_imgs(torch_imgs, size, square_ok=False, verbose=True):

    imgs = []
    for torch_img in torch_imgs:
        # torch to pil image
        img = tvf.ToPILImage()(torch_img)

        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - resize with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    return imgs


def prepare_dataset():
    tf.config.experimental.set_visible_devices([], "GPU")

    core.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs,
        bindings=(FLAGS.gin_bindings or [])
        + [
            "Config.engine_cls=None",
            "Config.model_cls=None",
            "SEQUENCE='paper-windmill'",
            f"iPhoneParser.data_root='{DATA_ROOT}'",
            f"iPhoneDatasetFromAllFrames.split='train'",
            f"iPhoneDatasetFromAllFrames.training=False",
            f"iPhoneDatasetFromAllFrames.bkgd_points_batch_size=0",
        ],
        skip_unknown=True,
    )

    config_str = gin.config_str()
    logging.info(f"*** Configuration:\n{config_str}")

    config = core.Config()

    dataset = config.dataset_cls()
    return dataset

def load_dycheck(dataset, idx):
    sample = dataset[idx]
    rgb = sample['rgb']
    camera = dataset.cameras[idx]
    intrin = camera.intrin
    extrin = camera.extrin
    return rgb, intrin, extrin

def calculate_pose_diff(gt_rel_pose, pred_rel_pose):

    # calculate translation diff
    gt_t = gt_rel_pose[:3, 3]
    pred_t = pred_rel_pose[:3, 3]
    t_diff = np.linalg.norm(gt_t - pred_t)

    # calculate rotation diff
    gt_r = gt_rel_pose[:3, :3]
    pred_r = pred_rel_pose[:3, :3]
    rotation_diff_matrix = gt_r.T @ pred_r

    # Convert rotation matrix to rotation vector (in radians)
    rotation_diff_rad = R.from_matrix(rotation_diff_matrix).as_rotvec()
    # norm of the rotation vector is the angle of rotation
    rotation_diff_rad = np.linalg.norm(rotation_diff_rad)

    # Convert angle to degrees
    rotation_diff_deg = np.rad2deg(rotation_diff_rad)

    return t_diff, rotation_diff_deg

def main(_):

    # dycheck dataset
    dataset = prepare_dataset()
    print('Dataset length:', len(dataset))

    # load dust3r model
    model = load_model(model_path, device)

    # load images
    # indices = [0, 1, 2]
    # indices = [1, 2]
    indices = list(range(10))

    samples = [(rgb, intrin, extrin) for rgb, intrin, extrin in [load_dycheck(dataset, idx) for idx in indices]]
    images = process_torch_imgs([sample[0] for sample in samples], size=512)

    # run dust3r matching
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # align the predictions
    # scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals() # list: N x 3
    poses = scene.get_im_poses() # tensor: [N, 4, 4]
    pts3d = scene.get_pts3d() # list: N x [H, W, 3]
    confidence_masks = scene.get_masks() # list: N x [H, W]

    # gt rel pose for all sequences
    extrins = [sample[2] for sample in samples]
    gt_rel_poses = [extrins[i] @ np.linalg.inv(extrins[i+1]) for i in range(len(extrins)-1)]

    # pred rel pose for all sequences
    pred_rel_poses = [poses[i].detach().cpu().numpy() @ np.linalg.inv(poses[i+1].detach().cpu().numpy()) for i in range(len(poses)-1)]

    # measure pose diff between gt and pred
    pose_diffs = [calculate_pose_diff(gt_rel_pose, pred_rel_pose) for gt_rel_pose, pred_rel_pose in zip(gt_rel_poses, pred_rel_poses)]
    print('Pose diffs:', pose_diffs)

    # # measure pose diff
    # pred_pose1 = poses[0].cpu().numpy()
    # pred_pose2 = poses[1].cpu().numpy()
    # gt_rel_pose = extrin2 @ np.linalg.inv(extrin1)
    # pred_rel_pose = pred_pose1 @ np.linalg.inv(pred_pose2)
    # pose_diff = np.linalg.norm(gt_rel_pose - pred_rel_pose)
    # print('gt_pose1:', extrin1)
    # print('gt_pose2:', extrin2)
    # print('pred_pose1:', pred_pose1)
    # print('pred_pose2:', pred_pose2)
    # print('gt_rel_pose:', gt_rel_pose)
    # print('pred_rel_pose:', pred_rel_pose)
    # print('pose_diff:', pose_diff)

    # # find 2D-2D matches between the two images
    # pts2d_list, pts3d_list = [], []
    # for i in range(2):
    #     conf_i = confidence_masks[i].cpu().numpy()
    #     pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
    #     pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    # reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    # print(f'found {num_matches} matches')
    # matches_im1 = pts2d_list[1][reciprocal_in_P2]
    # matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # # visualize a few matches
    # n_viz = 10
    # match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    # viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    # H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    # img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img = np.concatenate((img0, img1), axis=1)
    # pl.figure()
    # pl.imshow(img)
    # cmap = pl.get_cmap('jet')
    # for i in range(n_viz):
    #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
    #     pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    # # pl.show(block=True)
    # pl.savefig('matches_dycheck.png')


if __name__ == "__main__":
    app.run(main)


