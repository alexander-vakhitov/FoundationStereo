
import os, sys
import glob

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")
from omegaconf import OmegaConf
from tqdm import tqdm
from core.utils.utils import InputPadder
from Utils import *
import pandas
from pathlib import Path
from core.foundation_stereo import *
import cv2

if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--left_file_dir",
        required=True,
        type=str,
        help="Directory containing left camera images",
    )
    parser.add_argument(
        "--right_file_dir",
        required=True,
        type=str,
        help="Directory containing right camera images",
    )

    parser.add_argument(
        "--intrinsic_file",
        default=f"{code_dir}/../assets/K.txt",
        type=str,
        help="camera intrinsic matrix and baseline file",
    )
    parser.add_argument(
        "--ckpt_dir",
        default=f"{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth",
        type=str,
        help="pretrained model path",
    )
    parser.add_argument(
        "--out_dir",
        default=f"{code_dir}/../output/",
        type=str,
        help="the directory to save results",
    )
    parser.add_argument(
        "--ply_out_dir",
        default="",
        type=str,
        help="the directory to save resulting point clouds",
    )
    parser.add_argument(
        "--out_vis_dir",
        default="",
        type=str,
        help="the directory to save resulting visualizations",
    )
    parser.add_argument(
        "--scale",
        default=1,
        type=float,
        help="downsize the image by scale, must be <=1",
    )
    parser.add_argument(
        "--hiera",
        default=0,
        type=int,
        help="hierarchical inference (only needed for high-resolution images (>1K))",
    )
    parser.add_argument(
        "--z_far", default=20, type=float, help="max depth to clip in point cloud"
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of flow-field updates during forward pass",
    )
    parser.add_argument("--get_pc", type=int, default=1, help="save point cloud output")
    parser.add_argument(
        "--remove_invisible",
        default=1,
        type=int,
        help="remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable",
    )
    parser.add_argument(
        "--denoise_cloud",
        type=int,
        default=1,
        help="whether to denoise the point cloud",
    )
    parser.add_argument(
        "--denoise_nb_points",
        type=int,
        default=5,
        help="number of points to consider for radius outlier removal",
    )
    parser.add_argument(
        "--denoise_radius",
        type=float,
        default=0.2,
        help="radius to use for outlier removal",
    )
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt["model"])

    model.cuda()
    model.eval()

    # Find matching image pairs
#    logging.info(f"Finding image pairs in {args.left_dir} and {args.right_dir}")
#    image_pairs = find_image_pairs(args.left_dir, args.right_dir)
#    logging.info(f"Found {len(image_pairs)} image pairs")

    if args.intrinsic_file:
        with open(args.intrinsic_file, "r") as f:
            lines = f.readlines()
            K = (
                np.array(list(map(float, lines[0].rstrip().split())))
                .astype(np.float32)
                .reshape(3, 3)
            )
            baseline = float(lines[1])
            
    left_file_dir = Path(args.left_file_dir)
    left_timestamps = []
    for child in left_file_dir.iterdir():
        if child.suffix == '.jpg':
            left_timestamps.append(int(child.stem))
    right_file_dir = Path(args.right_file_dir)
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(exist_ok=True)
    is_ply_output = len(args.ply_out_dir) > 0
    if is_ply_output:
        ply_output_dir = Path(args.ply_out_dir)
        ply_output_dir.mkdir(exist_ok=True)
        
    is_save_vis = len(args.out_vis_dir) > 0
    if is_save_vis:
        Path(args.out_vis_dir).mkdir(exist_ok=True)
        
    print(f'To process: {len(left_timestamps)}')
    
    black_mask = None

    for left_ts in left_timestamps:
        left_file = left_file_dir / f'{left_ts}.jpg'
        right_file = right_file_dir / f'{left_ts}.jpg'        
        # Create output directory for this pair
        # pair_out_dir = os.path.join(args.out_dir, f"{basename}")
        # os.makedirs(pair_out_dir, exist_ok=True)
        if (f"{left_ts}.npy" in os.listdir(args.out_dir)):
            logging.info(f"Skipping {left_ts}, already processed")
            continue

        img0 = imageio.imread(left_file)
        img1 = imageio.imread(right_file)
        if (len(img0.shape)) == 2:
            h, w = img0.shape
            img0 = np.tile(img0.reshape(h, w, 1), (1, 1, 3))
            img1 = np.tile(img1.reshape(h, w, 1), (1, 1, 3))
        scale = args.scale
        assert scale <= 1, "scale must be <=1"
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
        H, W = img0.shape[:2]
        img0_ori = img0.copy()

        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.cuda.amp.autocast(True):
            if not args.hiera:
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(
                    img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5
                )
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)
        
        if black_mask is None:
            black_mask = (img0_ori[:, :, 0] == 0)
            kernel = np.ones((20,20),np.uint8)
            dil_mask = cv2.dilate(black_mask.astype(np.uint8) * 255,kernel,iterations = 1)
            black_mask = dil_mask>0
        print(f'Black mask {np.sum(black_mask)}')
        disp[black_mask] = np.inf
        
        vis = vis_disparity(disp)
        vis = np.concatenate([img0_ori, vis], axis=1)
        
        if args.remove_invisible:
            yy, xx = np.meshgrid(
                np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij"
            )
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        K[:2] *= scale
        depth = K[0, 0] * baseline / disp
        
        np.save(Path(args.out_dir) / f"{left_ts}.npy", depth)
        
                        
        if is_save_vis:
            imageio.imwrite(Path(args.out_vis_dir) / f"{left_ts}.png", vis)

        if args.get_pc:
            # np.save(f"{args.out_dir}/{basename}", depth)
            xyz_map = depth2xyzmap(depth, K)
            # save xyz map
            # np.save(f"{args.out_dir}/{basename}_xyz.npy", xyz_map)
            pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
            keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (
                np.asarray(pcd.points)[:, 2] <= args.z_far
            )
            keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)

            if args.denoise_cloud:
                # logging.info("[Optional step] denoise point cloud...")
                cl, ind = pcd.remove_radius_outlier(
                    nb_points=args.denoise_nb_points, radius=args.denoise_radius
                )
                inlier_cloud = pcd.select_by_index(ind)
                o3d.io.write_point_cloud(
                    Path(args.ply_out_dir) / f"{left_ts}.ply", inlier_cloud
                )
            else:
                o3d.io.write_point_cloud(Path(args.ply_out_dir) / f"{left_ts}.ply", pcd)


