import os
import time
import torch
import cv2
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from CT import CTGeometry, DGR
from utils.general_utils import safe_state
import uuid
import torch.nn.functional as F
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from params import ModelParams, PipelineParams, OptimizationParams
from utils.utils import quick_evaluation, fast_volume_reconstruction, total_variation_loss, evaluation_3d
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians = DGR(dataset.sh_degree)
    scene = CTGeometry(dataset, gaussians, mode='lact')
    gaussians.training_ct_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, max_iter), desc="Training progress")
    first_iter += 1
    save_flag = False
    for iteration in tqdm(range(first_iter, max_iter + 1)):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        reconstruct_volume, impact_shape, std = fast_volume_reconstruction(gaussians, gaussians.volume_shape)
        st = time.time()
        projs = gaussians.Fan_ray_trafo(reconstruct_volume)
        # print(time.time() - st)
        worldspace_points = gaussians.get_xyz
        radii = gaussians.max_radii2D
        Ll1 = l1_loss(projs[:, :projs.shape[1]//2], gaussians.gt_projs[:, :projs.shape[1]//2])
        volume_tv_loss = total_variation_loss(reconstruct_volume)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(projs, gaussians.gt_projs.cuda())) + volume_tv_loss
        # loss = Ll1
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(projs, gaussians.gt_projs.cuda()))
        loss.backward()
        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 1000 == 0:
                if not save_flag:
                    tb_writer = prepare_output_and_logger(dataset)
                    scene.model_path = dataset.model_path
                    save_flag = True
                    # save reconstruct_volume with numpy
                np.save(os.path.join(dataset.model_path, "reconstruct_volume_%s.npy"%iteration), reconstruct_volume.cpu().detach().numpy())
            if iteration % 10 == 0 or iteration == 1:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                # progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
                progress_bar.update(10)
                print('Iteration:', iteration, 'Gaussian count', gaussians.get_xyz.shape[0], 'Box Size', impact_shape.cpu().numpy().tolist(), 'loss', format(loss.item(), '.2f'),  end=', ')
                quick_evaluation(reconstruct_volume.detach(), gaussians.gt_image)
            if iteration % 100 == 0 or iteration == 1:
                evaluation_3d(reconstruct_volume, gaussians.gt_image)
            if iteration == max_iter:
                progress_bar.close()

            # # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                if not save_flag:
                    tb_writer = prepare_output_and_logger(dataset)
                    scene.model_path = dataset.model_path
                    save_flag = True
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if True:
                # Keep track of max radii in image-space for pruning
                visibility_filter = torch.ones(gaussians.get_xyz.shape[0],dtype = torch.bool)
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(worldspace_points, visibility_filter)
                if iteration >= 100 and iteration % 100 == 0:
                    gaussians.densify_and_prune_ct(opt.densify_grad_threshold, gaussians.scene_extent)
                
                if True:
                    sample_img = reconstruct_volume[8].cpu().numpy()
                    # save sample_img as png to 'vis' directory
                    if not os.path.exists(gaussians.vis_dir):
                        os.makedirs(gaussians.vis_dir)
                    resp = cv2.imwrite(os.path.join(gaussians.vis_dir, "sample_%s.png"%iteration), (sample_img*255).astype(np.uint8))
                    
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity_ct()
            # Optimizer step
            if iteration < max_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # # Create Tensorboard writer
    # tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : CTGeometry, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.set_device(0)
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[4_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[4_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    max_iter = 10000
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
    
