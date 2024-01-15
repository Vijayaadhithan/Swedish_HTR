import argparse
import logging
import random
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from natsort import os_sorted
from tqdm import tqdm

from core.preprocess import preprocess_datasets
from core.setup import setup_cfg
from run import Predictor
from utils.image_utils import load_image_array_from_path, save_image_array_to_path
from utils.logging_utils import get_logger_name
from utils.tempdir import OptionalTemporaryDirectory

logger = logging.getLogger(get_logger_name())


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval of prediction of model using visualizer")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs="+", help="optional args to change", default=[])

    io_args = parser.add_argument_group("IO")
    # io_args.add_argument("-t", "--train", help="Train input folder/file",
    #                         nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-o", "--output", help="Output folder", type=str, required=True)

    tmp_args = parser.add_argument_group("tmp files")
    tmp_args.add_argument("--tmp_dir", help="Temp files folder", type=str, default=None)
    tmp_args.add_argument("--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")

    parser.add_argument("--sorted", action="store_true", help="Sorted iteration")
    parser.add_argument("--save", nargs="?", const="all", default=None, help="Save images instead of displaying")

    args = parser.parse_args()

    return args


_keypress_result = None


def keypress(event):
    global _keypress_result
    # print('press', event.key)
    if event.key in ["q", "escape"]:
        sys.exit()
    if event.key in [" ", "right"]:
        _keypress_result = "forward"
        return
    if event.key in ["backspace", "left"]:
        _keypress_result = "back"
        return
    if event.key in ["e", "delete"]:
        _keypress_result = "delete"
        return
    if event.key in ["w"]:
        _keypress_result = "bad"
        return


def on_close(event):
    sys.exit()


def main(args) -> None:
    """
    Currently running the validation set and showing the ground truth and the prediction side by side

    Args:
        args (argparse.Namespace): arguments for where to find the images
    """
    if args.save and not args.output:
        raise ValueError("Cannot run saving when there is not save location given (--output)")

    # Setup config
    cfg = setup_cfg(args)

    with OptionalTemporaryDirectory(name=args.tmp_dir, cleanup=not args.keep_tmp_dir) as tmp_dir:
        preprocess_datasets(cfg, None, args.input, tmp_dir, save_image_locations=False)
        predictor = Predictor(cfg=cfg)

        # train_loader = DatasetCatalog.get("train")
        val_loader = DatasetCatalog.get("val")
        metadata = MetadataCatalog.get("val")
        # print(metadata)

        @lru_cache(maxsize=10)
        def load_image(filename):
            image = load_image_array_from_path(filename, mode="color")
            if image is None:
                raise TypeError(f"Image {filename} is None, loading failed")
            return image

        @lru_cache(maxsize=10)
        def load_sem_seg(filename):
            sem_seg_gt = load_image_array_from_path(filename, mode="grayscale")
            if sem_seg_gt is None:
                raise TypeError(f"Image {filename} is None, loading failed")
            return sem_seg_gt

        @lru_cache(maxsize=10)
        def create_gt_visualization(image_filename, sem_seg_filename):
            image = load_image(image_filename)
            sem_seg_gt = load_sem_seg(sem_seg_filename)
            vis_im_gt = Visualizer(image[..., ::-1].copy(), metadata=metadata, scale=1)
            vis_im_gt = vis_im_gt.draw_sem_seg(sem_seg_gt, alpha=0.4)
            return vis_im_gt.get_image()

        @lru_cache(maxsize=10)
        def create_pred_visualization(image_filename):
            image = load_image(image_filename)
            logger.info(f"Predict: {image_filename}")
            outputs = predictor(image)
            pred = torch.argmax(outputs[0]["sem_seg"], dim=-3).to("cpu")
            # outputs["panoptic_seg"] = (outputs["panoptic_seg"][0].to("cpu"),
            #                            outputs["panoptic_seg"][1])
            vis_im = Visualizer(image[..., ::-1].copy(), metadata=metadata, scale=1)

            vis_im = vis_im.draw_sem_seg(pred, alpha=0.4)
            return vis_im.get_image()

        fig, axes = plt.subplots(1, 2)
        fig.tight_layout()
        fig.canvas.mpl_connect("key_press_event", keypress)
        fig.canvas.mpl_connect("close_event", on_close)
        axes[0].axis("off")
        axes[1].axis("off")

        fig_manager = None
        if not args.save:
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()

        if args.save:
            pbar = tqdm(total=len(val_loader), desc="Saving")

        # for i, inputs in enumerate(np.random.choice(val_loader, 3)):
        if args.sorted:
            loader = os_sorted(val_loader, key=lambda x: x["file_name"])
        else:
            loader = val_loader
            random.shuffle(val_loader)

        bad_results = np.zeros(len(loader), dtype=bool)
        delete_results = np.zeros(len(loader), dtype=bool)

        i = 0
        while 0 <= i < len(loader):
            inputs = loader[i]

            vis_gt = create_gt_visualization(inputs["file_name"], inputs["sem_seg_file_name"])
            vis_pred = create_pred_visualization(inputs["file_name"])

            # pano_gt = torch.IntTensor(rgb2id(cv2.imread(inputs["pan_seg_file_name"], cv2.IMREAD_COLOR)))
            # print(inputs["segments_info"])

            # vis_im = vis_im.draw_panoptic_seg(outputs["panoptic_seg"][0], outputs["panoptic_seg"][1])
            # vis_im_gt = vis_im_gt.draw_panoptic_seg(pano_gt, [item | {"isthing": True} for item in inputs["segments_info"]])
            if not args.save:
                fig_manager.window.setWindowTitle(inputs["file_name"])

            # HACK Just remove the previous axes, I can't find how to resize the image otherwise
            axes[0].clear()
            axes[1].clear()
            axes[0].axis("off")
            axes[1].axis("off")

            axes[0].imshow(vis_pred)
            axes[1].imshow(vis_gt)

            if args.save is not None:
                # TODO Move saving to separate function
                pbar.update(1)
                if args.save not in ["all", "both", "pred", "gt"]:
                    raise ValueError(f"{args.save} is not a valid save mode")

                output_dir = Path(args.output)
                if not output_dir.is_dir():
                    logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
                    output_dir.mkdir(parents=True)

                if args.save in ["all", "both"]:
                    save_path = output_dir.joinpath(Path(inputs["file_name"]).stem + "_both.jpg")
                    # Save to 4K res
                    fig.set_size_inches(16, 9)
                    fig.savefig(str(save_path), dpi=240)
                if args.save in ["all", "pred"]:
                    save_path = output_dir.joinpath(Path(inputs["file_name"]).stem + "_pred.jpg")
                    save_image_array_to_path(save_path, vis_pred[..., ::-1])
                if args.save in ["all", "gt"]:
                    save_path = output_dir.joinpath(Path(inputs["file_name"]).stem + "_gt.jpg")
                    save_image_array_to_path(save_path, vis_gt[..., ::-1])

                i += 1
                continue

            if delete_results[i]:
                fig.suptitle("Delete")
            elif bad_results[i]:
                fig.suptitle("Bad")
            else:
                fig.suptitle("")
            # f.title(inputs["file_name"])
            global _keypress_result
            _keypress_result = None
            fig.canvas.draw()
            while _keypress_result is None:
                plt.waitforbuttonpress()
            if _keypress_result == "delete":
                # print(i+1, f"{inputs['original_file_name']}: DELETE")
                delete_results[i] = not delete_results[i]
                bad_results[i] = False
            elif _keypress_result == "bad":
                # print(i+1, f"{inputs['original_file_name']}: BAD")
                bad_results[i] = not bad_results[i]
                delete_results[i] = False
            elif _keypress_result == "forward":
                # print(i+1, f"{inputs['original_file_name']}")
                i += 1
            elif _keypress_result == "back":
                # print(i+1, f"{inputs['original_file_name']}: DELETE")
                i -= 1

        if args.save:
            pbar.close()

    if args.output and (delete_results.any() or bad_results.any()):
        output_dir = Path(args.output)
        if not output_dir.is_dir():
            logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)
        if delete_results.any():
            output_delete = output_dir.joinpath("delete.txt")
            with output_delete.open(mode="w") as f:
                for i in delete_results.nonzero()[0]:
                    path = Path(loader[i]["original_file_name"])
                    line = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve()
                    f.write(f"{line}\n")
        if bad_results.any():
            output_bad = output_dir.joinpath("bad.txt")
            with output_bad.open(mode="w") as f:
                for i in bad_results.nonzero()[0]:
                    path = Path(loader[i]["original_file_name"])
                    line = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve()
                    f.write(f"{line}\n")

        remaining_results = np.logical_not(np.logical_or(bad_results, delete_results))
        if remaining_results.any():
            output_remaining = output_dir.joinpath("correct.txt")
            with output_remaining.open(mode="w") as f:
                for i in remaining_results.nonzero()[0]:
                    path = Path(loader[i]["original_file_name"])
                    line = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve()
                    f.write(f"{line}\n")


if __name__ == "__main__":
    args = get_arguments()
    main(args)
