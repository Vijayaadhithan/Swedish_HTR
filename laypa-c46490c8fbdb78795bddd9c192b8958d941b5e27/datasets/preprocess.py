import argparse
import json
import logging
import os
import sys
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import imagesize
import numpy as np
from tqdm import tqdm

from datasets.augmentations import ResizeLongestEdge, ResizeScaling, ResizeShortestEdge

# from multiprocessing.pool import ThreadPool as Pool

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xml_converter import XMLConverter
from utils.copy_utils import copy_mode
from utils.image_utils import load_image_array_from_path, save_image_array_to_path
from utils.input_utils import clean_input_paths, get_file_paths
from utils.logging_utils import get_logger_name
from utils.path_utils import check_path_accessible, image_path_to_xml_path


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        parents=[Preprocess.get_parser(), XMLConverter.get_parser()],
        description="Preprocessing an annotated dataset of documents with pageXML",
    )

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str)
    io_args.add_argument("-o", "--output", help="Output folder", required=True, type=str)
    args = parser.parse_args()
    return args


class Preprocess:
    """
    Used for almost all preprocessing steps to prepare datasets to be used by the training loop
    """

    def __init__(
        self,
        input_paths=None,
        output_dir=None,
        resize_mode="none",
        resize_sampling="choice",
        scaling=0.5,
        min_size=[1024],
        max_size=2048,
        xml_converter=None,
        disable_check=False,
        overwrite=False,
    ) -> None:
        """
        Used for almost all preprocessing steps to prepare datasets to be used by the training loop

        Args:
            input_paths (Sequence[Path], optional): the used input dir/files to generate the dataset. Defaults to None.
            output_dir (Path, optional): the destination dir of the generated dataset. Defaults to None.
            resize_mode (str, optional): resize images before saving. Defaults to none.
            resize_sampling (str, optional): sample type used when resizing. Defaults to "choice".
            min_size (list, optional): when resizing, the length the shortest edge is resized to. Defaults to [1024].
            max_size (int, optional): when resizing, the max length a side may have. Defaults to 2048.
            xml_to_image (XMLImage, optional): Class for turning pageXML to an image format. Defaults to None.
            disable_check (bool, optional): flag to turn of filesystem checks, useful if run was already successful once. Defaults to False.
            overwrite (bool, optional): flag to force overwrite of images. Defaults to False.

        Raises:
            TypeError: Did not provide a XMLImage object to convert from XML to image
            ValueError: If resize mode is choice must provide the min size with 2 or more values
            ValueError: If resize mode is range must provide the min size with 2 values
            NotImplementedError: resize mode given is not
        """

        self.logger = logging.getLogger(get_logger_name())

        self.input_paths: Optional[Sequence[Path]] = None
        if input_paths is not None:
            self.set_input_paths(input_paths)

        self.output_dir: Optional[Path] = None
        if output_dir is not None:
            self.set_output_dir(output_dir)

        if not isinstance(xml_converter, XMLConverter):
            raise TypeError(f"Must provide conversion from xml to image. Current type is {type(xml_converter)}, not XMLImage")

        self.xml_converter = xml_converter

        self.disable_check = disable_check
        self.overwrite = overwrite

        # Formats found here: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#imread
        self.image_formats = [
            ".bmp",
            ".dib",
            ".jpeg",
            ".jpg",
            ".jpe",
            ".jp2",
            ".png",
            ".webp",
            ".pbm",
            ".pgm",
            ".ppm",
            ".pxm",
            ".pnm",
            ".pfm",
            ".sr",
            ".ras",
            ".tiff",
            ".tif",
            ".exr",
            ".hdr",
            ".pic",
        ]

        self.resize_mode = resize_mode
        self.scaling = scaling
        self.resize_sampling = resize_sampling
        self.min_size = min_size
        self.max_size = max_size

        if self.resize_sampling == "choice":
            if len(self.min_size) < 1:
                raise ValueError("Must specify at least one choice when using the choice option.")
        elif self.resize_sampling == "range":
            if len(self.min_size) != 2:
                raise ValueError("Must have two int to set the range")
        else:
            raise NotImplementedError('Only "choice" and "range" are accepted values')

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        """
        Return argparser that has the arguments required for the preprocessing.

        Returns:
            argparse.ArgumentParser: the argparser for preprocessing
        """
        parser = argparse.ArgumentParser(add_help=False)
        pre_process_args = parser.add_argument_group("preprocessing")

        pre_process_args.add_argument("--resize", action="store_true", help="Resize input images")
        pre_process_args.add_argument(
            "--resize_sampling",
            default="choice",
            choices=["range", "choice"],
            type=str,
            help="How to select the size when resizing",
        )
        pre_process_args.add_argument(
            "--scaling", default=0.5, type=float, help="Scaling factor while resizing with mode scaling"
        )

        pre_process_args.add_argument("--min_size", default=[1024], nargs="*", type=int, help="Min resize shape")
        pre_process_args.add_argument("--max_size", default=2048, type=int, help="Max resize shape")

        pre_process_args.add_argument("--disable_check", action="store_true", help="Don't check if all images exist")

        pre_process_args.add_argument("--overwrite", action="store_true", help="Overwrite the images and label masks")

        return parser

    def set_input_paths(self, input_paths: str | Path | Sequence[str | Path]) -> None:
        """
        Setter for image paths, also cleans them to be a list of Paths

        Args:
            input_paths (str | Path | Sequence[str | Path]): path(s) from which to extract the images

        Raises:
            FileNotFoundError: input path not found on the filesystem
            PermissionError: input path not accessible
        """
        input_paths = clean_input_paths(input_paths)

        all_input_paths = []

        for input_path in input_paths:
            if not input_path.exists():
                raise FileNotFoundError(f"Input ({input_path}) is not found")

            if not os.access(path=input_path, mode=os.R_OK):
                raise PermissionError(f"No access to {input_path} for read operations")

            input_path = input_path.resolve()
            all_input_paths.append(input_path)

        self.input_paths = all_input_paths

    def get_input_paths(self) -> Optional[Sequence[Path]]:
        """
        Getter of the input paths

        Returns:
            Optional[Sequence[Path]]: path(s) from which to extract the images
        """
        return self.input_paths

    def set_output_dir(self, output_dir: str | Path) -> None:
        """
        Setter of output dir, turn string to path. And resolve full path

        Args:
            output_dir (str | Path): output path of the processed images
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.is_dir():
            self.logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)

        self.output_dir = output_dir.resolve()

    def get_output_dir(self) -> Optional[Path]:
        """
        Getter of the output dir

        Returns:
            Optional[Path]: output path of the processed images
        """
        return self.output_dir

    @staticmethod
    def check_paths_exists(paths: list[Path]) -> None:
        """
        Check if all paths given exist and are readable

        Args:
            paths (list[Path]): paths to be checked
        """
        all(check_path_accessible(path) for path in paths)

    def resize_image_old(self, image: np.ndarray) -> np.ndarray:
        """
        Old version of image resizing, uses the multiple of 256 and smaller than maxsize * minsize

        Args:
            image (np.ndarray): image array HxWxC

        Returns:
            np.ndarray: resized image
        """
        old_height, old_width, channels = image.shape
        counter = 1
        height = np.ceil(old_height / (256 * counter)) * 256
        width = np.ceil(old_width / (256 * counter)) * 256
        while height * width > self.min_size[-1] * self.max_size:
            height = np.ceil(old_height / (256 * counter)) * 256
            width = np.ceil(old_width / (256 * counter)) * 256
            counter += 1

        res_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

        return res_image

    def sample_edge_length(self):
        """
        Samples the shortest edge for resizing

        Raises:
            NotImplementedError: not choice or sample for resize mode

        Returns:
            int: shortest edge length
        """
        if self.resize_sampling == "range":
            short_edge_length = np.random.randint(self.min_size[0], self.min_size[1] + 1)
        elif self.resize_sampling == "choice":
            short_edge_length = np.random.choice(self.min_size)
        else:
            raise NotImplementedError('Only "choice" and "range" are accepted values')

        return short_edge_length

    def get_output_shape(self, old_height, old_width) -> tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.

        Returns:
            tuple[int, int]: height and width
        """
        if self.resize_mode == "none":
            return old_height, old_width
        elif self.resize_mode in ["shortest_edge", "longest_edge"]:
            edge_length = self.sample_edge_length()
            if edge_length == 0:
                return old_height, old_width
            if self.resize_mode == "shortest_edge":
                return ResizeShortestEdge.get_output_shape(old_height, old_width, edge_length, self.max_size)
            else:
                return ResizeLongestEdge.get_output_shape(old_height, old_width, edge_length, self.max_size)
        elif self.resize_mode == "scaling":
            return ResizeScaling.get_output_shape(old_height, old_width, self.scaling, self.max_size)
        else:
            raise NotImplementedError(f"{self.resize_mode} is not a known resize mode")

    def resize_image(self, image: np.ndarray, image_shape: Optional[tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image. If image size is given resize to given value. Otherwise sample the shortest edge and resize to the calculated output shape

        Args:
            image (np.ndarray): image array HxWxC
            image_shape (Optional[tuple[int, int]], optional): desired output shape. Defaults to None.

        Returns:
            np.ndarray: resized image
        """
        old_height, old_width, channels = image.shape

        if image_shape is not None:
            height, width = image_shape
        else:
            height, width = self.get_output_shape(old_height, old_width)

        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

        return resized_image

    def save_image(
        self,
        image_path: Path,
        image_stem: str,
        image_shape: tuple[int, int],
    ):
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")
        image_dir = self.output_dir.joinpath("original")
        image_dir.mkdir(parents=True, exist_ok=True)
        if self.resize_mode == "none":
            out_image_path = image_dir.joinpath(image_path.name)
        else:
            out_image_path = image_dir.joinpath(image_stem + ".png")

        def _save_image_helper():
            """
            Quick helper function for opening->resizing->saving
            """

            if self.resize_mode == "none":
                copy_mode(image_path, out_image_path, mode="symlink")
            else:
                image = load_image_array_from_path(image_path)

                if image is None:
                    raise TypeError(f"Image {image_path} is None, loading failed")
                image = self.resize_image(image, image_shape=image_shape)
                save_image_array_to_path(out_image_path, image)

        # TODO Maybe replace with guard clauses
        # Check if image already exist and if it doesn't need resizing
        if self.overwrite or not out_image_path.exists():
            _save_image_helper()
        else:
            out_image_shape = imagesize.get(out_image_path)[::-1]
            if out_image_shape != image_shape:
                _save_image_helper()

        return str(out_image_path.relative_to(self.output_dir))

    def save_mask(
        self,
        xml_path: Path,
        image_stem: str,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ):
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")
        mask_dir = self.output_dir.joinpath("sem_seg")
        mask_dir.mkdir(parents=True, exist_ok=True)
        out_mask_path = mask_dir.joinpath(image_stem + ".png")

        def _save_mask_helper():
            """
            Quick helper function for opening->converting to image->saving
            """
            mask = self.xml_converter.to_image(xml_path, original_image_shape=original_image_shape, image_shape=image_shape)

            save_image_array_to_path(out_mask_path, mask)

        # Check if image already exist and if it doesn't need resizing
        if self.overwrite or not out_mask_path.exists():
            _save_mask_helper()
        else:
            out_mask_shape = imagesize.get(out_mask_path)[::-1]
            if out_mask_shape != image_shape:
                _save_mask_helper()

        return str(out_mask_path.relative_to(self.output_dir))

    def save_instances(
        self,
        xml_path: Path,
        image_stem: str,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ):
        if self.output_dir is None:
            raise ValueError("Cannot run when the output dir is not set")
        instances_dir = self.output_dir.joinpath("instances")
        instances_dir.mkdir(parents=True, exist_ok=True)
        out_instances_path = instances_dir.joinpath(image_stem + ".json")
        out_instances_size_path = instances_dir.joinpath(image_stem + ".txt")

        def _save_instances_helper():
            """
            Quick helper function for opening->rescaling->saving
            """
            instances = self.xml_converter.to_json(xml_path, original_image_shape=original_image_shape, image_shape=image_shape)
            json_instances = {"image_size": image_shape, "annotations": instances}
            with open(out_instances_path, "w") as f:
                json.dump(json_instances, f)
            with open(out_instances_size_path, "w") as f:
                f.write(f"{image_shape[0]},{image_shape[1]}")

        # Check if image already exist and if it doesn't need resizing
        if self.overwrite or not out_instances_path.exists() or not out_instances_size_path.exists():
            _save_instances_helper()
        else:
            with open(out_instances_size_path, "r") as f:
                out_intstances_shape = tuple(int(x) for x in f.read().strip().split(","))
            if out_intstances_shape != image_shape:
                _save_instances_helper()

        return str(out_instances_path.relative_to(self.output_dir))

    def save_panos(
        self,
        xml_path: Path,
        image_stem: str,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ):
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")
        panos_dir = self.output_dir.joinpath("panos")
        panos_dir.mkdir(parents=True, exist_ok=True)
        out_pano_path = panos_dir.joinpath(image_stem + ".png")
        out_segments_info_path = panos_dir.joinpath(image_stem + ".json")

        def _save_panos_helper():
            """
            Quick helper function for opening->rescaling->saving
            """
            pano, segments_info = self.xml_converter.to_pano(
                xml_path, original_image_shape=original_image_shape, image_shape=image_shape
            )

            save_image_array_to_path(out_pano_path, pano)

            json_pano = {"image_size": image_shape, "segments_info": segments_info}
            with open(out_segments_info_path, "w") as f:
                json.dump(json_pano, f)

        # Check if image already exist and if it doesn't need resizing
        if self.overwrite or not out_pano_path.exists():
            _save_panos_helper()
        else:
            out_mask_shape = imagesize.get(out_pano_path)[::-1]
            if out_mask_shape != image_shape:
                _save_panos_helper()

        return str(out_pano_path.relative_to(self.output_dir)), str(out_segments_info_path.relative_to(self.output_dir))

    def process_single_file(self, image_path: Path) -> dict:
        """
        Process a single image and pageXML to be used during training

        Args:
            image_path (Path): Path to input image

        Raises:
            TypeError: Cannot return if output dir is not set

        Returns:
            dict: Preprocessing results
        """
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")

        image_stem = image_path.stem
        xml_path = image_path_to_xml_path(image_path, self.disable_check)
        # xml_path = self.input_dir.joinpath("page", image_stem + '.xml')

        original_image_shape = tuple(int(value) for value in imagesize.get(image_path)[::-1])
        if self.resize_mode != "none":
            image_shape = self.get_output_shape(old_height=original_image_shape[0], old_width=original_image_shape[1])
        else:
            image_shape = original_image_shape

        results = {}
        results["output_sizes"] = image_shape
        results["original_image_paths"] = str(image_path)

        out_image_path = self.save_image(image_path, image_stem, image_shape)
        results["image_paths"] = out_image_path

        out_mask_path = self.save_mask(xml_path, image_stem, original_image_shape, image_shape)
        results["sem_seg_paths"] = out_mask_path

        out_instances_path = self.save_instances(xml_path, image_stem, original_image_shape, image_shape)
        results["instances_paths"] = out_instances_path

        out_pano_path, out_segments_info_path = self.save_panos(xml_path, image_stem, original_image_shape, image_shape)
        results["pano_paths"] = out_pano_path
        results["segments_info_paths"] = out_segments_info_path

        return results

    def run(self) -> None:
        """
        Run preprocessing on all images currently on input paths, save to output dir

        Raises:
            TypeError: Input paths must be set
            TypeError: Output dir must be set
            ValueError: Must find at least one image in all input paths
            ValueError: Must find at least one pageXML in all input paths
        """
        if self.input_paths is None:
            raise TypeError("Cannot run when the input path is None")
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")

        image_paths = get_file_paths(self.input_paths, self.image_formats, self.disable_check)
        xml_paths = [image_path_to_xml_path(image_path, self.disable_check) for image_path in image_paths]

        if len(image_paths) == 0:
            raise ValueError(f"No images found when checking input ({self.input_paths})")

        if len(xml_paths) == 0:
            raise ValueError(f"No pagexml found when checking input  ({self.input_paths})")

        if not self.disable_check:
            self.check_paths_exists(image_paths)
            self.check_paths_exists(xml_paths)

        # Single thread
        # results = []
        # for image_path in tqdm(image_paths, desc="Preprocessing"):
        #     results.append(self.process_single_file(image_path))

        # Multithread
        with Pool(os.cpu_count()) as pool:
            results = list(
                tqdm(
                    iterable=pool.imap_unordered(self.process_single_file, image_paths),
                    total=len(image_paths),
                    desc="Preprocessing",
                )
            )

        # Assuming all key are the same make one dict
        results = {"data": list_of_dict_to_dict_of_list(results), "classes": self.xml_converter.get_regions()}

        output_path = self.output_dir.joinpath("info.json")
        with open(output_path, "w") as f:
            json.dump(results, f)


def list_of_dict_to_dict_of_list(input_list: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Convert a list of dicts into dict of lists. All dicts much have the same keys. The output number of dicts matches the length of the list

    Args:
        input_list (list[dict[str, Any]]): list of dicts

    Returns:
        dict[str, list[Any]]: dict of lists
    """
    output_dict = {key: [item[key] for item in input_list] for key in input_list[0].keys()}
    return output_dict


def main(args) -> None:
    xml_converter = XMLConverter(
        mode=args.mode,
        line_width=args.line_width,
        regions=args.regions,
        merge_regions=args.merge_regions,
        region_type=args.region_type,
    )
    process = Preprocess(
        input_paths=args.input,
        output_dir=args.output,
        resize_mode=args.resize_mode,
        resize_sampling=args.resize_sampling,
        min_size=args.min_size,
        max_size=args.max_size,
        xml_converter=xml_converter,
        disable_check=args.disable_check,
        overwrite=args.overwrite,
    )
    process.run()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
