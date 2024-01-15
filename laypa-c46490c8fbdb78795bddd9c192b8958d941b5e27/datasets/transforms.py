# Modified from P2PaLA

import argparse

import cv2
import detectron2.data.transforms as T
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates

# REVIEW Check if there is a benefit for using scipy instead of the standard torchvision


class ResizeTransform(T.Transform):
    """
    Resize image Using cv2
    """

    def __init__(self, height: int, width: int, new_height: int, new_width: int) -> None:
        """
        Resize image Using cv2

        Args:
            height (int): initial height
            width (int): initial width
            new_height (int): height after resizing
            new_width (int): width after resizing
        """
        super().__init__()
        self.height = height
        self.width = width
        self.new_height = new_height
        self.new_width = new_width

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize Image

        Args:
            img (np.ndarray): image array HxWxC

        Returns:
            np.ndarray: resized images
        """
        img = img.astype(np.float32)
        old_height, old_width, channels = img.shape
        assert (old_height, old_width) == (self.height, self.width), "Input dims do not match specified dims"

        res_image = cv2.resize(img, (self.new_width, self.new_height), interpolation=cv2.INTER_LINEAR)

        return res_image

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Resize coords

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            np.ndarray: resized coordinates
        """
        coords[:, 0] = coords[:, 0] * (self.new_width * 1.0 / self.width)
        coords[:, 1] = coords[:, 1] * (self.new_height * 1.0 / self.height)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Resize segmentation (using nearest neighbor interpolation)

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: resized segmentation
        """
        old_height, old_width = segmentation.shape
        assert (old_height, old_width) == (self.height, self.width), "Input dims do not match specified dims"

        res_segmentation = cv2.resize(segmentation, (self.new_width, self.new_height), interpolation=cv2.INTER_NEAREST)

        return res_segmentation

    def inverse(self) -> T.Transform:
        """
        Inverse the resize by flipping old and new height
        """
        return ResizeTransform(self.new_height, self.new_width, self.height, self.width)


class HFlipTransform(T.Transform):
    """
    Perform horizontal flip. Taken from fvcore
    """

    def __init__(self, width: int):
        """
        Perform horizontal flip

        Args:
            width (int): image width
        """
        super().__init__()
        self.width = width

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        img = img.astype(np.float32)
        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=1)
        else:
            return np.flip(img, axis=-2)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords

    def inverse(self) -> T.Transform:
        """
        The inverse is to flip again
        """
        return self


class VFlipTransform(T.Transform):
    """
    Perform vertical flip
    """

    def __init__(self, height: int):
        """
        Perform vertical flip

        Args:
            height (int): image height
        """
        super().__init__()
        self.height = height

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        img = img.astype(np.float32)
        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=0)
        else:
            return np.flip(img, axis=-3)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 1] = self.height - coords[:, 1]
        return coords

    def inverse(self) -> T.Transform:
        """
        The inverse is to flip again
        """
        return self


class WarpFieldTransform(T.Transform):
    """
    Apply a warp field (optical flow) to an image
    """

    def __init__(self, warpfield: np.ndarray) -> None:
        """
        Apply a warp field (optical flow) to an image

        Args:
            warpfield (np.ndarray): flow of pixels in the image
        """
        super().__init__()
        self.warpfield = warpfield

    @staticmethod
    def generate_grid(img: np.ndarray, warpfield: np.ndarray) -> np.ndarray:
        """
        Generate the new locations of pixels based on the offset warpfield

        Args:
            img (np.ndarray): of shape HxW or HxWxC
            warpfield (np.ndarray): HxW warpfield with movement per pixel

        Raises             :
        NotImplementedError: Only support for HxW and HxWxC right now

        Returns:
            np.ndarray: new pixel coordinates
        """
        if img.ndim == 2:
            x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")
            indices = np.reshape(x + warpfield[..., 0], (-1, 1)), np.reshape(y + warpfield[..., 1], (-1, 1))
            return np.asarray(indices)
        elif img.ndim == 3:
            x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]), indexing="ij")
            indices = (
                np.reshape(x + warpfield[..., 0, None], (-1, 1)),
                np.reshape(y + warpfield[..., 1, None], (-1, 1)),
                np.reshape(z, (-1, 1)),
            )
            return np.asarray(indices)
        else:
            raise NotImplementedError("No support for multi dimensions (NxHxWxC) right now")

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Warp an image with a specified warpfield, using spline interpolation

        Args:
            img (np.ndarray): image array HxWxC

        Returns:
            np.ndarray: warped image
        """
        img = img.astype(np.float32)
        indices = self.generate_grid(img, self.warpfield)
        sampled_img = map_coordinates(img, indices, order=1, mode="constant", cval=0).reshape(img.shape)

        return sampled_img

    def apply_coords(self, coords: np.ndarray):
        """
        Coords moving might be possible but might move some out of bounds
        """
        # TODO This may be possible, and seems necessary for the instance predictions
        # raise NotImplementedError
        # IDEA self.recompute_boxes in dataset_mapper, with moving polygon values
        # HACK Currently just returning original coordinates
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Warp a segmentation with a specified warpfield, using spline interpolation with order 0

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: warped segmentation
        """
        indices = self.generate_grid(segmentation, self.warpfield)
        # cval=0 means background cval=255 means ignored
        sampled_segmentation = map_coordinates(segmentation, indices, order=0, mode="constant", cval=0).reshape(
            segmentation.shape
        )

        return sampled_segmentation

    def inverse(self) -> T.Transform:
        """
        No inverse for a warp is possible since information is lost
        """
        raise NotImplementedError


class AffineTransform(T.Transform):
    """
    Apply an affine transformation to an image
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Apply an affine transformation to an image

        Args:
            matrix (np.ndarray): affine matrix applied to the pixels in image
        """
        super().__init__()
        self.matrix = matrix

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply an affine transformation to the image

        Args:
            img (np.ndarray): image array HxWxC

        Raises:
            NotImplementedError: wrong dimensions of image

        Returns:
            np.ndarray: transformed image
        """
        img = img.astype(np.float32)
        if img.ndim == 2:
            return affine_transform(img, self.matrix, order=1, mode="constant", cval=0)
        elif img.ndim == 3:
            transformed_img = np.empty_like(img)
            for i in range(img.shape[-1]):  # HxWxC
                transformed_img[..., i] = affine_transform(img[..., i], self.matrix, order=1, mode="constant", cval=0)
            return transformed_img
        else:
            raise NotImplementedError("No support for multi dimensions (NxHxWxC) right now")

    def apply_coords(self, coords: np.ndarray):
        """
        Apply affine transformation to coordinates

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            np.ndarray: transformed coordinates
        """
        coords = coords.astype(np.float32)
        return cv2.transform(coords[:, None, :], self.matrix)[:, 0, :2]

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply an affine transformation to the segmentation

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: transformed segmentation
        """
        # cval=0 means background cval=255 means ignored
        return affine_transform(segmentation, self.matrix, order=0, mode="constant", cval=0)

    def inverse(self) -> T.Transform:
        """
        Inverse not always possible, since information may be lost
        """
        raise NotImplementedError


class GrayscaleTransform(T.Transform):
    """
    Convert an image to grayscale
    """

    def __init__(self, image_format: str = "RGB") -> None:
        """
        Convert an image to grayscale

        Args:
            image_format (str, optional): type of image format. Defaults to "RGB".
        """
        super().__init__()

        self.rgb_weights = np.asarray([0.299, 0.587, 0.114]).astype(np.float32)

        self.image_format = image_format

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Turn to grayscale by applying weights to the color image and than tile to get 3 channels again

        Args:
            img (np.ndarray): image array HxWxC

        Returns:
            np.ndarray: grayscale version of image
        """
        img = img.astype(np.float32)
        if self.image_format == "BGR":
            grayscale = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        else:
            grayscale = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        return grayscale

    def apply_coords(self, coords: np.ndarray):
        """
        Color transform does not affect coords

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Color transform does not affect segmentation

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: original segmentation
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        No inverse possible. Grayscale cannot return to color
        """
        raise NotImplementedError


class GaussianFilterTransform(T.Transform):
    """
    Apply one or more gaussian filters
    """

    def __init__(self, sigma: float = 4, order: int = 0, iterations: int = 1) -> None:
        """
        Apply one or more gaussian filters

        Args:
            sigma (float, optional): Gaussian deviation. Defaults to 4.
            order (int, optional): order of gaussian derivative. Defaults to 0.
            iterations (int, optional): times the kernel is applied. Defaults to 1.
        """
        self.sigma = sigma
        self.order = order
        self.iterations = iterations

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply gaussian filters to the original image

        Args:
            img (np.ndarray): image array HxWxC

        Returns:
            np.ndarray: blurred image
        """
        img = img.astype(np.float32)
        transformed_img = img.copy()
        for _ in range(self.iterations):
            for i in range(img.shape[-1]):  # HxWxC
                transformed_img[..., i] = gaussian_filter(transformed_img[..., i], sigma=self.sigma, order=self.order)
        return transformed_img

    def apply_coords(self, coords: np.ndarray):
        """
        Blurring should not affect the coordinates

        Args:
            coords (np.ndarray): loating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Blurring should not affect the segmentation

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: original segmentation
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        No inverse of blurring is possible since information is lost
        """
        raise NotImplementedError


class BlendTransform(T.Transform):
    """
    Transforms pixel colors with PIL enhance functions.
    """

    def __init__(self, src_image: np.ndarray, src_weight: float, dst_weight: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (ndarray): Input image is blended with this image.
                The two images must have the same shape, range, channel order
                and dtype.
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self.src_image = src_image
        self.src_weight = src_weight
        self.dst_weight = dst_weight

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply blend transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.
        Returns:
            ndarray: blended image(s).
        """
        img = img.astype(np.float32)
        return self.src_weight * self.src_image + self.dst_weight * img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        The inverse is a no-op.
        """
        raise NotImplementedError


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Testing the image augmentation and transformations")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input file", required=True, type=str)

    args = parser.parse_args()
    return args


def test(args) -> None:
    from pathlib import Path

    import cv2
    from PIL import Image

    input_path = Path(args.input)

    if not input_path.is_file():
        raise FileNotFoundError(f"Image {input_path} not found")

    print(f"Loading image {input_path}")
    image = cv2.imread(str(input_path))[..., ::-1]
    print(image.dtype)

    affine = AffineTransform(np.eye(3))
    output_image = affine.apply_image(image)

    im = Image.fromarray(image)
    im.show("Original")

    im = Image.fromarray(output_image.round().clip(0, 255).astype(np.uint8))
    im.show("Transformed")


if __name__ == "__main__":
    args = get_arguments()
    test(args)
