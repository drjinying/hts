from torch import nn
from torchvision.models.detection import transform

class DefrDetTransform(transform.GeneralizedRCNNTransform):
    """
    Transformations for input image and target (box pairs).
    Reference: Fred Zhang <frederic.zhang@anu.edu.au>

    Arguments(Positional):
        min_size(int)
        max_size(int)
        image_mean(list[float] or tuple[float])
        image_std(list[float] or tuple[float])

    Refer to torchvision.models.detection for more details
    """
    def __init__(self, image_size, image_mean, image_std):
        self.image_size = image_size
        super().__init__(0, 0, image_mean, image_std, size_divisible=1)

    def resize(self, image, detection):
        """
        Override method to resize box pairs
        """
        h, w = image.shape[-2:]
        image = nn.functional.interpolate(
            image[None], size=self.image_size,
            mode='bilinear', align_corners=False
        )[0]

        # min_size = float(min(image.shape[-2:]))
        # max_size = float(max(image.shape[-2:]))
        # scale_factor = min(
        #     self.min_size[0] / min_size,
        #     self.max_size / max_size
        # )

        # image = nn.functional.interpolate(
        #     image[None], scale_factor=scale_factor,
        #     mode='bilinear', align_corners=False,
        #     recompute_scale_factor=True
        # )[0]

        if detection is None:
            return image, detection

        detection['box_h'] = transform.resize_boxes(detection['box_h'][None],
            (h, w), image.shape[-2:])[0]
        detection['box_o'] = transform.resize_boxes(detection['box_o'][None],
            (h, w), image.shape[-2:])[0]

        return image, detection

    def postprocess(self, results, image_shapes, original_image_sizes):
        if self.training:
            loss = results.pop()

        for pred, im_s, o_im_s in zip(results, image_shapes, original_image_sizes):
            boxes_h, boxes_o = pred['boxes_h'], pred['boxes_o']
            boxes_h = transform.resize_boxes(boxes_h[None], im_s, o_im_s)[0]
            boxes_o = transform.resize_boxes(boxes_o[None], im_s, o_im_s)[0]
            pred['boxes_h'], pred['boxes_o'] = boxes_h, boxes_o

        if self.training:
            results.append(loss)

        return results