# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops

import cv2
import math

class PosePredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a pose model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.pose import PosePredictor

        args = dict(model='yolov8n-pose.pt', source=ASSETS)
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
        ```
    """



    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes PosePredictor, sets task to 'pose' and logs a warning for using 'mps' as device."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'pose'
        if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
            LOGGER.warning("WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                           'See https://github.com/ultralytics/ultralytics/issues/4031.')

    def postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))


        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), * self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)

            for idx in range(pred.shape[0]):
                # if pred.shape[0] > 0:
                output = pred
                # idx = i
                left_shoulder_y = pred_kpts[idx][5][1]  # output[idx][23]
                left_shoulder_x = pred_kpts[idx][5][0]  # output[idx][22]
                right_shoulder_y = pred_kpts[idx][6][1]  # output[idx][26]jjj

                left_body_y = pred_kpts[idx][11][1]  # output[idx][41]
                left_body_x = pred_kpts[idx][11][0]  # output[idx][40]
                right_body_y = pred_kpts[idx][12][1]  # output[idx][44]

                len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))

                left_foot_y = pred_kpts[idx][15][1]  # output[idx][53]
                right_foot_y = pred_kpts[idx][16][1]  # output[idx][56]

                if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                        len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2):
                    print('person: ', idx, ' fall down')

                    path = self.batch[0]
                    img_path = path[i] if isinstance(path, list) else path
                    start_x = output[idx][:4].cpu().numpy()[0].round()
                    start_y = output[idx][:4].cpu().numpy()[1].round()
                    # pdb.set_trace()
                    cv2.putText(orig_img, 'Person Fall Detected', (int(start_x), int(start_y)+30), 0, 1, [255, 255, 0],
                                thickness=3, lineType=cv2.LINE_AA)

            # pdb.set_trace()

            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            img_path = self.batch[0][i]
            pred_kpts[0,:,]

            # part = ["鼻子"
            #     , "左眼"
            #     , "右眼"
            #     , "左脖子"
            #     , "右脖子"
            #     , "左二头"
            #     , "右二头"
            #     , "左手"
            #     , "右手"
            #     , "无变化2"
            #     , "无变化1"
            #     , "左大腿"
            #     , "右大腿"
            #     , "左大腿到小腿"
            #     , "右大腿到小腿"
            #     , "左小腿"
            #     , "右小腿"]
            #
            # part2 = ["nose"
            #     , "left eye"
            #     , "right eye"
            #     , "left neck"
            #     , "right neck"
            #     , "Left Second Head"
            #     , "right second head"
            #     , "left hand"
            #     , "right hand"
            #     , "No change 2"
            #     , "No change 1"
            #     , "left thigh"
            #     , "right thigh"
            #     , "Left thigh to calf"
            #     , "right thigh to calf"
            #     , "left calf"
            #     , "right calf"]

            # note
            # pred_kpts_part = np.column_stack((pred_kpts[0], part))
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=None))
        return results
