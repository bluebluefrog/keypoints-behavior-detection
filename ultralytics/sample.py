# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
import math
import cv2


class PosePredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'pose'

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
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
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

                # pdb.set_trace()
                if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                        len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2):
                    print('person: ', idx, ' fall down')

                    path = self.batch[0]
                    img_path = path[i] if isinstance(path, list) else path
                    start_x = output[idx][:4].cpu().numpy()[0].round()
                    start_y = output[idx][:4].cpu().numpy()[1].round()
                    # pdb.set_trace()
                    cv2.putText(orig_img, 'Person Fell down', (int(start_x), int(start_y)), 0, 1, [255, 255, 0],
                                thickness=3, lineType=cv2.LINE_AA)
                    # cv2.putText(orig_img, 'Person Fell down', (10, 10), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)
                # results.append(
                #     Results(orig_img=orig_img,
                #             path=img_path,
                #             names=self.model.names,
                #             boxes=pred[:, :6] )#,
                #             #keypoints=pred_kpts)
                #             )

            # pdb.set_trace()
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(
                Results(orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        # boxes=pred[:, :6],
                        keypoints=pred_kpts))
        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO to predict objects in an image or video."""
    model = cfg.model or 'pose_detection/yolov8m-pose.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    source = 'rtsp://root:zhanghm@@169.254.62.148/axis-media/media.amp?camera=1&videocodec=h264&resolution=1280x720'
    # source = './fall.png'
    args = dict(model=model, source=source, show=True, save=False, boxes=False)

    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()