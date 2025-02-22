from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.utils.timer import Timer

PRINT_TIMER = False


def normalize_kp(
    kp_source,
    kp_driving,
    kp_driving_initial,
    adapt_movement_scale=False,
    use_relative_movement=False,
    use_relative_jacobian=False,
):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source["value"][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(
            kp_driving_initial["value"][0].data.cpu().numpy()
        ).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = kp_driving["value"] - kp_driving_initial["value"]
        kp_value_diff *= adapt_movement_scale
        kp_new["value"] = kp_value_diff + kp_source["value"]

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"])
            )
            kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

    return kp_new


def headpose_pred_to_degree(pred, idx_tensor):
    # device = pred.device
    # with Timer("idx_tensor", print_=PRINT_TIMER):
    # idx_tensor = [idx for idx in range(66)]
    # idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    # with Timer("softmax", print_=PRINT_TIMER):
    pred = F.softmax(pred)
    degree = torch.sum(pred * idx_tensor, 1) * 3 - 99
    return degree


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat(
        [
            torch.ones_like(pitch),
            torch.zeros_like(pitch),
            torch.zeros_like(pitch),
            torch.zeros_like(pitch),
            torch.cos(pitch),
            -torch.sin(pitch),
            torch.zeros_like(pitch),
            torch.sin(pitch),
            torch.cos(pitch),
        ],
        dim=1,
    )
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat(
        [
            torch.cos(yaw),
            torch.zeros_like(yaw),
            torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.ones_like(yaw),
            torch.zeros_like(yaw),
            -torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.cos(yaw),
        ],
        dim=1,
    )
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat(
        [
            torch.cos(roll),
            -torch.sin(roll),
            torch.zeros_like(roll),
            torch.sin(roll),
            torch.cos(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.ones_like(roll),
        ],
        dim=1,
    )
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum("bij,bjk,bkm->bim", pitch_mat, yaw_mat, roll_mat)

    return rot_mat


@torch.compile()
def keypoint_transformation(kp_canonical, he, idx_tensor, wo_exp=False):
    with Timer("get yaw pitch roll", print_=PRINT_TIMER):
        kp = kp_canonical["value"]  # (bs, k, 3)
        yaw, pitch, roll = he["yaw"], he["pitch"], he["roll"]

        # with Timer("get pitch ", print_=PRINT_TIMER):
        pitch = headpose_pred_to_degree(pitch, idx_tensor)
        # with Timer("get yaw ", print_=PRINT_TIMER):
        yaw = headpose_pred_to_degree(yaw, idx_tensor)
        # with Timer("get roll", print_=PRINT_TIMER):
        roll = headpose_pred_to_degree(roll, idx_tensor)
        if "yaw_in" in he:
            yaw = he["yaw_in"]
        if "pitch_in" in he:
            pitch = he["pitch_in"]
        if "roll_in" in he:
            roll = he["roll_in"]
    with Timer("get_rotation_matrix", print_=PRINT_TIMER):
        rot_mat = get_rotation_matrix(yaw, pitch, roll)  # (bs, 3, 3)

    t, exp = he["t"], he["exp"]
    if wo_exp:
        exp = exp * 0
    with Timer(
        "kp rotation, translation, add expression deviation", print_=PRINT_TIMER
    ):
        # keypoint rotation
        kp_rotated = torch.einsum("bmp,bkp->bkm", rot_mat, kp)

        # keypoint translation
        t[:, 0] = t[:, 0] * 0
        t[:, 2] = t[:, 2] * 0
        t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
        kp_t = kp_rotated + t

        # add expression deviation
        exp = exp.view(exp.shape[0], -1, 3)
        kp_transformed = kp_t + exp

    return {"value": kp_transformed}


def make_animation(
    source_image,
    source_semantics,
    target_semantics,
    generator,
    kp_detector,
    he_estimator,
    mapping,
    yaw_c_seq=None,
    pitch_c_seq=None,
    roll_c_seq=None,
    use_exp=True,
    use_half=False,
):
    with torch.no_grad():
        with Timer("source kp detector", print_=PRINT_TIMER):
            kp_canonical = kp_detector(source_image)
        with Timer("source mapping", print_=PRINT_TIMER):
            he_source = mapping(source_semantics)
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = (
            torch.FloatTensor(idx_tensor)
            .type_as(he_source["yaw"])
            .to(he_source["yaw"].device)
        )
        with Timer("source kp transform", print_=PRINT_TIMER):
            kp_source = keypoint_transformation(kp_canonical, he_source, idx_tensor)
        batch_size = 32  # TODO: fix this hard code
        source_image_batch = source_image.repeat(batch_size, 1, 1, 1)
        kp_source_batch = {"value": kp_source["value"].repeat(batch_size, 1, 1)}
        kp_norm_batch = {"value": []}
        predictions = []
        for frame_idx in tqdm(range(target_semantics.shape[1]), "Face Renderer:"):
            # still check the dimension
            # print(target_semantics.shape, source_semantics.shape)
            with Timer("target mapping", print_=PRINT_TIMER):
                target_semantics_frame = target_semantics[:, frame_idx]
                he_driving = mapping(target_semantics_frame)
                if yaw_c_seq is not None:
                    he_driving["yaw_in"] = yaw_c_seq[:, frame_idx]
                if pitch_c_seq is not None:
                    he_driving["pitch_in"] = pitch_c_seq[:, frame_idx]
                if roll_c_seq is not None:
                    he_driving["roll_in"] = roll_c_seq[:, frame_idx]
            with Timer("target kp transform", print_=PRINT_TIMER):
                kp_driving = keypoint_transformation(
                    kp_canonical, he_driving, idx_tensor
                )
                kp_norm = kp_driving
            kp_norm_batch["value"].append(kp_norm["value"])
            with Timer("OcclusionAwareSPADEGenerator", print_=PRINT_TIMER):
                # out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
                # print(
                #     source_image.shape, kp_source["value"].shape, kp_norm["value"].shape
                # )
                # if isinstance(generator, torch._dynamo.OptimizedModule):
                #     for k, v in out.items():
                #         # out[k] = v.detach() #detach prevent overwriting https://github.com/pytorch/pytorch/issues/104435
                #         out[k] = v + 0
                """
                source_image_new = out['prediction'].squeeze(1)
                kp_canonical_new =  kp_detector(source_image_new)
                he_source_new = he_estimator(source_image_new)
                kp_source_new = keypoint_transformation(kp_canonical_new, he_source_new, wo_exp=True)
                kp_driving_new = keypoint_transformation(kp_canonical_new, he_driving, wo_exp=True)
                out = generator(source_image_new, kp_source=kp_source_new, kp_driving=kp_driving_new)
                """
                if (frame_idx + 1) % batch_size == 0 or (
                    frame_idx + 1
                ) == target_semantics.shape[1]:
                    kp_norm_batch["value"] = torch.stack(
                        kp_norm_batch["value"], dim=1
                    ).squeeze(0)
                    if kp_norm_batch["value"].shape[0] != batch_size:  # last epoch
                        kp_source_batch["value"] = kp_source_batch["value"][
                            : kp_norm_batch["value"].shape[0], :, :
                        ]
                        source_image_batch = source_image_batch[
                            : kp_norm_batch["value"].shape[0], :, :, :
                        ]
                    out = generator(
                        source_image_batch,
                        kp_source=kp_source_batch,
                        kp_driving=kp_norm_batch,
                    )
                    if isinstance(generator, torch._dynamo.OptimizedModule):
                        for k, v in out.items():
                            # out[k] = v.detach() #detach prevent overwriting https://github.com/pytorch/pytorch/issues/104435
                            out[k] = v + 0
                    predictions.append(out["prediction"])
                    kp_norm_batch["value"] = []  # reset
        # predictions_ts = torch.stack(predictions, dim=1)
        predictions_ts = torch.cat(predictions, dim=0).unsqueeze(0)

        # predictions_ts = out["prediction"].unsqueeze(0)
    return predictions_ts


class AnimateModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        source_image = x["source_image"]
        source_semantics = x["source_semantics"]
        target_semantics = x["target_semantics"]
        yaw_c_seq = x["yaw_c_seq"]
        pitch_c_seq = x["pitch_c_seq"]
        roll_c_seq = x["roll_c_seq"]

        predictions_video = make_animation(
            source_image,
            source_semantics,
            target_semantics,
            self.generator,
            self.kp_extractor,
            self.mapping,
            use_exp=True,
            yaw_c_seq=yaw_c_seq,
            pitch_c_seq=pitch_c_seq,
            roll_c_seq=roll_c_seq,
        )

        return predictions_video
