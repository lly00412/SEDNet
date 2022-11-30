from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .virtual_kitti_dataset import VIRTUALKITTIDataset2
from .drivingstereo_dataset import DrivingStereoDataset
from .apolloscape_dataset import ApolloscapeDataset
from .ethslam_dataset import ETHSLAMDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "vkitti2":VIRTUALKITTIDataset2,
    "drivingstereo":DrivingStereoDataset,
    "apolloscape":ApolloscapeDataset,
    "ethslam":ETHSLAMDataset,
}
