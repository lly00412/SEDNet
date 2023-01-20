from models.gwcnet import GwcNet_G, GwcNet_GC, GwcNet_GCS
from models.loss import model_loss

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-gc": GwcNet_GC,
    "gwcnet-gcs": GwcNet_GCS
}
