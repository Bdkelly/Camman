import torch
from PIL import Image

from training.detection.train import trainer


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets):
        assert images[0].shape == (3, 640, 640)
        assert targets[0]["boxes"].shape == (1, 4)
        return {"loss": self.weight.square()}


def test_detector_training_writes_checkpoint_and_final_weights(tmp_path, mocker):
    Image.new("RGB", (100, 50), "white").save(tmp_path / "ball.jpg")
    output = tmp_path / "output"
    mocker.patch("training.detection.train.fmodel", return_value=TinyDetector())
    trainer(
        [{"frame": "ball.jpg", "boxes": [[10, 5, 20, 10]]}],
        tmp_path,
        1,
        output=output,
        batch_size=1,
        workers=0,
        device="cpu",
        pretrained=False,
    )
    checkpoint = torch.load(output / "checkpoint_epoch_1.pth", weights_only=True)
    final = torch.load(output / "trained_model_final.pth", weights_only=True)
    assert checkpoint["epoch"] == 0
    assert final["weight"].item() < 1.0
    torch.testing.assert_close(checkpoint["model_state_dict"]["weight"], final["weight"])
