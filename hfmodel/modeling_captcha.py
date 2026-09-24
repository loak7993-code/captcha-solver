import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from .configuration_captcha import CaptchaConfig

class CaptchaCRNN(PreTrainedModel):
    config_class = CaptchaConfig

    def __init__(self, config):
        super().__init__(config)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(512, config.num_chars)
        self.post_init()

    def forward(self, x, labels=None):
        x = self.conv_layer(x)
        x = x.permute(0, 3, 1, 2)
        batch, width, channels, height = x.size()
        x = x.view(batch, width, -1)
        x, _ = self.lstm(x)
        logits = self.classifier(x)
        
        return SequenceClassifierOutput(logits=logits)