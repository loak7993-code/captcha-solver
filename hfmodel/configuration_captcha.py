from transformers import PretrainedConfig

class CaptchaConfig(PretrainedConfig):
    model_type = "captcha_crnn"
    def __init__(self, num_chars=63, **kwargs):
        super().__init__(**kwargs)
        self.num_chars = num_chars
