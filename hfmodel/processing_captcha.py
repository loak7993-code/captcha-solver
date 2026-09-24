import string
import torch
import torchvision.transforms.functional as F
from transformers.processing_utils import ProcessorMixin

class CaptchaProcessor(ProcessorMixin):
    attributes = []
    def __init__(self, vocab=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab or (string.ascii_lowercase + string.ascii_uppercase + string.digits)
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.vocab)}
        self.idx_to_char[0] = ""

    def __call__(self, images):
        """
        Converts PIL images to the tensor format the CRNN expects.
        """
        if not isinstance(images, list):
            images = [images]
            
        processed_images = []
        for img in images:
            # Convert to Grayscale
            img = img.convert("L")
            # Resize to your model's expected input (Width, Height)
            img = img.resize((150, 40))
            # Convert to Tensor and Scale to [0, 1]
            img_tensor = F.to_tensor(img) 
            processed_images.append(img_tensor)
            
        return {"pixel_values": torch.stack(processed_images)}

    def batch_decode(self, logits):
        """
        CTC decoding logic.
        """
        tokens = torch.argmax(logits, dim=-1)
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
            
        decoded_strings = []
        for batch_item in tokens:
            char_list = []
            for i in range(len(batch_item)):
                token = batch_item[i].item()
                if token != 0:
                    if i > 0 and batch_item[i] == batch_item[i - 1]:
                        continue
                    char_list.append(self.idx_to_char.get(token, ""))
            decoded_strings.append("".join(char_list))
        return decoded_strings