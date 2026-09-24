"""Final local training: mixed-difficulty corpus with the augmentation that
actually converges.

Post-mortem on the failed run: the custom `HardDS` aug (contrast 0.6-1.6 +
brightness 0.7-1.3 + blur) combined with per-image (x-mean)/std normalisation
destroyed stroke contrast, and CTC never got below a blank-collapse plateau
(loss 2.75, val 0.000). The light aug here - rotate +-4, occasional blur -
converges hard (loss 0.08, val 0.937 by ep 15 on the same mixed data).
"""
import crnn

if __name__ == "__main__":
    crnn.train("data/trainmix3k", "data/trainmix3k/labels.json",
               epochs=35, out="crnn_hard.pt")
