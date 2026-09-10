from torchinfo import summary

from unet import UNET

model = UNET(5, 16, 1, padding=1, downhill=4)

summary(model, input_size=(64, 5, 64, 64), device='mps')
