model = torch.load("cifar10_model_full.pth", map_location=torch.device("cpu"))
model.eval()

