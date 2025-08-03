import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("intermediate results:")

        x = self.fc1(x)
        print(x.dtype)

        x = self.relu(x)
        print(x.dtype)

        x = self.ln(x)
        print(x.dtype)

        x = self.fc2(x)
        print(x.dtype)

        return x


def main():
    # NOTE: torch.autocast(device_type) only affects operations on the specified
    # device, so it's important to move model and input to the device! For
    # example, autocast won't happen if the model is on CPU.

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ToyModel(32, 32).to(device)
    x = torch.rand((8, 32), dtype=torch.float32).to(device)
    z = torch.rand((8, 32), dtype=torch.float32).to(device)

    print("Without mixed precision:")
    y = model(x)
    loss = (y - z).mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"{name}: param dtype {p.dtype}, grad dtype {p.grad.dtype}")
        else:
            print(f"{name}: param dtype {p.dtype}")
    print(f"Output: {y.dtype}")
    print()

    print("With mixed precision:")
    with torch.autocast(device_type=device, dtype=torch.float16):
        y = model(x)
        loss = (y - z).mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                print(f"{name}: param dtype {p.dtype}, grad dtype {p.grad.dtype}")
            else:
                print(f"{name}: param dtype {p.dtype}")
        print(f"Output: {y.dtype}")


if __name__ == "__main__":
    main()
