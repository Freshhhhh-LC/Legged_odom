import os
import glob
import yaml
import argparse
import torch
from utils.model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    parser.add_argument("--output", type=str, help="Path of exported model to save.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint

    model = DenoisingRMA(
        11,
        3 * 2 + 2 + 1 + 1 + 2 + 1 + 2 * 13 + 4 * 11,
        50,
        14,
        64,
    )
    if not checkpoint or (checkpoint == "-1") or (checkpoint == -1):
        checkpoint = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(checkpoint))
    model_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(model_dict["model"], strict=False)

    model.eval()
    script_model = torch.jit.script(model)
    if args.output is None:
        os.makedirs("deploy/models", exist_ok=True)
        args.output = f"deploy/models/{args.task}.pt"
    script_model.save(args.output)
    print(f"Saved model to {args.output}")
