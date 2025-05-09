import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: segbench <train|infer> …")
        sys.exit(1)
    cmd, sys.argv = sys.argv[1], ["segbench"] + sys.argv[2:]
    if cmd == "train":
        from segbench.train import main as _train
        _train()
    elif cmd == "infer":
        from segbench.infer import main as _infer
        _infer()
    else:
        print(f"Unknown {cmd}")
        sys.exit(1)
