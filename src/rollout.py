
import numpy as np, torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from train import TinyUNet, gen_sequence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega", type=float, default=1.5)
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    nx=64; ny=64; dt=0.02
    seq = gen_sequence(nx,ny,steps=5,dt=dt,omega=args.omega,kappa=args.kappa,amp=args.amp)
    state = seq[-1]

    model = TinyUNet()
    model.load_state_dict(torch.load("outputs/model.pt", map_location="cpu"))
    model.eval()

    states=[state.copy()]
    for t in range(args.steps):
        s = states[-1].astype(np.float32)
        p = np.array([args.omega,args.kappa,args.amp],dtype=np.float32)
        pmap = np.stack([np.full_like(s,p[0]), np.full_like(s,p[1]), np.full_like(s,p[2])],axis=0)
        x = np.concatenate([s[None,...], pmap], axis=0)[None,...]
        with torch.no_grad():
            nxt = model(torch.from_numpy(x)).numpy()[0,0]
        states.append(nxt)

    states = np.array(states)
    out = Path("outputs"); out.mkdir(exist_ok=True, parents=True)

    # plot first and last
    plt.figure()
    plt.imshow(states[0], origin="lower", aspect="auto")
    plt.colorbar(); plt.title("Initial")
    plt.savefig(out/"upt_initial.png", dpi=150); plt.close()

    plt.figure()
    plt.imshow(states[-1], origin="lower", aspect="auto")
    plt.colorbar(); plt.title("U-PT Rollout Final")
    plt.savefig(out/"upt_final.png", dpi=150); plt.close()

    np.save(out/"upt_rollout.npy", states)
    print(f"Saved rollout to {out/'upt_rollout.npy'} with shape {states.shape}")

if __name__ == "__main__":
    main()
