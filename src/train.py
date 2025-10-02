
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse, math, os
from pathlib import Path
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('medium')

def make_velocity(nx, ny, omega=1.0):
    x = np.linspace(-1,1,nx)
    y = np.linspace(-1,1,ny)
    X,Y = np.meshgrid(x,y,indexing='ij')
    u = -omega*Y
    v =  omega*X
    return u,v

def step_advdiff(c,u,v,dx,dy,dt,kappa):
    cx = np.zeros_like(c)
    cy = np.zeros_like(c)
    cx[:,1:-1] = np.where(u[:,1:-1]>0,(c[:,1:-1]-c[:,:-2])/dx,(c[:,2:]-c[:,1:-1])/dx)
    cy[1:-1,:] = np.where(v[1:-1,:]>0,(c[1:-1,:]-c[:-2,:])/dy,(c[2:,:]-c[1:-1,:])/dy)
    lap = (np.roll(c,1,0)-2*c+np.roll(c,-1,0))/dx**2 + (np.roll(c,1,1)-2*c+np.roll(c,-1,1))/dy**2
    return c - dt*(u*cx + v*cy) + dt*kappa*lap

def gen_sequence(nx=64,ny=64,steps=20,dt=0.02,omega=1.0,kappa=1e-3,amp=1.0):
    x = np.linspace(-1,1,nx); y = np.linspace(-1,1,ny)
    X,Y = np.meshgrid(x,y,indexing='ij')
    c = np.exp(-((X-0.3)**2+(Y+0.2)**2)/0.05) + 0.5*amp*np.exp(-((X+0.4)**2+(Y-0.3)**2)/0.03)
    u,v = make_velocity(nx,ny,omega)
    dx,dy = x[1]-x[0], y[1]-y[0]
    seq=[c.copy()]
    for _ in range(steps):
        c = step_advdiff(c,u,v,dx,dy,dt,kappa)
        seq.append(c.copy())
    return np.array(seq)  # [T+1,nx,ny]

class AdDataset(Dataset):
    def __init__(self, N=80, nx=64, ny=64):
        self.samples=[]
        rng = np.random.RandomState(0)
        for _ in range(N):
            omega = 0.5 + 2.0*rng.rand()
            kappa = 5e-4 + 2e-3*rng.rand()
            amp   = 0.5 + 1.5*rng.rand()
            seq = gen_sequence(nx,ny,steps=12,dt=0.02,omega=omega,kappa=kappa,amp=amp)
            # make pairs (state, params) -> next_state
            for t in range(seq.shape[0]-1):
                s = seq[t]; sn = seq[t+1]
                params = np.array([omega,kappa,amp],dtype=np.float32)
                self.samples.append((s.astype(np.float32), params, sn.astype(np.float32)))
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        s,p, sn = self.samples[idx]
        # channels: [state, omega, kappa, amp] where params are broadcasted
        nx,ny = s.shape
        pmap = np.stack([np.full_like(s,p[0]), np.full_like(s,p[1]), np.full_like(s,p[2])],axis=0)
        x = np.concatenate([s[None,...], pmap], axis=0)
        return torch.from_numpy(x), torch.from_numpy(sn[None,...])

class TinyUNet(nn.Module):
    def __init__(self,c_in=4,c_out=1,feat=32):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(c_in,feat,3,padding=1), nn.ReLU(), nn.Conv2d(feat,feat,3,padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(nn.Conv2d(feat,2*feat,3,padding=1), nn.ReLU(), nn.Conv2d(2*feat,2*feat,3,padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.mid = nn.Sequential(nn.Conv2d(2*feat,4*feat,3,padding=1), nn.ReLU(), nn.Conv2d(4*feat,2*feat,1))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv2d(4*feat,2*feat,3,padding=1), nn.ReLU())
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(nn.Conv2d(3*feat,feat,3,padding=1), nn.ReLU())
        self.out = nn.Conv2d(feat,c_out,1)
    def forward(self,x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        m = self.mid(p2)
        u1 = self.up1(m)
        c1 = torch.cat([u1,d2],dim=1)
        c1 = self.dec1(c1)
        u2 = self.up2(c1)
        c2 = torch.cat([u2,d1],dim=1)
        c2 = self.dec2(c2)
        return self.out(c2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    ds = AdDataset(N=60)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    model = TinyUNet()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    losses=[]
    for ep in range(args.epochs):
        model.train()
        for x,y in dl:
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred,y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Epoch {ep+1}/{args.epochs} loss {np.mean(losses[-len(dl):]):.5f}")

    out = Path("outputs"); out.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), out/"model.pt")

    # plot loss
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(losses)
    plt.xlabel("iteration"); plt.ylabel("MSE loss"); plt.title("Training loss")
    plt.savefig(out/"loss.png", dpi=150); plt.close()
    print(f"Saved weights to {out/'model.pt'}")

if __name__ == "__main__":
    main()
