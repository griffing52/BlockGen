"""Run: .venv/bin/python -m blockgen.experiments_analysis (writes outputs/analysis/*)."""
"""Tokenization-methods figure + learned-embedding analysis (birch vs oak?)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np, torch
from pathlib import Path
from collections import Counter

from blockgen.curation.curate import Curator
from blockgen.experiments_gen import houses_subset, canonicalize, ar_feasible
from blockgen.utils.serialize import (build_block_vocab, structure_to_tokens, structure_to_grid,
                                      BOS_TOKEN, EOS_TOKEN, NUM_SPECIAL)
from blockgen.utils.data import _resource_location_for
from blockgen.tokenizers.standard_vocab import STANDARD_VOCAB
from blockgen.renderer.render import render_schem

DIM=12; SEQ=1600
OUT=Path("outputs/analysis"); OUT.mkdir(parents=True, exist_ok=True)
CKPT=Path("outputs/run_20260630_232913_gen/ar_12.pt")

lab=Curator.from_labeled_cache(max_dim=24)
canon=canonicalize(houses_subset(lab), DIM)
vocab=build_block_vocab(canon, max_dim=DIM)
shared=ar_feasible(canon, vocab, SEQ)
print("vocab", vocab.vocab_size, "blocks", vocab.num_blocks)

def short_name(i):
    tok=vocab.id_to_block_token[i]
    return STANDARD_VOCAB.get(tok, STANDARD_VOCAB.get(tok.split(":")[0], f"id{tok}")).split("(")[0]

# token frequencies in corpus
freq=Counter()
for s in shared:
    for t in structure_to_tokens(s, vocab):
        if vocab.is_block(t): freq[t-vocab.block_offset]+=1

# ============================ FIG 1: tokenization methods ============================
# pick a small structure for legibility
small=min(shared, key=lambda s: s.crop_to_non_air().occupied_mask.sum() if s.crop_to_non_air().occupied_mask.sum()>=8 else 1e9)
sc=small.crop_to_non_air()
toks=structure_to_tokens(small, vocab)
grid=structure_to_grid(small, DIM, vocab)

fig=plt.figure(figsize=(20,5))
# panel 1: structure
ax=fig.add_subplot(1,4,1,projection="3d"); render_schem(small,ax=ax,max_dim=24,show=False)
ax.set_title(f"(1) Structure\n{sc.shape}, {int(sc.occupied_mask.sum())} blocks",fontsize=11)
ax.set_xlabel("");ax.set_ylabel("");ax.set_zlabel("")

# panel 2: AR token stream as chips
ax=fig.add_subplot(1,4,2); ax.axis("off")
ax.set_title("(2) AR token sequence  [BOS,(X,Y,Z,BLOCK)*,EOS]\nair NOT emitted — only occupied voxels",fontsize=11)
labels=[]; colors=[]
qpos=0  # 0..3 within quad after BOS
for t in toks:
    if t==BOS_TOKEN: labels.append("BOS"); colors.append("#888"); qpos=0; continue
    if t==EOS_TOKEN: labels.append("EOS"); colors.append("#888"); continue
    if vocab.is_coord(t):
        axis="XYZ"[qpos%3]; labels.append(f"{axis}={vocab.decode_coord(t)}"); colors.append("#9ecae1"); qpos+=1
    elif vocab.is_block(t):
        bid,bd=vocab.decode_block(t); labels.append(short_name(t-vocab.block_offset)[:10]); colors.append("#fdae6b"); qpos=0
N=min(len(labels),33)
cw=0.165; ch=0.5; perrow=6
for k in range(N):
    r=k//perrow; c=k%perrow
    ax.add_patch(Rectangle((c*cw,-r*0.62),cw*0.93,ch,color=colors[k],ec="k",lw=0.5,transform=ax.transData))
    ax.text(c*cw+cw*0.46,-r*0.62+ch/2,labels[k],ha="center",va="center",fontsize=6.5)
if len(labels)>N: ax.text(0,-((N//perrow)+1)*0.62,f"... +{len(labels)-N} more tokens ({len(labels)} total)",fontsize=8)
ax.set_xlim(0,perrow*cw); ax.set_ylim(-((N//perrow)+1)*0.62,ch+0.1)

# panel 3: grid representation (one y-layer)
ax=fig.add_subplot(1,4,3)
ylayer=int(np.argmax([(grid[:,y,:]>0).sum() for y in range(DIM)]))
sl=grid[:,ylayer,:]
ax.imshow((sl>0),cmap="Greys",vmin=0,vmax=1)
for (i,j),v in np.ndenumerate(sl):
    ax.text(j,i,int(v),ha="center",va="center",fontsize=6,color="red" if v==0 else "black")
ax.set_title(f"(3) Fixed grid (diffusion), y={ylayer} slice\nair = class 0 (red); +1 MASK class",fontsize=11)
ax.set_xticks([]);ax.set_yticks([])

# panel 4: graph representation
ax=fig.add_subplot(1,4,4,projection="3d")
occ=np.argwhere(sc.occupied_mask)
ax.scatter(occ[:,0],occ[:,2],occ[:,1],s=40,c="#3182bd",depthshade=True)
occ_set=set(map(tuple,occ.tolist()))
nb=[(1,0,0),(0,1,0),(0,0,1)]
for x,y,z in occ.tolist():
    for dx,dy,dz in nb:
        p=(x+dx,y+dy,z+dz)
        if p in occ_set: ax.plot([x,p[0]],[z,p[2]],[y,p[1]],c="#aaa",lw=0.6)
ax.set_title("(4) Graph (block+port)\nnodes=blocks, edges=adjacency → studs/pins",fontsize=11)
ax.set_xlabel("");ax.set_ylabel("");ax.set_zlabel("")
fig.suptitle("Tokenization / representation methods — same structure, three model inputs (+ what counts as air)",fontsize=14)
fig.tight_layout(); fig.savefig(OUT/"tokenization_methods.png",dpi=120,bbox_inches="tight"); plt.close(fig)
print("wrote tokenization_methods.png")

# ============================ FIG 2: embedding analysis ============================
sd=torch.load(CKPT, map_location="cpu")
emb=sd["token_embedding.weight"].numpy()
block_emb=emb[vocab.block_offset: vocab.block_offset+vocab.num_blocks]  # (num_blocks, d)
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

# index helper by (id,data)
pair_to_idx={vocab.block_index_to_pair[i]:i for i in range(vocab.num_blocks)}
def get(idx_id,data):
    return pair_to_idx.get((idx_id,data))

WOOD=[("oak",5,0),("spruce",5,1),("birch",5,2),("jungle",5,3),("acacia",5,4),("darkoak",5,5)]
WOOL=[("white",35,0),("orange",35,1),("magenta",35,2),("ltblue",35,3),("yellow",35,4),
      ("lime",35,5),("pink",35,6),("gray",35,7),("ltgray",35,8),("cyan",35,9),
      ("purple",35,10),("blue",35,11),("brown",35,12),("green",35,13),("red",35,14),("black",35,15)]

def fam_heatmap(ax, fam, title):
    present=[(n,get(i,d)) for n,i,d in fam if get(i,d) is not None]
    present=[(n,j) for n,j in present if freq.get(j,0)>=3]  # need to be trained
    if len(present)<2:
        ax.text(0.5,0.5,f"{title}\n<2 trained variants present",ha="center"); ax.axis("off"); return None
    names=[n for n,_ in present]; idxs=[j for _,j in present]
    M=np.array([[cos(block_emb[a],block_emb[b]) for b in idxs] for a in idxs])
    im=ax.imshow(M,cmap="viridis",vmin=-0.2,vmax=1.0)
    ax.set_xticks(range(len(names)));ax.set_xticklabels(names,rotation=90,fontsize=7)
    ax.set_yticks(range(len(names)));ax.set_yticklabels(names,fontsize=7)
    ax.set_title(f"{title}\n({len(names)} variants, n>=3)",fontsize=10)
    off=M[~np.eye(len(M),dtype=bool)]
    return float(off.mean())

fig,axes=plt.subplots(1,3,figsize=(18,5.5))
m_wood=fam_heatmap(axes[0],WOOD,"Wood-plank variants cos-sim")
m_wool=fam_heatmap(axes[1],WOOL,"Wool-color variants cos-sim")

# baseline: random within-vocab cos-sim among well-trained tokens
trained=[i for i in range(vocab.num_blocks) if freq.get(i,0)>=3]
rng=np.random.default_rng(0)
pairs=rng.choice(trained,(400,2))
base=np.mean([cos(block_emb[a],block_emb[b]) for a,b in pairs if a!=b])

# PCA 2D of trained embeddings, colored by family
X=block_emb[trained]; Xc=X-X.mean(0)
U,S,Vt=np.linalg.svd(Xc,full_matrices=False); P=Xc@Vt[:2].T
def fam_of(i):
    bid,_=vocab.block_index_to_pair[i]
    rl=_resource_location_for(bid, vocab.block_index_to_pair[i][1])
    return rl.split(":")[-1]
fams=[fam_of(i) for i in trained]
top=[f for f,_ in Counter(fams).most_common(7)]
ax=axes[2]
for f in top:
    mask=[j for j,ff in enumerate(fams) if ff==f]
    ax.scatter(P[mask,0],P[mask,1],s=18,label=f)
ax.legend(fontsize=7,loc="best"); ax.set_title("PCA of trained block embeddings\n(colored by material family)",fontsize=10)
ax.set_xlabel("PC1");ax.set_ylabel("PC2")
fig.suptitle(f"Learned AR block embeddings (ar_12.pt) — wood within-fam={m_wood:.2f}  wool within-fam={m_wool:.2f}  random-baseline={base:.2f}",fontsize=13)
fig.tight_layout(); fig.savefig(OUT/"embedding_analysis.png",dpi=120,bbox_inches="tight"); plt.close(fig)
print(f"wrote embedding_analysis.png  wood={m_wood} wool={m_wool} baseline={base:.3f}")
print(f"trained tokens (n>=3): {len(trained)}/{vocab.num_blocks}")
