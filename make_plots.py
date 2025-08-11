from matplotlib.colors import CenteredNorm, LinearSegmentedColormap, LogNorm, BoundaryNorm
from matplotlib.pyplot import colormaps, plot
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from numpy import ma

from core.single_plot import plot_bandstructure


plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amstext}',
    'figure.figsize': (4,4),
    'figure.dpi': 200,
    'legend.fontsize': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
    })

def renorm_cmap(cmap, f, name, n=256):
    if hasattr(cmap, 'colors'):
        colors = cmap.colors
        n = len(colors)
    else:
        cmap.resampled(n)
        colors = cmap(range(n))

    nodes = f(np.linspace(0, 1, n))
    return LinearSegmentedColormap.from_list(name, list(zip(nodes, colors)))

def f(x):
    x = x + 1
    return (x*x*x - 1) / 7

viridis_rn = renorm_cmap(colormaps['viridis'], f, 'viridis_rn')
plasma_rn = renorm_cmap(colormaps['plasma'], f, 'plasma_rn')

def custom_cax(ax):
    return inset_axes(
        ax,
        width="5%",
        height="90%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

data = np.load('newraw/scan_5_50_70_-70_-1_70.npz')

av = data['vsl'][4:-5,:]
bv = data['v0'][4:-5,:]
width, gap, chern, berryfluc, trvioliso = [data[o][4:-5,:] for o in ['bandwidth', 'bandgap', 'chern', 'berryfluctuation', 'tracecondviolation']]

mask = (gap < 1) | (chern < 0.6) | (chern > 1.4) | (-1.33 * av + 35 < bv)
widthm = ma.array(width, mask=mask)
gapm = ma.array(gap, mask=mask)
berryflucm = ma.array(berryfluc, mask=mask)
trviolisom = ma.array(trvioliso, mask=mask)

gap_max = bv[0, np.argmax(gapm, axis=-1)]
berrfluc_min = bv[0, np.argmin(berryflucm, axis=-1)]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

mesh1 = ax1.pcolormesh(av, bv, widthm, cmap=viridis_rn)
cax1 = custom_cax(ax1)
cb1 = fig1.colorbar(mesh1, cax=cax1)
cb1.ax.set_title(r"$W${\normalsize (meV)}", loc='left', pad=10)

mesh2 = ax2.pcolormesh(av, bv, trviolisom, cmap=plasma_rn)
cax2 = custom_cax(ax2)
cb2 = fig2.colorbar(mesh2, cax=cax2)
cb2.ax.set_title(r"$\overline{T}$", pad=10)
cb2.ax.set_yticks([4,8,12,16,20,24])

mesh3 = ax3.pcolormesh(av, bv, berryflucm, cmap=plasma_rn)#, vmin=0, vmax=3)
cax3 = custom_cax(ax3)
cb3 = fig3.colorbar(mesh3, cax=cax3)
cb3.ax.set_title(r"$F$", pad=10)

mesh4 = ax4.pcolormesh(av, bv, gapm, cmap=viridis_rn)
cax4 = custom_cax(ax4)
cb4 = fig4.colorbar(mesh4, cax=cax4)
cb4.ax.set_title(r"$\Delta${\normalsize (meV)}", loc='left', pad=10)

for ax in [ax1,ax2,ax3,ax4]:
    ax.scatter(30, -30, marker='*', s=250, color='r')
    ax.plot([20,35], [-12,-35], color='r')
    ax.set_xlabel(r"$V_{\text{SL}}$ (meV)")
    ax.set_ylabel(r"$V_0$ (meV)")

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig1.savefig('plots/paper/W.png', bbox_inches='tight', dpi=fig1.dpi)
fig2.savefig('plots/paper/T.png', bbox_inches='tight', dpi=fig2.dpi)
fig3.savefig('plots/paper/F.png', bbox_inches='tight', dpi=fig3.dpi)
fig4.savefig('plots/paper/Delta.png', bbox_inches='tight', dpi=fig4.dpi)

def chern_and_gap(raw, name, rect=None, newthresh=1.5):
    data = np.load(raw)

    av = data['vsl']
    bv = data['v0']
    width, gap, chern = data['bandwidth'], data['bandgap'], data['chern']
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    new_mask = (np.abs(np.round(chern) - chern) > 0.15) | (gap < newthresh)
    chern = ma.array(chern, mask=new_mask)

    chern_norm = BoundaryNorm(boundaries=[-2.5,-1.5,-0.5,0.5,1.5,2.5], ncolors=256)

    mesh1 = ax1.pcolormesh(av, bv, chern, cmap='RdBu_r', norm=chern_norm)
    cax1 = custom_cax(ax1)
    cb1 = fig1.colorbar(mesh1, cax=cax1)
    cb1.ax.set_title(r"$\mathcal{C}$")
    cb1.ax.set_yticks([-2,-1,0,1,2])
    cb1.ax.minorticks_off()
    ax1.set_facecolor('lightgray')

    mesh2 = ax2.pcolormesh(av, bv, gap, cmap=viridis_rn)
    cax2 = custom_cax(ax2)
    cb2 = fig2.colorbar(mesh2, cax=cax2)
    cb2.ax.set_title(r"$\Delta${\normalsize (meV)}", loc='left', pad=10)

    if rect is not None:
        ax1.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], edgecolor='red', facecolor='none', linewidth=2))
        ax2.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], edgecolor='red', facecolor='none', linewidth=2))

    for ax in [ax1, ax2]:
        ax.set_xlabel(r"$V_{\text{SL}}$ (meV)")
        ax.set_ylabel(r"$V_0$ (meV)")

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('plots/paper/' + name + 'C.png', bbox_inches='tight', dpi=fig1.dpi)
    fig2.savefig('plots/paper/' + name + 'Delta.png', bbox_inches='tight', dpi=fig2.dpi)

chern_and_gap('newraw/scan_-50_50_70_-70_70_70.npz', 'triagb0_', [av[0,0], bv[0,0], av[-1,-1] - av[0,0], bv[-1,-1] - bv[0,0]])
chern_and_gap('newraw/scan_50nm_-30_30_70_-42_42_70.npz', 'triag50nm_')
chern_and_gap('newraw/scan_b1_-50_50_70_-70_70_70.npz', 'triagb1_')
chern_and_gap('newraw/scan_sq_-50_50_70_-70_70_70.npz', 'sqb0_')
chern_and_gap('newraw/scan_sq_b1_-50_50_70_-70_70_70.npz', 'sqb1_')

data = np.load('newraw/scanl_t0.05_bq30_600_-600_20_50_20.npz')

l = data['l']
gap, width, trvioliso, berryfluc = [data[o] for o in ['bandwidth', 'bandgap',  'berryfluctuation', 'tracecondviolation']]

fig5, ax5 = plt.subplots(figsize=(5,4))
fig6, ax6 = plt.subplots(figsize=(5,4))
# fig7, ax7 = plt.subplots()

ax5.plot(l, gap, label=r"$\Delta$")
ax5.plot(l, width, label=r"$W$", color='tab:red')
ax5.plot(l, 144/l, c='black', ls='--', label=r"$U_c$")
ax5.fill_between(l, 96 / l, 288 / l, color='black', alpha = 0.1, linewidth=0)
ax5.legend()
ax6.set_xticks([20,25,30,35,40,45,50])
ax5.set_yticks([5,10,15,20])
ax5.set_xlabel(r"$L$ (nm)")
ax5.set_ylabel(r"(meV)")

ax6.plot(l[::2], trvioliso[::2] / 2 / np.pi, color='tab:blue', label=r"$\overline{T}/2\pi$")
ax6.plot(l[::2], berryfluc[::2], color='tab:red', label=r"$F$")
ax6.legend()
ax6.set_xlabel(r"$L$ (nm)")
ax6.set_xticks([20,25,30,35,40,45,50])

fig5.tight_layout()
fig6.tight_layout()
fig5.savefig('plots/paper/lE.png', bbox_inches='tight', dpi=fig5.dpi)
fig6.savefig('plots/paper/lgeom.png', bbox_inches='tight', dpi=fig6.dpi)

def single(raw, name):
    data = np.load(raw)

    ts, hs_ts, Es, hs_labels = data['t'], data['hspoints'], data['energy'], data['hslabels']
    xv, yv, berry = data['kx'], data['ky'], data['berrycurv']

    fig1, ax1 = plt.subplots(figsize=(5,4))
    fig2, ax2 = plt.subplots()

    ax1.plot(ts, Es, c='k')

    focus = data['bandind']
    ax1.plot(ts, Es[:,focus], c='r')

    focus_range = 1.2 * Es[:, focus-5:focus+6]
    yscale = np.abs(focus_range).max()
    ax1.set_ylim(-yscale, yscale)
    
    labels = list(hs_labels)
    labels.append(labels[0])
    ax1.set_xticks(ticks=hs_ts, labels=labels)
    for t in hs_ts:
        ax1.axvline(x=t, ls=':', c='k')

    ax1.yaxis.labelpad = -10
    ax1.set_ylabel("Energy (meV)")
    ax1.set_title("")

    mesh = ax2.pcolormesh(xv, yv, berry, shading='gouraud', norm=CenteredNorm(), cmap='RdBu_r')
    cax2 = custom_cax(ax2)
    cb = fig2.colorbar(mesh, cax=cax2)
    cb.ax.yaxis.labelpad = -10
    cb.ax.set_title(r"$\Omega$ (nm${}^2$)", loc='left', pad=10)
    ax2.yaxis.labelpad = -10
    ax2.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax2.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax2.set_xlabel(r"$k_x$ (nm${}^{-1}$)")
    ax2.set_ylabel(r"$k_y$ (nm${}^{-1}$)")
    ax2.add_patch(patches.Polygon(data['bzverts'], edgecolor='black', facecolor='none', linewidth=2))

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('plots/paper/' + name + 'bands.png', bbox_inches='tight', dpi=fig1.dpi)
    fig2.savefig('plots/paper/' + name + 'berry.png', bbox_inches='tight', dpi=fig2.dpi)

single('newraw/single_30_-30.npz', 'triag_single_')
single('newraw/single_sq_30_30.npz', 'sq_single_')

# plt.show()