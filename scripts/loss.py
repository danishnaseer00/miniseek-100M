# comparison between loss of 17.7M and 101.4M parameter models on WikiText-103

import matplotlib.pyplot as plt

tokens_millions = [117.9, 235.9, 353.8, 471.7, 589.7]

val_loss_17 = [3.9735, 3.8260, 3.7438, 3.6793, 3.6582]
val_loss_101 = [3.578750, 3.357910, 3.251548, 3.186434, 3.169377]

train_loss_17 = [4.65, 4.02, 3.85, 3.74, 3.6944]
train_loss_101 = [4.245337, 3.454156, 3.234603, 3.072189, 2.958632]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10,
})

fig, ax = plt.subplots(figsize=(9, 5.4), dpi=300)

c_17 = '#0072B2'    
c_101 = '#E69F00'       
c_train_17 = '#6E7781'  
c_train_101 = '#424A53' 


ax.plot(tokens_millions, val_loss_17, color=c_17, lw=2.2, marker='o', markersize=6, 
        label='Val Loss (17.7M)')
ax.plot(tokens_millions, val_loss_101, color=c_101, lw=2.2, marker='s', markersize=6, 
        label='Val Loss (101.4M)')

ax.plot(tokens_millions, train_loss_17, color=c_train_17, lw=1.8, ls='--', marker='o', 
        fillstyle='none', markersize=6, label='Train Loss (17.7M)')
ax.plot(tokens_millions, train_loss_101, color=c_train_101, lw=1.8, ls='--', marker='^', 
        markersize=6, label='Train Loss (101.4M)')

ax.annotate('Train: 3.69', xy=(tokens_millions[-1], train_loss_17[-1]), xytext=(16, 11),
            textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_train_17,
            bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_train_17, lw=0.9, alpha=0.95))

ax.annotate('Val: 3.66', xy=(tokens_millions[-1], val_loss_17[-1]), xytext=(16, -13),
            textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_17,
            bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_17, lw=0.9, alpha=0.95))

ax.annotate('Val: 3.17', xy=(tokens_millions[-1], val_loss_101[-1]), xytext=(16, 10),
            textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_101,
            bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_101, lw=0.9, alpha=0.95))


ax.annotate('Train: 2.96', xy=(tokens_millions[-1], train_loss_101[-1]), xytext=(16, -10),
            textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_train_101,
            bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_train_101, lw=0.9, alpha=0.95))

ax.set_xlim(90, 715)
ax.set_ylim(2.7, 4.85)
ax.set_xlabel('Tokens seen (millions)', fontweight='bold')
ax.set_ylabel('Cross-entropy loss', fontweight='bold')
ax.set_title('WikiText-103 Convergence: 17.7M vs. 101.4M Parameters', fontweight='bold', pad=12)

ax.grid(True, ls='--', alpha=0.45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='white', edgecolor='#D0D0D0', loc='upper right')

plt.tight_layout()
plt.savefig('./figures/loss.png', dpi=300, bbox_inches='tight')
plt.show()