import matplotlib.pyplot as plt

tokens_millions = [117.9, 235.9, 353.8, 471.7, 589.7]

val_loss_17 = [3.9735, 3.8260, 3.7438, 3.6793, 3.6582]
val_loss_101 = [3.578750, 3.357910, 3.251548, 3.186434, 3.169377]

train_loss_17 = [4.65, 4.02, 3.85, 3.74, 3.6944]
train_loss_101 = [4.245337, 3.454156, 3.234603, 3.072189, 2.958632]

science_tokens_millions = [0.0, 1000.0, 2000.0]

science_val_loss = [3.169377, 3.784951, 3.558926]
science_train_loss = [2.958632, 3.811602, 3.596839]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10,
})

c_17 = '#0072B2'
c_101 = '#E69F00'
c_train_17 = '#6E7781'
c_train_101 = '#424A53'
c_switch = '#B8252C'

fig, (ax_w, ax_s) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300, sharey=True)


ax_w.plot(tokens_millions, val_loss_17, color=c_17, lw=2.2, marker='o', markersize=6,
          label='Val Loss (17.7M)')
ax_w.plot(tokens_millions, val_loss_101, color=c_101, lw=2.2, marker='s', markersize=6,
          label='Val Loss (101.4M)')

ax_w.plot(tokens_millions, train_loss_17, color=c_train_17, lw=1.8, ls='--', marker='o',
          fillstyle='none', markersize=6, label='Train Loss (17.7M)')
ax_w.plot(tokens_millions, train_loss_101, color=c_train_101, lw=1.8, ls='--', marker='^',
          markersize=6, label='Train Loss (101.4M)')

ax_w.annotate('Train: 3.69', xy=(tokens_millions[-1], train_loss_17[-1]), xytext=(16, 11),
              textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_train_17,
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_train_17, lw=0.9, alpha=0.95))
ax_w.annotate('Val: 3.66', xy=(tokens_millions[-1], val_loss_17[-1]), xytext=(16, -13),
              textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_17,
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_17, lw=0.9, alpha=0.95))
ax_w.annotate('Val: 3.17', xy=(tokens_millions[-1], val_loss_101[-1]), xytext=(16, 10),
              textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_101,
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_101, lw=0.9, alpha=0.95))
ax_w.annotate('Train: 2.96', xy=(tokens_millions[-1], train_loss_101[-1]), xytext=(16, -10),
              textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_train_101,
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_train_101, lw=0.9, alpha=0.95))

ax_w.set_xlim(90, 715)
ax_w.set_ylim(2.7, 4.85)
ax_w.set_xlabel('WikiText-103 tokens seen (millions)', fontweight='bold')
ax_w.set_ylabel('Cross-entropy loss', fontweight='bold')
ax_w.set_title('Phases 1-2: WikiText-103 (17.7M vs 101.4M)', fontweight='bold', pad=12)
ax_w.grid(True, ls='--', alpha=0.45)
ax_w.spines['top'].set_visible(False)
ax_w.spines['right'].set_visible(False)


ax_s.plot(science_tokens_millions, science_val_loss, color=c_101, lw=2.2, marker='s',
          markersize=8, label='Val Loss (101.4M)')
ax_s.plot(science_tokens_millions, science_train_loss, color=c_train_101, lw=1.8, ls='--',
          marker='^', markersize=8, label='Train Loss (101.4M)')

ax_s.annotate('Phase-2 end\n(WikiText)', xy=(0.0, 3.169377), xytext=(-20, 24),
              textcoords='offset points', fontsize=8.5, fontweight='bold', color=c_101, ha='center',
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_101, lw=0.9, alpha=0.95))
ax_s.annotate('corpus + LR\nchange', xy=(0.0, 3.784951), xytext=(28, -8),
              textcoords='offset points', fontsize=8.5, fontweight='bold', color=c_101,
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_101, lw=0.9, alpha=0.95))

ax_s.annotate('Val: 3.78', xy=(1000.0, science_val_loss[1]), xytext=(8, -22),
              textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_101,
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_101, lw=0.9, alpha=0.95))
ax_s.annotate('Val: 3.56', xy=(2000.0, science_val_loss[2]), xytext=(12, 6),
              textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_101,
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_101, lw=0.9, alpha=0.95))
ax_s.annotate('Train: 3.60', xy=(2000.0, science_train_loss[2]), xytext=(12, -16),
              textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_train_101,
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor=c_train_101, lw=0.9, alpha=0.95))

ax_s.set_xlim(-120, 2180)
ax_s.set_xlabel('Science-corpus tokens seen (millions)', fontweight='bold')
ax_s.set_title('Phase 3: 1B science-corpus continuation (101.4M)', fontweight='bold', pad=12)
ax_s.grid(True, ls='--', alpha=0.45)
ax_s.spines['top'].set_visible(False)
ax_s.spines['right'].set_visible(False)
ax_s.legend(frameon=True, facecolor='white', edgecolor='#D0D0D0', loc='upper right')

plt.tight_layout()
plt.savefig('./figures/loss.png', dpi=300, bbox_inches='tight')
plt.show()