# from turtle import title
import pylab as plt
import numpy as np

plt.rc("font", size=15)  # controls default text size
plt.rc("axes", titlesize=15)  # fontsize of the title
plt.rc("axes", labelsize=15)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=15)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=15)  # fontsize of the y tick labels
plt.rc("legend", fontsize=15)  # fontsize of the legend
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


def phase_flow_train(model, t_train, z_train, lim):
    [X, Y] = np.meshgrid(np.linspace(-lim, lim, 10), np.linspace(-lim, lim, 10))
    dX, dY = np.array(model(t_train, [X, Y]))
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(5)
    ax.scatter(z_train[:, 0], z_train[:, 1], marker=".")
    ax.quiver(X, Y, dX, dY, cmap="gray_r", color=(0.5, 0.5, 0.5))
    plt.xlabel("$q$")
    plt.ylabel("$p$", rotation=0)
    plt.legend(
        ["Training data"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        fancybox=True,
        shadow=False,
        ncol=2,
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.tight_layout()
    return plt


def loss_plot(err_t):
    fig, ax = plt.subplots(1, 1)
    ax.semilogy(err_t)
    ax.set_xlabel("iterations")
    ax.set_ylabel("total loss")
    ax.set_title("MSE Error")


def plot_ld_time(t, yy, color="k", title=""):
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(3)
    fig.set_figwidth(4)
    ax.plot(t, yy.detach()[:, 0], color=color)
    ax.plot(t, yy.detach()[:, 1], "--", color=color)
    plt.xlim(t[0], t[-1])
    plt.xlabel("t")
    plt.ylabel("{q,p}")
    plt.title(title)
    plt.grid(which="minor", linestyle=":")
    plt.tight_layout()
    return plt


def plot_gr_time(t_train, sol_train, color="k", title="ground truth"):
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(3)
    fig.set_figwidth(4)
    ax.plot(t_train, sol_train.T[:, 0], color=color)
    ax.plot(t_train, sol_train.T[:, 1], "--", color=color)
    plt.xlim(t_train[0], t_train[-1])
    plt.xlabel("t")
    plt.ylabel("{q,p}")
    plt.title(title)
    plt.tight_layout()
    return plt


def plot_abs_time(t_train, sol_train, yy):
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(3)
    fig.set_figwidth(4)
    ax.semilogy(t_train, np.abs(yy.detach() - sol_train.T))
    plt.xlim(t_train[0], t_train[-1])
    plt.title("Absolute error")
    plt.xlabel("t")
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":")
    plt.tight_layout()
    return plt


def comparison_plot(t, t_train, sol_train, yy):
    plt.figure(figsize=(24, 5))
    plt.tight_layout()
    plt.subplot(1, 3, 1)
    plt.plot(t, yy.detach())
    plt.xlabel("t")
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":")
    plt.legend(
        ["$q(\hat q,\hat p)$", "$ q(\hat q,\hat p)$"],
        loc="upper center",
        bbox_to_anchor=(0.28, 1.15),
        fancybox=True,
        shadow=False,
        ncol=2,
    )
    plt.subplot(1, 3, 2)
    plt.plot(t_train, sol_train.T)
    plt.legend(
        ["$q$", "$ p$"],
        loc="upper center",
        bbox_to_anchor=(0.18, 1.15),
        fancybox=True,
        shadow=False,
        ncol=2,
    )
    plt.xlabel("t")
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":")
    plt.subplot(1, 3, 3)
    plt.semilogy(t_train, np.abs(yy.detach() - sol_train.T))
    plt.title("Absolute error")
    plt.xlabel("t")
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":")
    return plt


def Hamiltonian_latent(t, latent):
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(5)
    ax.plot(t, latent.detach().numpy())
    plt.title("Hamiltonian")
    plt.ylim(-5, 5)
    plt.xlim(t[0], t[-1])
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":")
    plt.xlabel("t")
    plt.ylabel("$\hat H(\hat q,\hat p)$")
    plt.tight_layout()
    return plt


def Hamiltonian_can(t, can):
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(5)
    ax.plot(t, can)
    plt.title("Hamiltonian")
    plt.ylim(-5, 5)
    plt.xlim(t[0], t[-1])
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":")
    plt.xlabel("t")
    plt.ylabel("$H(q,p)$")
    plt.tight_layout()
    return plt


def Hamiltonian_error(t, can, learned):
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(5)
    ax.plot(t, np.abs(can - learned.detach().numpy()))
    plt.title("Hamiltonian error")
    plt.ylim(-0.25, 0.25)
    plt.xlim(t[0], t[-1])
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":")
    plt.xlabel("t")
    plt.ylabel("$|H(q(\hat q,\hat p),p(\hat q,\hat p))-H(q,p)|$")
    plt.tight_layout()
    return plt


def Hamiltonian_plot(t, can, learned, latent):
    plt.figure(figsize=(24, 5))
    plt.subplot(1, 3, 1)
    plt.plot(t, latent.detach().numpy())
    plt.title("Hamiltonian")
    plt.ylim(-5, 5)
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":")
    plt.xlabel("t")
    plt.ylabel("$\hat H(\hat q,\hat p)$")
    plt.tight_layout()
    plt.subplot(1, 3, 2)
    plt.plot(t, can)
    plt.title("Hamiltonian")
    plt.ylim(-5, 5)
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":")
    plt.xlabel("t")
    plt.ylabel("$H(q,p)$")
    plt.tight_layout()
    plt.subplot(1, 3, 3)
    plt.plot(t, np.abs(can - learned.detach().numpy()))
    plt.title("Hamiltonian error")
    plt.ylim(-0.25, 0.25)
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":")
    plt.xlabel("t")
    plt.ylabel("$|H(q(\hat q,\hat p),p(\hat q,\hat p))-H(q,p)|$")
    plt.tight_layout()
    return plt


def phase_flow(learned, true, lim, color_idx, method_name):
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(5)
    for i in range(0, 1):
        ax.plot(
            true[0, :, i].reshape(-1, 1),
            true[1, :, i].reshape(-1, 1),
            "o",
            markersize=5,
            markevery=50,
            color="k",
            label="ground truth",
        )
        ax.plot(
            learned[0, :, i].reshape(-1, 1),
            learned[1, :, i].reshape(-1, 1),
            color=colors[color_idx],
            label=method_name,
        )

    for i in range(1, learned.shape[-1]):
        ax.plot(
            true[0, :, i].reshape(-1, 1),
            true[1, :, i].reshape(-1, 1),
            "o",
            markersize=5,
            markevery=50,
            color="k",
        )
        ax.plot(
            learned[0, :, i].reshape(-1, 1),
            learned[1, :, i].reshape(-1, 1),
            color=colors[color_idx],
        )
    ax.legend()
    plt.xlim(-lim[0], lim[0])
    plt.ylim(-lim[1], lim[1])
    plt.legend()
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        fancybox=True,
        shadow=False,
        ncol=2,
    )
    #     plt.minorticks_on()
    #     plt.grid(which='minor', linestyle=':')
    plt.xlabel("$q$")
    plt.ylabel("$p$")
    plt.tight_layout()
    return plt


def plot_soln(time, soln, color="k", title="ground truth", yscale=None):

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(time, soln.T[:, 0], color=color)
    ax.plot(time, soln.T[:, 1], "--", color=color)
    plt.xlim(time[0], time[-1])

    plt.xlabel("t")
    plt.ylabel("{q,p}")
    plt.title(title)
    plt.tight_layout()
    if yscale:
        plt.yscale("log")

    return fig

def plot_soln_phasespace(soln, color="k", title="ground truth", **kwargs):

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(soln.T[:, 0], soln.T[:, 1], color=color)
    plt.xlabel("q")
    plt.ylabel("p")
    plt.title(title)
    plt.tight_layout()

    return fig

def save_figs_phasespace(plt_ground_truth, plt_learned, idx, path=""):
    plt_ground_truth.savefig(
        path + f"plot_groundtruth_ps_{idx}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt_ground_truth.savefig(
        path + f"plot_groundtruth_ps_{idx}.pdf", bbox_inches="tight", pad_inches=0.1
    )
    plt_learned.savefig(
        path + f"plot_learned_ps_{idx}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt_learned.savefig(
        path + f"plot_learned_ps_{idx}.pdf", bbox_inches="tight", pad_inches=0.1
    )

def plotting_saving_plots_phasespace(
    gt_sol,
    decoded_latent_sol,
    color_idx,
    method_name,
    idx,
    path="",
    closing_plts=True,
):
    plt_ground_truth = plot_soln_phasespace(gt_sol, color=colors[0])
    plt_learned = plot_soln_phasespace(
        decoded_latent_sol, color=colors[color_idx], title=method_name
    )

    save_figs_phasespace(plt_ground_truth, plt_learned, idx=idx, path=path)

    if closing_plts:
        plt.close("all")



def save_figs(plt_ground_truth, plt_learned, plt_err, idx, path=""):
    plt_ground_truth.savefig(
        path + f"plot_groundtruth_time_{idx}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt_ground_truth.savefig(
        path + f"plot_groundtruth_time_{idx}.pdf", bbox_inches="tight", pad_inches=0.1
    )
    plt_learned.savefig(
        path + f"plot_learned_time_{idx}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt_learned.savefig(
        path + f"plot_learned_time_{idx}.pdf", bbox_inches="tight", pad_inches=0.1
    )
    plt_err.savefig(
        path + f"plot_abstime_{idx}.png", dpi=300, bbox_inches="tight", pad_inches=0.1
    )
    plt_err.savefig(
        path + f"plot_abstime_{idx}.pdf", bbox_inches="tight", pad_inches=0.1
    )


def plotting_saving_plots(
    t,
    gt_sol,
    decoded_latent_sol,
    color_idx,
    method_name,
    idx,
    path="",
    closing_plts=True,
):
    plt_ground_truth = plot_soln(t, gt_sol, color=colors[0])
    plt_learned = plot_soln(
        t, decoded_latent_sol, color=colors[color_idx], title=method_name
    )
    plt_err = plot_soln(
        t, abs(gt_sol - decoded_latent_sol), title="absolute error", yscale="log"
    )

    save_figs(plt_ground_truth, plt_learned, plt_err, idx=idx, path=path)

    if closing_plts:
        plt.close("all")


# ###########################################
# ##########################################
# def NLS_POD_plots():
#     fig1, ax1 = plt.subplots(1, 1, figsize=(5, 3))
#     fig2, ax2 = plt.subplots(1, 1, figsize=(5, 3))

#     for k in range(data.shape[-1]):
#         ax1.plot(t, data[:, k].cpu())
#         ax2.plot(t, decoded_latent_sol[:, k].cpu())

#     ax1.set(xlabel="time", ylabel="POD coeffs")
#     ax2.set(xlabel="time", ylabel="POD coeffs")

#     ax1.axvspan(0, 80, color="blue", alpha=0.15, label="training")
#     ax1.axvspan(80, 160, color="green", alpha=0.15, label="testing")

#     ax2.axvspan(0, 80, color="blue", alpha=0.15, label="training")
#     ax2.axvspan(80, 160, color="green", alpha=0.15, label="testing")
#     ax1.legend()
#     ax1.legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.2),
#         fancybox=True,
#         shadow=False,
#         ncol=2,
#     )

#     ax2.legend()
#     ax2.legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.2),
#         fancybox=True,
#         shadow=False,
#         ncol=2,
#     )

#     fig1.savefig(
#         params.path + f"pod_coeffs_ground_truth_{i}.png",
#         dpi=300,
#         bbox_inches="tight",
#         pad_inches=0.1,
#     )
#     fig1.savefig(
#         params.path + f"pod_coeffs_ground_truth_{i}.pdf",
#         bbox_inches="tight",
#         pad_inches=0.1,
#     )

#     fig2.savefig(
#         params.path + f"pod_coeffs_learned_{i}.png",
#         dpi=300,
#         bbox_inches="tight",
#         pad_inches=0.1,
#     )
#     fig2.savefig(
#         params.path + f"pod_coeffs_learned_{i}.pdf", bbox_inches="tight", pad_inches=0.1
#     )
