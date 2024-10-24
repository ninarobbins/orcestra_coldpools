def get_colors(circle_names):
    mpi_colors = ["#1A237E", "#1976D2", "#81D4FA", "#00695C"]
    atr_colors = ["#FF8F00", "#FFC107"]
    colors = {}
    for circle in circle_names:
        if "atr" in circle:
            colors[circle] = atr_colors.pop(0)
        else:
            colors[circle] = mpi_colors.pop(0)
    return colors


def set_yticks_with_color(axes, yticks, ytick_colors, **kwargs):
    for ax in axes:
        ax.set_yticks(yticks, **kwargs)
        yticks_ax = ax.yaxis.get_major_ticks()
        for tick, c in zip(yticks_ax, ytick_colors):
            tick.tick1line.set_markeredgecolor(c)
            tick.tick2line.set_markeredgecolor(c)


def set_xticks_with_color(axes, xticks, xtick_colors, **kwargs):
    for ax in axes:
        ax.set_xticks(xticks, **kwargs)
        xticks_ax = ax.xaxis.get_major_ticks()
        for tick, c in zip(xticks_ax, xtick_colors):
            tick.tick1line.set_markeredgecolor(c)
            tick.tick2line.set_markeredgecolor(c)
