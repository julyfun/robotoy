from robotoy.itp_state.itp_state import ItpState
import numpy as np

from matplotlib import pyplot as plt
from loguru import logger
import matplotlib.font_manager as fm
from matplotlib import rcParams

font_path = '/usr/share/fonts/opentype/SourceHanSerifCN-Bold-2.otf'  # macOS上宋体路径
songti_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

Flt = np.float32
EPS = 1e-6

def arr(*l):
    return np.array(l, dtype=Flt)

def test_itp_state():
    x0 = arr(-1)
    v0 = arr(-1)
    v_max = arr(3.0)
    a_max = arr(20.0)
    j_max = arr(330)
    fps = 300.0
    itp_state = ItpState()
    itp_state.init(v_max=v_max, a_max=a_max, j_max=j_max, fps=fps)
    itp_state.init(x0=x0, v0=v0)


    t_samples = np.arange(0, 4, 1 / 60)
    freq = 0.5
    amp = 0.5
    x_samples = np.sin(2 * np.pi * freq * t_samples) * amp
    v_samples = 2 * np.pi * freq * np.cos(2 * np.pi * freq * t_samples) * amp
    # v_samples = np.zeros_like(x_samples)

    # t_samples = np.arange(0, 1.0, 1 / 60)
    # x_samples = 2.4 * np.ones_like(t_samples)
    # v_samples = np.zeros_like(t_samples)

    last_itpltn_t = 0
    res = []
    t_vals = []
    for i, (x, v, t) in enumerate(zip(x_samples, v_samples, t_samples)):
        points_needed = int((t - last_itpltn_t) * fps)
        # logger.info(f"start: {last_itpltn_t}")
        res += itp_state.interpolate(
            arr(x), arr(v), points_needed, first_delta_t=(last_itpltn_t - t)
        )
        t_vals += [last_itpltn_t + i / fps for i in range(points_needed)]
        last_itpltn_t += points_needed / fps

    res = np.array(res)
    x_vals = np.array(res[:, 0])
    v_vals = np.array(res[:, 1])
    a_vals = np.array(res[:, 2])
    j_vals = np.array(res[:, 3])

    # [DEBUG] for plot
    fixed_j_vals = np.zeros_like(j_vals)
    for i in range(0, len(j_vals) - 1):
        if (i > 0 and j_vals[i - 1] * j_vals[i] < 0) or (i > 1 and j_vals[i - 1] * j_vals[i - 2] < 0) or (i > 2 and j_vals[i - 2] * j_vals[i - 3] < 0):
            fixed_j_vals[i] = 0
        else:
            fixed_j_vals[i] = j_vals[i]
    j_vals = fixed_j_vals

    def plot_subplots(t, data, titles, y_labels, extra_data=None, extra_t=None):
        # 设置字体和样式
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 8,                 
            'font.family': 'serif',
            'mathtext.fontset': 'stix',
            'axes.labelsize': 9,           
            'axes.titlesize': 10,          
            'xtick.labelsize': 8,          
            'ytick.labelsize': 8,
            'legend.fontsize': 7,          
            # 'figure.figsize': (6, 8),      
            'figure.dpi': 300,
            'lines.linewidth': 1.0,        
            'lines.markersize': 3,         
            'axes.linewidth': 0.8,         
            'grid.linewidth': 0.4,         
        })

        fig, axs = plt.subplots(4, 1, figsize=(5, 6))
        fig.subplots_adjust(hspace=0.60)       # 稍微减小子图间距

        # ...existing code...

        for i, (ax, data_vals, title, y_label) in enumerate(zip(axs, data, titles, y_labels)):
            for j in range(data_vals.shape[1]):
                ax.plot(
                    t,
                    data_vals[:, j],
                    color='black',
                    marker='.',
                    markersize=2,              # 减小数据点大小
                    linestyle='-',
                    zorder=3,
                )
            
            if extra_data is not None and extra_t is not None and i < len(extra_data):
                ax.plot(
                    extra_t,
                    extra_data[i],
                    'x',
                    color='#FF2C00',
                    markersize=3,              # 减小采样点大小
                    linestyle='--',
                    zorder=2,
                )

            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.4, zorder=1)
            
            # 设置刻度
            ax.tick_params(direction='in', length=2, width=0.8, colors='black',
                        grid_color='gray', grid_alpha=0.5)
            
            # 设置边框
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
            
            ax.set_title(title, pad=6, fontproperties=songti_font, fontsize=7)
            if i == 3:
                ax.set_xlabel('时间 (s)', labelpad=4, fontproperties=songti_font, fontsize=7)
                time_segments = [
                    (0.00, 0.06, 't_7'),
                    (0.06, 0.14, 't_6'),
                    (0.14, 0.20, 't_5'),
                    (0.20, 0.80, 't_4'),
                    (0.80, 0.86, 't_3'),
                    (0.86, 0.94, 't_2'),
                    (0.94, 0.99, 't_1')
                ]
                
                # [t label]
                # ymin = ax.get_ylim()[0]
                # y_pos = ymin - (ax.get_ylim()[1] - ymin) * 0.20  # 调整标注位置
                
                # for t_start, t_end, label in time_segments:
                #     # 添加箭头标注
                #     ax.annotate('', 
                #         xy=(t_start, y_pos), 
                #         xytext=(t_end, y_pos),
                #         arrowprops=dict(arrowstyle='<->',
                #                       linewidth=0.8,
                #                       shrinkA=0,
                #                       shrinkB=0))
                    
                #     # 添加文本标签
                #     mid_point = (t_start + t_end) / 2
                #     ax.text(mid_point, y_pos * 0.62,
                #            r"$%s$" % label,
                #            horizontalalignment='center',
                #            verticalalignment='top',
                #            fontsize=8)
                
                #
                # ax.set_ylim(bottom=y_pos * 1.2)
              
            
            # 优化图例
            # ax.legend(frameon=True, 
            #          fancybox=False, 
            #          edgecolor='black',
            #          ncol=2, 
            #          loc='upper right', 
            #          bbox_to_anchor=(0.99, 0.99),
            #          fontsize=7,               # 明确设置图例字体大小
            #          borderaxespad=0.1)        # 减小图例边距

        plt.show()


    data = [x_vals, v_vals, a_vals, j_vals]
    titles = ["位置", "速度", "加速度", "急动度"]
    y_labels = ["位置", "速度", "加速度", "急动度"]
    extra_data = [x_samples, v_samples]
    extra_t = t_samples

    plot_subplots(t_vals, data, titles, y_labels, extra_data, extra_t)

test_itp_state()
