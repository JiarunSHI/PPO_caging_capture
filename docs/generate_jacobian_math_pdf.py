"""Generate the Chinese mathematical design PDF for constrained Jacobian control."""

import argparse
import datetime as _datetime
import math
from pathlib import Path
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle


REGULAR_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
BOLD_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def _visual_width(text):
    width = 0
    for character in str(text):
        if character == "\t":
            width += 4
        elif unicodedata.east_asian_width(character) in {"W", "F", "A"}:
            width += 2
        else:
            width += 1
    return width


def _wrap_visual(text, maximum=98):
    text = str(text).strip()
    if not text:
        return [""]
    lines = []
    current = ""
    current_width = 0
    last_break = -1
    for character in text:
        if character == "\n":
            lines.append(current.rstrip())
            current = ""
            current_width = 0
            last_break = -1
            continue
        character_width = 2 if unicodedata.east_asian_width(character) in {"W", "F", "A"} else 1
        current += character
        current_width += character_width
        if character in " ，。；：、,.!?;:)]}）】》" or character.isspace():
            last_break = len(current)
        if current_width > maximum:
            if last_break > max(1, len(current) // 2):
                head = current[:last_break].rstrip()
                tail = current[last_break:].lstrip()
            else:
                head = current[:-1].rstrip()
                tail = character
            lines.append(head)
            current = tail
            current_width = _visual_width(current)
            last_break = -1
    if current or not lines:
        lines.append(current.rstrip())
    return lines


class ChinesePdf:
    PAGE_WIDTH_IN = 8.27
    PAGE_HEIGHT_IN = 11.69
    TOP = 0.91
    BOTTOM = 0.075
    LEFT = 0.095
    RIGHT = 0.925

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf = PdfPages(str(self.output_path))
        self.regular = FontProperties(fname=REGULAR_FONT)
        self.bold = FontProperties(fname=BOLD_FONT)
        self.page_number = 0
        self.fig = None
        self.y = self.TOP

    def _new_page(self, chapter="数学原理与实现"):
        if self.fig is not None:
            self._finish_page()
        self.page_number += 1
        self.fig = plt.figure(
            figsize=(self.PAGE_WIDTH_IN, self.PAGE_HEIGHT_IN), facecolor="white"
        )
        self.y = self.TOP
        self.fig.text(
            self.LEFT,
            0.965,
            "UR5e 100 Hz 位姿约束 Jacobian 控制",
            fontproperties=self.bold,
            fontsize=9.0,
            color="#355070",
            ha="left",
            va="top",
        )
        self.fig.text(
            self.RIGHT,
            0.965,
            chapter,
            fontproperties=self.regular,
            fontsize=8.0,
            color="#6b7280",
            ha="right",
            va="top",
        )
        self.fig.lines.append(
            plt.Line2D(
                [self.LEFT, self.RIGHT],
                [0.946, 0.946],
                transform=self.fig.transFigure,
                color="#d7dee8",
                linewidth=0.8,
            )
        )

    def _finish_page(self):
        if self.fig is None:
            return
        self.fig.lines.append(
            plt.Line2D(
                [self.LEFT, self.RIGHT],
                [0.053, 0.053],
                transform=self.fig.transFigure,
                color="#d7dee8",
                linewidth=0.7,
            )
        )
        self.fig.text(
            self.LEFT,
            0.032,
            "坐标：Base 右手系；控制点：attachment_site；姿态：旋转向量",
            fontproperties=self.regular,
            fontsize=7.6,
            color="#6b7280",
            ha="left",
        )
        self.fig.text(
            self.RIGHT,
            0.032,
            "第 {} 页".format(self.page_number),
            fontproperties=self.regular,
            fontsize=7.6,
            color="#6b7280",
            ha="right",
        )
        self.pdf.savefig(self.fig, bbox_inches=None)
        plt.close(self.fig)
        self.fig = None

    def close(self):
        self._finish_page()
        metadata = self.pdf.infodict()
        metadata["Title"] = "UR5e 100Hz 位姿约束 Jacobian 数学原理与实现"
        metadata["Author"] = "PPO Caging Capture Project"
        metadata["Subject"] = "Time-parameterized constrained differential inverse kinematics"
        metadata["Keywords"] = "UR5e, Jacobian, QP, TCP pose, 100Hz, differential IK"
        self.pdf.close()

    def _line_height(self, fontsize, leading=1.55):
        return float(fontsize) * leading / (self.PAGE_HEIGHT_IN * 72.0)

    def _ensure(self, height, chapter="数学原理与实现"):
        if self.fig is None:
            self._new_page(chapter)
        if self.y - height < self.BOTTOM:
            self._new_page(chapter)

    def spacer(self, amount=0.010):
        self._ensure(amount)
        self.y -= amount

    def h1(self, text):
        height = 0.050
        self._ensure(height, text)
        self.fig.text(
            self.LEFT,
            self.y,
            text,
            fontproperties=self.bold,
            fontsize=17,
            color="#17324d",
            ha="left",
            va="top",
        )
        self.y -= height

    def h2(self, text):
        height = 0.040
        self._ensure(height)
        self.fig.text(
            self.LEFT,
            self.y,
            text,
            fontproperties=self.bold,
            fontsize=12.5,
            color="#26547c",
            ha="left",
            va="top",
        )
        self.y -= height

    def paragraph(self, text, fontsize=10.3, color="#202b38", indent=0.0):
        lines = _wrap_visual(text, 96 if indent == 0.0 else 91)
        line_height = self._line_height(fontsize)
        height = line_height * len(lines) + 0.010
        self._ensure(height)
        for line in lines:
            self.fig.text(
                self.LEFT + indent,
                self.y,
                line,
                fontproperties=self.regular,
                fontsize=fontsize,
                color=color,
                ha="left",
                va="top",
            )
            self.y -= line_height
        self.y -= 0.010

    def bullet(self, text, level=0):
        indent = 0.027 + 0.025 * level
        lines = _wrap_visual(text, 90 - 5 * level)
        line_height = self._line_height(10.1)
        height = line_height * len(lines) + 0.006
        self._ensure(height)
        self.fig.text(
            self.LEFT + 0.006 + 0.025 * level,
            self.y,
            "•" if level == 0 else "–",
            fontproperties=self.bold,
            fontsize=10.5,
            color="#2c6e9b",
            va="top",
        )
        for index, line in enumerate(lines):
            self.fig.text(
                self.LEFT + indent,
                self.y,
                line,
                fontproperties=self.regular,
                fontsize=10.1,
                color="#202b38",
                ha="left",
                va="top",
            )
            self.y -= line_height
        self.y -= 0.006

    def equation(self, lines, title=None):
        if isinstance(lines, str):
            lines = [lines]
        wrapped = []
        for line in lines:
            wrapped.extend(_wrap_visual(line, 108))
        line_height = self._line_height(11.0, 1.65)
        title_height = self._line_height(9.0, 1.4) if title else 0.0
        height = 0.024 + title_height + line_height * len(wrapped) + 0.018
        self._ensure(height)
        bottom = self.y - height + 0.006
        patch = FancyBboxPatch(
            (self.LEFT, bottom),
            self.RIGHT - self.LEFT,
            height - 0.004,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            transform=self.fig.transFigure,
            facecolor="#f3f7fb",
            edgecolor="#b9cee0",
            linewidth=0.8,
        )
        self.fig.patches.append(patch)
        cursor = self.y - 0.010
        if title:
            self.fig.text(
                self.LEFT + 0.018,
                cursor,
                title,
                fontproperties=self.bold,
                fontsize=9.0,
                color="#42657f",
                va="top",
            )
            cursor -= title_height
        for line in wrapped:
            self.fig.text(
                self.LEFT + 0.025,
                cursor,
                line,
                fontproperties=self.regular,
                fontsize=11.0,
                color="#112b3c",
                va="top",
            )
            cursor -= line_height
        self.y -= height + 0.006

    def callout(self, title, text, color="#e8f4ea", edge="#73a580"):
        lines = _wrap_visual(text, 88)
        line_height = self._line_height(10.0)
        height = 0.036 + line_height * len(lines) + 0.018
        self._ensure(height)
        bottom = self.y - height + 0.006
        patch = FancyBboxPatch(
            (self.LEFT, bottom),
            self.RIGHT - self.LEFT,
            height - 0.004,
            boxstyle="round,pad=0.007,rounding_size=0.008",
            transform=self.fig.transFigure,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.0,
        )
        self.fig.patches.append(patch)
        self.fig.text(
            self.LEFT + 0.018,
            self.y - 0.011,
            title,
            fontproperties=self.bold,
            fontsize=11.0,
            color="#234c33",
            va="top",
        )
        cursor = self.y - 0.045
        for line in lines:
            self.fig.text(
                self.LEFT + 0.018,
                cursor,
                line,
                fontproperties=self.regular,
                fontsize=10.0,
                color="#20362a",
                va="top",
            )
            cursor -= line_height
        self.y -= height + 0.008

    def definition(self, key, value):
        lines = _wrap_visual(value, 75)
        line_height = self._line_height(9.8)
        height = max(0.028, line_height * len(lines)) + 0.008
        self._ensure(height)
        self.fig.text(
            self.LEFT + 0.008,
            self.y,
            key,
            fontproperties=self.bold,
            fontsize=9.8,
            color="#315b7d",
            va="top",
        )
        cursor = self.y
        for line in lines:
            self.fig.text(
                self.LEFT + 0.225,
                cursor,
                line,
                fontproperties=self.regular,
                fontsize=9.8,
                color="#202b38",
                va="top",
            )
            cursor -= line_height
        self.y -= height

    def flow(self, steps):
        for index, step in enumerate(steps, start=1):
            lines = _wrap_visual(step, 82)
            line_height = self._line_height(9.7)
            height = max(0.043, line_height * len(lines) + 0.022)
            self._ensure(height + 0.012)
            bottom = self.y - height
            patch = FancyBboxPatch(
                (self.LEFT + 0.055, bottom),
                self.RIGHT - self.LEFT - 0.075,
                height - 0.004,
                boxstyle="round,pad=0.005,rounding_size=0.006",
                transform=self.fig.transFigure,
                facecolor="#f8fafc",
                edgecolor="#a9bfd1",
                linewidth=0.8,
            )
            self.fig.patches.append(patch)
            self.fig.text(
                self.LEFT + 0.018,
                self.y - 0.006,
                str(index),
                fontproperties=self.bold,
                fontsize=12.0,
                color="#ffffff",
                ha="center",
                va="top",
                bbox=dict(boxstyle="circle,pad=0.28", facecolor="#2c6e9b", edgecolor="none"),
            )
            cursor = self.y - 0.008
            for line in lines:
                self.fig.text(
                    self.LEFT + 0.073,
                    cursor,
                    line,
                    fontproperties=self.regular,
                    fontsize=9.7,
                    color="#1d2d3d",
                    va="top",
                )
                cursor -= line_height
            self.y -= height + 0.012

    def cover(self):
        self.page_number += 1
        self.fig = plt.figure(
            figsize=(self.PAGE_WIDTH_IN, self.PAGE_HEIGHT_IN), facecolor="#f8fbfe"
        )
        self.fig.patches.append(
            Rectangle(
                (0.0, 0.86),
                1.0,
                0.14,
                transform=self.fig.transFigure,
                facecolor="#17324d",
                edgecolor="none",
            )
        )
        self.fig.text(
            0.11,
            0.78,
            "UR5e 100 Hz",
            fontproperties=self.bold,
            fontsize=25,
            color="#26547c",
            va="top",
        )
        self.fig.text(
            0.11,
            0.705,
            "位姿约束 Jacobian 控制",
            fontproperties=self.bold,
            fontsize=28,
            color="#17324d",
            va="top",
        )
        self.fig.text(
            0.11,
            0.64,
            "数学原理、QP 约束、离散执行与当前代码映射",
            fontproperties=self.regular,
            fontsize=15,
            color="#52677a",
            va="top",
        )
        callout = FancyBboxPatch(
            (0.11, 0.43),
            0.78,
            0.13,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            transform=self.fig.transFigure,
            facecolor="#e8f4ea",
            edgecolor="#73a580",
            linewidth=1.2,
        )
        self.fig.patches.append(callout)
        self.fig.text(
            0.14,
            0.535,
            "结论",
            fontproperties=self.bold,
            fontsize=13,
            color="#24533a",
            va="top",
        )
        conclusion = (
            "当前实现默认控制完整 TCP 六维位姿：位置采用 Base 系三维误差，姿态采用最短旋转向量误差；"
            "六维几何 Jacobian 同时映射线速度与角速度。只有显式指定 --position-only 时才关闭姿态控制。"
        )
        for line_index, line in enumerate(_wrap_visual(conclusion, 73)):
            self.fig.text(
                0.14,
                0.495 - 0.027 * line_index,
                line,
                fontproperties=self.regular,
                fontsize=11,
                color="#20362a",
                va="top",
            )
        self.fig.text(
            0.11,
            0.31,
            "控制点：attachment_site（当前 MJCF 中与 Cage/抓子根坐标系重合）",
            fontproperties=self.regular,
            fontsize=10.5,
            color="#465a6c",
        )
        self.fig.text(
            0.11,
            0.275,
            "控制周期：Δt = 0.01 s；MuJoCo 物理步长：0.002 s；每周期 5 个动力学步",
            fontproperties=self.regular,
            fontsize=10.5,
            color="#465a6c",
        )
        self.fig.text(
            0.11,
            0.24,
            "方法：时间参数化笛卡尔参考 + 反馈 twist + 带边界最小二乘 QP",
            fontproperties=self.regular,
            fontsize=10.5,
            color="#465a6c",
        )
        self.fig.text(
            0.11,
            0.14,
            "PPO Caging Capture Project",
            fontproperties=self.bold,
            fontsize=10.5,
            color="#17324d",
        )
        self.fig.text(
            0.11,
            0.105,
            "生成日期：{}".format(_datetime.date.today().isoformat()),
            fontproperties=self.regular,
            fontsize=9.5,
            color="#6b7280",
        )
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None


def build_document(output_path):
    document = ChinesePdf(output_path)
    document.cover()

    document.h1("目录与符号约定")
    document.definition("第 1–2 节", "是否控制姿态；Base、TCP、目标坐标关系与 pose6 定义")
    document.definition("第 3–4 节", "位置/姿态的时间参数化，以及闭环六维末端速度命令")
    document.definition("第 5–6 节", "六维几何 Jacobian、带位置/速度/加速度边界的凸 QP")
    document.definition("第 7–8 节", "100 Hz 离散执行、PD 动力学、到达判据与时间可行性")
    document.definition("第 9–11 节", "当前参数、完整数据流、数值性质、限制与改进方向")
    document.spacer(0.018)
    document.h2("主要符号")
    document.definition("q, q_dot, q_ddot", "UR5e 六关节角、关节速度、关节加速度，均按项目固定关节顺序排列")
    document.definition("p, R", "Base 坐标系下 attachment_site 的位置与旋转矩阵")
    document.definition("p_0,R_0 / p_g,R_g", "控制开始时 TCP 位姿 / 目标 TCP 位姿")
    document.definition("p_r,R_r", "由规定总时间 T 生成的当前参考 TCP 位姿")
    document.definition("v, ω", "Base 坐标系表达的 TCP 线速度和空间角速度")
    document.definition("J_v,J_ω", "attachment_site 的平移与旋转几何 Jacobian")
    document.definition("Exp, Log", "SO(3) 指数映射与对数映射；∨ 表示从反对称矩阵取旋转向量")
    document.definition("Δt", "控制周期 0.01 s；不是视觉观测周期 0.05 s")
    document.callout(
        "最重要的范围说明",
        "本 PDF 描述的是当前 constrained_differential_ik 分支。它不同于课程 PPO，也不同于离线 Cage 标签的 EST 评价；但三者使用同一个 Base 坐标系和同一个 attachment_site TCP 契约。",
        color="#fff7e8",
        edge="#d9a441",
    )

    document.h1("1. 当前 Jacobian 是否考虑姿态")
    document.callout(
        "答案：考虑，而且默认开启",
        "配置 constrained_differential_ik.position_only = false。控制器构造六维任务误差 [位置3，姿态3]、六维参考 twist [线速度3，角速度3]，并使用 6×6 的 UR5e TCP 几何 Jacobian。命令行加 --position-only 后才只取 Jacobian 的前三行。",
    )
    document.paragraph(
        "这里的“姿态控制”不是让 Jacobian 自己决定抓子的朝向。目标姿态必须已经包含在目标 TCP pose6 中：例如 Cage 网络输出的 x 轴朝向物体、z 轴沿局部运动平面法向。Jacobian 控制器只负责让实际 attachment_site 追踪该目标姿态。"
    )
    document.equation(
        [
            "完整模式：x_e = [ p ; R ]，V = [ v ; ω ]，J(q) = [ J_v(q) ; J_ω(q) ] ∈ R^(6×6)",
            "位置模式：x_e = p，V = v，J(q) = J_v(q) ∈ R^(3×6)",
        ],
        title="任务维度",
    )
    document.bullet("完整模式同时优化线速度与角速度，因此位置和姿态可能在关节受限时发生权衡。")
    document.bullet("当前权重为位置 1.0、姿态 0.25，位置优先级更高，但姿态并没有被关闭。")
    document.bullet("最终成功判据仍单独检查姿态误差和 TCP 角速度，不会因为 QP 中姿态权重较小就忽略姿态。")

    document.h1("2. 坐标关系与 TCP 位姿误差")
    document.h2("2.1 坐标系")
    document.paragraph(
        "所有位置、旋转、线速度和角速度都表达在 MuJoCo world/Base 右手坐标系中。控制点由 XML 中的 attachment_site 指定。当前 scene_eye_on_hand.xml 中，该 site 与 Cage/抓子根坐标系刚性重合；其 x 轴沿 link6 的伸长方向。"
    )
    document.paragraph(
        "目标 pose6 使用 [x,y,z, r_x,r_y,r_z]。后三维不是欧拉角，而是旋转向量：方向是旋转轴，长度是旋转角（rad）。代码先把旋转向量转换为旋转矩阵，再进行 SO(3) 运算，避免直接相减旋转向量。"
    )
    document.h2("2.2 位置误差")
    document.equation("e_p = p_r − p", title="Base 系位置误差")
    document.paragraph(
        "p_r 是当前时间参考点，不一定等于最终目标 p_g；这样反馈控制追踪的是一条随时间推进的直线，而不是从第一步起直接追最终点。"
    )
    document.h2("2.3 姿态误差")
    document.equation(
        [
            "R_e = R_r R^T",
            "e_R = Log(R_e)∨ ∈ R^3",
            "∥e_R∥ = 当前 TCP 到参考姿态的最短旋转角",
        ],
        title="Base 系空间旋转误差",
    )
    document.paragraph(
        "乘法顺序 R_r R^T 与代码 relative_pose_error6 完全一致。该误差是空间形式，方向表达在 Base 系中，与 MuJoCo 返回的空间角速度 Jacobian 相匹配。对接近 180° 的相对旋转，对数映射存在轴选择敏感性，因此实际 Cage 跟踪应避免单步给出接近 π 的姿态跳变。"
    )

    document.h1("3. 给定总时间的笛卡尔参考轨迹")
    document.h2("3.1 时间归一化")
    document.equation("s(t) = clip(t / T, 0, 1)", title="归一化进度")
    document.h2("3.2 位置直线")
    document.equation(
        [
            "p_r(t) = p_0 + s(t)(p_g − p_0)",
            "v_ff(t) = (p_g − p_0)/T，0 ≤ t < T；t ≥ T 时 v_ff = 0",
        ],
        title="恒速位置参考",
    )
    document.paragraph(
        "T 直接决定参考线速度。T 越短，v_ff 越大；若所需 twist 无法由受限关节速度实现，QP 会得到最接近的可行速度，但规定时间将被判定为不可满足。"
    )
    document.h2("3.3 姿态最短旋转插值")
    document.equation(
        [
            "φ = Log(R_g R_0^T)∨",
            "R_r(t) = Exp([s(t)φ]×) R_0",
            "ω_ff(t) = φ/T，0 ≤ t < T；t ≥ T 时 ω_ff = 0",
        ],
        title="与位置使用同一时间参数的姿态参考",
    )
    document.paragraph(
        "该插值沿 SO(3) 的最短旋转路径推进，不使用欧拉角逐分量插值，因此没有欧拉角万向节锁。位置和姿态共享同一个 s(t)，在理论参考上同时于 T 到达目标。"
    )
    document.callout(
        "当前时间参数化的限制",
        "参考速度在 t=0 从零跳到常数，在 t=T 又从常数跳到零；关节加速度约束和真实 PD 惯性会使 TCP 在 T 时仍有残余速度。因此代码把“位置到达”和“低速稳定”同时检查，并允许额外 settling_time。若必须严格准时且零速到达，应升级为梯形速度或五次 S 曲线。",
        color="#fff3e6",
        edge="#dd8a3c",
    )

    document.h1("4. 闭环六维末端速度命令")
    document.h2("4.1 前馈与反馈")
    document.equation(
        [
            "v_cmd = v_ff + K_p e_p",
            "ω_cmd = ω_ff + K_R e_R",
            "V_cmd = [ v_cmd ; ω_cmd ]",
        ],
        title="参考速度前馈 + 当前误差反馈",
    )
    document.paragraph(
        "前馈项负责按照规定时间推进，反馈项修正 PD 滞后、模型误差和局部数值误差。当前 K_p=5 s^-1、K_R=4 s^-1。它们是任务空间误差反馈增益，而不是 MuJoCo 关节执行器的 PD 增益。"
    )
    document.h2("4.2 twist 限幅")
    document.equation(
        [
            "若 ∥v_cmd∥ > v_max，则 v_cmd ← v_cmd · v_max/∥v_cmd∥",
            "若 ∥ω_cmd∥ > ω_max，则 ω_cmd ← ω_cmd · ω_max/∥ω_cmd∥",
            "当前 v_max = 0.60 m/s，ω_max = 2.0 rad/s",
        ],
        title="方向保持的范数限幅",
    )
    document.paragraph(
        "这里不是逐轴裁剪，而是按向量范数等比例缩放，因此不会改变期望线速度或角速度的方向。限幅发生在 QP 之前，避免给优化器一个明显不可执行的过大任务命令。"
    )

    document.h1("5. UR5e TCP 几何 Jacobian")
    document.h2("5.1 速度映射")
    document.equation(
        [
            "v = J_v(q) q_dot",
            "ω = J_ω(q) q_dot",
            "V = J(q) q_dot，J(q) = [J_v(q); J_ω(q)]",
        ],
        title="几何 Jacobian",
    )
    document.paragraph(
        "机械臂型号确定后 Jacobian 的符号结构固定，但数值随当前 q 改变。代码每个 10 ms 周期调用 MuJoCo mj_jacSite，对 attachment_site 重新计算 J_v 和 J_ω；因此不是在初始姿态算一次后长期复用。"
    )
    document.h2("5.2 为什么不用直接逆矩阵")
    document.bullet("J 可能接近奇异，直接 J^(-1) 会放大速度并导致极大关节命令。")
    document.bullet("关节速度、下一步关节位置和关节加速度共同形成边界，简单逆矩阵无法同时满足。")
    document.bullet("正则化有界最小二乘在不可完全实现 V_cmd 时给出任务误差最小的可行 q_dot。")
    document.paragraph(
        "UR5e 是六自由度，完整位姿任务也是六维，但这并不保证所有构型下 J 都可逆；同时，约束激活后有效自由度会进一步减少。"
    )

    document.h1("6. 带联合约束的凸 QP")
    document.h2("6.1 目标函数")
    document.equation(
        [
            "q_dot* = arg min  ∥ W(J(q)q_dot − V_cmd) ∥² + λ∥q_dot∥²",
            "W = diag(w_p,w_p,w_p,w_R,w_R,w_R)",
            "当前 w_p=1.0，w_R=0.25，λ=0.001",
        ],
        title="加权任务误差 + 关节速度正则",
    )
    document.paragraph(
        "第一项使可实现 TCP twist 接近命令；第二项抑制不必要的大关节速度，并令 Hessian 在 λ>0 时正定。当前姿态权重小于位置权重，所以受限时更优先保持位置跟踪。"
    )
    document.h2("6.2 代码中的有界最小二乘形式")
    document.equation(
        [
            "A = [ WJ ; √λ I ]，b = [ WV_cmd ; 0 ]",
            "min  ∥Aq_dot − b∥²，subject to  l ≤ q_dot ≤ u",
            "H = J^T W^T WJ + λI，f = −J^T W^T W V_cmd",
        ],
        title="与标准凸二次规划等价",
    )
    document.paragraph(
        "代码使用 scipy.optimize.lsq_linear 的 trust-region reflective 方法。因为当前约束都能转换成逐关节上下界，所以不需要一般形式不等式 QP 求解器；数学上仍是凸二次优化。"
    )
    document.h2("6.3 关节速度边界")
    document.equation(
        [
            "−η q_dot,max ≤ q_dot ≤ η q_dot,max",
            "η = 0.70",
            "前三关节约 ±2.198 rad/s；后三关节约 ±4.396 rad/s",
        ],
        title="UR5e 配置速度的 70%",
    )
    document.h2("6.4 关节位置预测边界")
    document.equation(
        [
            "q_safe,min = q_min + m，q_safe,max = q_max − m",
            "(q_safe,min − q_k)/Δt ≤ q_dot,k ≤ (q_safe,max − q_k)/Δt",
            "当前 m = 0.08 rad，Δt = 0.01 s",
        ],
        title="保证下一控制周期不越过安全关节范围",
    )
    document.paragraph(
        "这里使用当前真实 q_k 构造速度边界，而不是等执行完后再裁剪越界结果。它是一步预测安全约束。"
    )
    document.h2("6.5 关节加速度边界")
    document.equation(
        [
            "q_dot,k-1 − q_ddot,max Δt ≤ q_dot,k ≤ q_dot,k-1 + q_ddot,max Δt",
            "前三关节 q_ddot,max=12 rad/s²；后三关节 q_ddot,max=20 rad/s²",
        ],
        title="相邻速度命令的变化率约束",
    )
    document.callout(
        "约束优先级（必须知道）",
        "位置和速度属于硬安全边界。若加速度区间与硬安全区间没有交集，当前实现仅对发生冲突的关节放宽加速度边界，并记录 acceleration_relaxation_count。也就是说，加速度通常是硬约束，但在保护关节位置/速度时允许审计式降级。",
        color="#fff3e6",
        edge="#dd8a3c",
    )

    document.h1("7. 100 Hz 离散执行与关节 PD")
    document.h2("7.1 速度输出积分")
    document.equation(
        [
            "q_cmd,k+1 = clip(q_cmd,k + q_dot,k* Δt, q_min, q_max)",
            "Δt = 0.01 s",
        ],
        title="QP 关节速度到位置伺服目标",
    )
    document.paragraph(
        "QP 输出本质上是关节速度命令。如果底层接受速度命令可直接发送；当前 MuJoCo UR5e 使用位置执行器，所以先积分为 q_cmd，再由 XML 中的关节伺服器执行。q_cmd 是连续积分状态，不是每步都从零生成。"
    )
    document.h2("7.2 MuJoCo 内部动力学")
    document.equation(
        "M(q)q_ddot + C(q,q_dot)q_dot + g(q) = τ_PD + τ_bias + τ_contact",
        title="实际机械臂状态传播（概念形式）",
    )
    document.paragraph(
        "关节位置执行器根据 q_cmd 与实际 q/q_dot 产生力矩，模型惯量、科氏项、重力补偿、执行器力矩范围和接触共同决定真实运动。控制代码每个 10 ms 调用 5 次 0.002 s 的 MuJoCo 动力学步，因此 100 Hz 是控制频率，不是物理积分频率。"
    )
    document.h2("7.3 50 ms 实际 FK 指标")
    document.equation(
        [
            "d_net,50(k) = ∥p_k − p_k-5∥",
            "d_path,50(k) = Σ(i=k-4…k) ∥p_i − p_i-1∥",
        ],
        title="五个 100 Hz 控制步的真实 TCP 移动",
    )
    document.paragraph(
        "p_i 来自每次 PD 和动力学执行后的 attachment_site 正运动学，不是参考轨迹位移，也不是 q_dot* 的理论积分。净位移反映有效跟进能力，路径长度还能暴露振荡和绕行。"
    )

    document.h1("8. 到达判据与时间可行性")
    document.h2("8.1 四个同时成立的条件")
    document.equation(
        [
            "∥p_g − p(T)∥ ≤ 0.005 m",
            "∥Log(R_g R(T)^T)∨∥ ≤ 3°",
            "∥v(T)∥ ≤ 0.020 m/s",
            "∥ω(T)∥ ≤ 0.10 rad/s",
        ],
        title="默认完整位姿成功判据",
    )
    document.paragraph(
        "位置模式会跳过后两项中的姿态误差和角速度要求，但仍检查位置误差和线速度。加入速度条件是为了排除“高速穿过目标点”的假成功。"
    )
    document.h2("8.2 指定时间与收敛阶段")
    document.paragraph(
        "仿真在 t=T 保存 prescribed_time_state 并计算 success_at_prescribed_time。若失败，继续执行 settling_time_s=0.50 s，此时参考前馈速度为零，只保留误差反馈；结束后计算 success_after_settling。两者必须分开报告，不能把延迟收敛冒充准时到达。"
    )
    document.callout(
        "已运行的实际 UR5e 示例",
        "给定 T=2.0 s：在 t=T 时位置误差约 0.069 mm、姿态误差约 0.0117°，但 TCP 线速度约 0.0289 m/s，高于 0.020 m/s，因此准时成功为 false；再收敛 0.5 s 后线速度约 0.00052 m/s，最终成功为 true。该结果说明时间判据正在真实发挥作用。",
    )
    document.h2("8.3 判断 T 不可行的原因")
    document.bullet("参考线速度或角速度先被任务空间限幅。")
    document.bullet("QP 的关节速度、位置或加速度边界激活，使 Jq_dot 无法达到 V_cmd。")
    document.bullet("Jacobian 接近奇异，正则化抑制了极端关节速度。")
    document.bullet("关节 PD、惯性、执行器力矩范围造成动态滞后。")
    document.bullet("目标 pose 本身不可达；使用 --goal-pose 时尤其需要外部可达性检查。")

    document.h1("9. 当前正式参数与代码映射")
    document.definition("控制模式", "position_only=false：默认六维位置+姿态；CLI --position-only 可覆盖")
    document.definition("控制/TCP", "100 Hz；attachment_site；Base 右手系；xyz+旋转向量")
    document.definition("任务反馈", "K_p=5 s^-1；K_R=4 s^-1；v_max=0.60 m/s；ω_max=2.0 rad/s")
    document.definition("QP 权重", "位置 1.0；姿态 0.25；正则化 λ=0.001；最大迭代 100")
    document.definition("关节约束", "速度上限比例 0.70；位置安全边距 0.08 rad；加速度 [12,12,12,20,20,20] rad/s²")
    document.definition("成功阈值", "5 mm、3°、0.020 m/s、0.10 rad/s；额外收敛时间 0.50 s")
    document.definition("核心实现", "cage_system/control/constrained_differential_ik.py")
    document.definition("UR5e 映射", "cage_system/sim/ur5e_control.py；按名称解析六关节、actuator 和 attachment_site")
    document.definition("正式配置", "configs/ur5e_jailer_curriculum.yaml 中 constrained_differential_ik 段")
    document.definition("命令入口", "python -m cage_system.cli run-constrained-jacobian")
    document.spacer(0.016)
    document.callout(
        "为什么姿态权重不是 1.0",
        "线速度单位是 m/s，角速度单位是 rad/s，两者数值尺度不同；W 实质上完成工程尺度归一和优先级设置。0.25 是当前选择，不是普适常数。应在验证集上同时观察位置误差、姿态误差、饱和率和时间成功率，再标定权重。",
        color="#eef3fb",
        edge="#7f9fca",
    )

    document.h1("10. 每个控制周期的数据流")
    document.flow(
        [
            "输入控制开始 TCP pose (p_0,R_0)、目标 TCP pose (p_g,R_g)、规定时间 T、当前 q/q_dot 和上一关节速度命令。",
            "计算 s(t)，生成直线位置参考 p_r、最短旋转参考 R_r，以及前馈线速度 v_ff 和角速度 ω_ff。",
            "从最新 MuJoCo 状态读取 attachment_site 的实际 p、R，计算 e_p=p_r−p 和 e_R=Log(R_r R^T)∨。",
            "构造 V_cmd=[v_ff+K_p e_p；ω_ff+K_R e_R]，然后分别对线速度和角速度做范数限幅。",
            "根据最新 q 调用 mj_jacSite，得到 attachment_site 的 J_v、J_ω；位置模式仅保留 J_v。",
            "把关节速度、一步位置预测和加速度限制合并成逐关节边界 l、u。",
            "求解 min∥Aq_dot−b∥²，得到满足边界且最接近 V_cmd 的 q_dot*；记录 QP 状态、成本和求解时间。",
            "将 q_dot* 按 0.01 s 积分为 q_cmd；关节位置执行器推进 5 个 MuJoCo 物理步。",
            "重新读取真实 TCP FK/Jacobian 速度，记录误差、接触、50 ms 位移，并在 t=T 检查时间成功。",
        ]
    )

    document.h1("11. 数学性质、已知限制与建议")
    document.h2("11.1 数学与数值性质")
    document.bullet("λ>0 时目标函数严格凸；给定非空边界区间，关节速度解唯一。")
    document.bullet("约束是逐关节 box bounds，故 bounded least squares 与标准 QP 等价。")
    document.bullet("每周期重算 J 和反馈误差，使局部线性化通过闭环连续更新，而不是一次线性化走完整条路径。")
    document.bullet("SO(3) 对数误差保证一般情况下沿最短角度纠正姿态，不存在欧拉角万向节锁。")
    document.h2("11.2 当前没有包含的内容")
    document.bullet("QP 尚未加入环境/自碰撞距离约束；仿真会记录 contact，但不能提前绕障。")
    document.bullet("QP 未显式加入执行器力矩和关节 jerk 约束；力矩上限只由 MuJoCo actuator 动力学体现。")
    document.bullet("目标 pose 可达性不由 --goal-pose 自动证明；--goal-joints 的 FK 目标具有已知关节解，更适合验证。")
    document.bullet("恒速时间参数化不保证 t=T 零速度，严格定时任务宜采用梯形或 S 曲线。")
    document.bullet("当前为单一目标的局部闭环微分 IK，不是 MoveIt 那样的全局无碰撞路径规划器。")
    document.h2("11.3 推荐增强顺序")
    document.bullet("第一优先：将直线恒速参考升级成满足端点零速度/零加速度的五次时间标度。")
    document.bullet("第二优先：加入距关节限位的连续速度阻尼，而不仅是一周期硬边界。")
    document.bullet("第三优先：用距离 Jacobian 把自碰撞和环境距离写成线性速度不等式。")
    document.bullet("第四优先：对 T 做预可行性检查，依据路径长度、twist 上限和 Jacobian 可实现速度给出最小建议时间。")
    document.bullet("实机部署：保持相同 attachment_site 标定，并用实际 ros_control 速度/位置接口复现相同 100 Hz 契约。")

    document.h1("12. 与 Cage 和课程 PPO 的关系")
    document.definition("Cage 网络", "从五帧因果视觉历史预测 Base 系目标 TCP pose6；不负责求关节命令")
    document.definition("约束 Jacobian", "经典闭环基础控制器；输入当前目标 TCP pose6 和 q/q_dot，输出受约束 q_dot")
    document.definition("JAiLeR PPO", "学习型反馈控制器；同样使用 TCP误差6+q6+q_dot6，但由策略输出归一化关节期望速度")
    document.definition("EST", "仅用于离线 Cage 候选动力学评价和标签选择，不进入 Jacobian QP")
    document.paragraph(
        "在线连接时，20 Hz Cage 网络每 50 ms 更新一次目标 TCP pose；100 Hz Jacobian 可在这 50 ms 内运行 5 次，每次使用最新 q/q_dot 和实际 TCP 重新闭环计算。目标在 5 个控制步内保持不变，不需要也不允许输入真实未来轨迹。"
    )
    document.callout(
        "最终理解",
        "当前 Jacobian 分支确实考虑姿态。它不是先算位置再附带一个姿态补丁，而是在同一个六维 twist、同一个六维几何 Jacobian和同一个受约束凸优化中联合处理位置与姿态；同时通过较小的姿态权重表达“位置优先、姿态继续跟踪”的工程取舍。",
    )
    document.h2("参考来源")
    document.bullet("用户文件：Jacobian_约束微分逆运动学_算法流程.pdf。")
    document.bullet("D. E. Whitney, Resolved Motion Rate Control of Manipulators and Human Prostheses, 1969。")
    document.bullet("F. Flacco, A. De Luca, O. Khatib, Motion/Control of Redundant Robots under Hard Joint Constraints, ICRA 2012 / T-RO 2015。")
    document.bullet("K. M. Lynch, F. C. Park, Modern Robotics：Velocity Kinematics、Trajectory Generation 与 Motion Control。")

    document.close()
    return document.output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = build_document(args.output)
    print(str(output))


if __name__ == "__main__":
    main()
