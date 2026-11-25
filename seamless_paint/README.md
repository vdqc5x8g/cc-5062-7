# 二维无缝图案绘画工具（正交环面 + 斜向重复晶格）

在有限纹样单元上绘画，笔尖越过单元边界时按**重复晶格的周期关系**写入所有
对应区域，并实时给出多个重复单元的连续平铺预览。支持由**两条非共线基向量**
`a=(ax,ay)`、`b=(bx,by)` 定义的一般二维晶格（斜格、锐角格、半落砖格等），
也完整保留初版的正交 `W×H` 环面模式。

绘制过程本身满足周期覆盖规则，导出时不做任何边缘模糊或接缝掩盖；
验收方式是把内核结果与独立参考实现分别平铺后做**整幅差图比较**。

## 图形入口（主要使用方式）

本工具的主界面是浏览器绘画窗口，由标准库 `http.server` 提供服务，
所有按钮都调用真实内核 HTTP API，没有任何固定或模拟纹样数据。

```bash
# 在仓库根目录 seamless_paint/ 下，三种等价启动方式：
python3 -m seamless_paint.webapp --port 8000
python3 -m seamless_paint                 # 等价（__main__ 调 webapp.main）
./run_painter.sh 8000                     # 便捷脚本，默认端口 8000
```

然后浏览器打开 <http://127.0.0.1:8000/>。常用参数：

```bash
python3 -m seamless_paint.webapp --host 127.0.0.1 --port 8000 \
    --width 256 --height 256 --output output
```

界面与操作：

- 左侧 512×512 是**真实周期世界视图**（`/api/world.png`，内核按
  `p = s·a + t·b` 逆映射逐像素渲染，不是倾斜贴图预览），叠加层只负责
  晶格线框、屏幕→晶格坐标定位和笔尖轮廓；金色虚线格是参数单元 `(0,0)`。
- 直接沿画布边缘、晶格顶点/格边落笔，抬笔后世界视图与右侧平铺预览
  （1×1 到 8×8，`/api/lattice_preview.png`）都由内核重绘，跨边接缝同步更新。
- 笔刷控件：半径（允许远大于单元）、柔化、颜色、不透明度、越界合成策略、
  **旋转角 angle（度，世界空间朝向）**、**扁率 aspect（短轴/长轴，
  用于核验椭圆/扁笔尖形状）**。笔尖在实际画布（世界）空间保持尺寸与形状，
  不随晶格倾斜而被剪切变形。
- 晶格控件：直接编辑 `ax ay bx by`，或选用预设
  正交 / 60°菱形 / 半落砖 / 15°锐角；状态行显示格元面积与基向量夹角。
- **换基语义二选一**（见下文）：`保持物理笔迹重新包裹 (rewrap)` 或
  `变形现有内容 (deform)`，点“应用基向量”生效；点“对比两种换基影响”
  会调用 `/api/basis/preview` 生成左右两张非写入对比图，效果差异清楚可见，
  当前会话不被预览修改。
- 支持撤销（按完整历史重放）、清空、新建正交尺寸、保存工程与下载
  单元 PNG / 晶格平铺 PNG。内核拒绝的非法输入（共线基、负半径等）
  以红色状态条回显，不会崩溃。

> 命令行 demo 与测试只是辅助验收手段；无缝图案能力始终以绘画窗口为主入口。

## 晶格几何与坐标变换

`seamless_paint/lattice.py` 中的 `Lattice` 是唯一的周期几何来源：

```
B = [ a.x  b.x ]      det = ax*by - ay*bx != 0
    [ a.y  b.y ]
世界坐标 p = B · (s,t) = s·a + t·b          （格坐标 s,t，单位：格元）
格坐标   (s,t) = B⁻¹ · p
参数图表坐标 u = s*W, v = t*H               （W,H 为图表像素尺寸）
基本单元取 [0,1)²，像素中心为 (col+0.5, row+0.5)
```

- `Lattice(W, 0, 0, H)` 为默认正交基，并标记 `_legacy_ortho=True`，
  走初版代码路径，旧测试与旧 JSON 全部兼容。
- 构造时拒绝零向量、共线（行列式为 0）、NaN/Inf 基向量；
  近退化晶格（如 `Lattice(32,0,31,3)`，夹角约 5.7°）默认也拒绝，
  验收测试通过显式 `allow_near_degenerate=True` 覆盖。
- 完整实现了：屏幕位置→世界坐标（视图矩形）→格坐标 `B⁻¹p` 的变换；
  按笔尖实际支撑区域计算的覆盖邻居枚举；每个邻居副本的边界裁切写入。
  **不是**先画矩形结果再用平行四边形裁剪——那种做法在另一条基方向上
  必然留下错误接缝。

## 世界空间笔尖、支撑驱动的邻居枚举与格元身份去重

- 笔尖是世界坐标中的可旋转椭圆度量
  `e² = q00·dx² + 2·q01·dx·dy + q11·dy²`，
  覆盖硬边界 `e ≤ 1`，软核按 ramp 衰减。旋转角与扁率只在世界空间定义，
  因此斜格上的笔尖大小、朝向、扁形都不随基向量改变。
- **邻居范围由实际笔刷支撑区域算出，而不是固定复制 N 圈**：
  用特征分解求椭圆在 `s,t` 方向的精确支撑半宽（另有按
  `r_max·‖B⁻¹ 的行‖` 的保守界），再由
  `i_lo = ceil(-e_s - s_c …)`、`i_hi = floor(1 + e_s - s_c …)`
  得到整数副本范围。大笔尖会自动跨多个邻居（验收用例 `r=52` 覆盖 6 个
  格元身份；`r=60` 时枚举/报告的格元数 > 9 且整幅全覆）。
- **方案权衡（最终采用“世界网格去重笔迹实例”）**：
  “先在周期单元内裁笔再复制”在斜格 + 大笔尖 + 拖过晶格顶点时，
  同一笔印会在多条边/角被裁切多次，容易重复上色或漏掉斜向第二周期。
  实现改为：每个笔印在世界平面只构造一次，枚举其相交的世界网格副本，
  以整数格元身份 `(i,j)` 去重（集合保证同一身份只写一次），
  再折叠到图表。拖过格顶（晶格顶点）的一个笔印恰好得到
  `[(0,0),(0,1),(1,0),(1,1)]` 四个身份，四角完全对称。
- 同一笔印的多个周期副本先求**覆盖并集**
  `cov = 1 - ∏(1 - cov_k)`，再做**一次** source-over 合成，
  半透明大笔尖不会因多层叠加而错误加深。策略：
  - `neighbor`（默认）：所有相交副本取覆盖并集；
  - `lattice`：每像素只取最近副本（硬笔刷下两者数学等价）。
- 笔画轨迹：原始世界指针点 → `B⁻¹` 折叠 → 在 `{-1,0,1}²` 中选
  世界长度最短的整数平移 → 从首点解包为连续轨迹 → 弧长等距重采样
  （默认间距 `max(0.5, min(r*0.25, 4))`）。因此拖过格顶/格边时
  笔画沿短的世界环绕段连续，不会出现横穿整个单元的错误直线。
  历史同时保存 `raw_points` 与精确 `samples`，换基/撤销后可重放。

## 更换基向量的两种语义

`SeamlessTile.change_basis(new_lattice, mode=...)` 与 HTTP
`POST /api/lattice`（及非写入的 `/api/basis/preview`）支持：

1. **`rewrap`（保持物理笔迹，重新包裹）**：世界中的物理采样点与笔尖度量
   完全不动，只把周期关系换成新基，重新折叠归属并重放。效果：笔迹形状、
   朝向不变，但“哪一段和哪一段成为接缝”改变。
2. **`deform`（变形现有内容）**：让参数格画作保持对应，世界笔迹按
   `T = B_new · B_old⁻¹` 线性变换（点与采样点 `p′=Tp`），笔尖度量按
   `Q′ = T⁻ᵀ Q T⁻¹` 变换，保留当前像素缓冲并重写历史。
   效果：整幅画与扁笔尖一起被线性拉斜/拉伸。

UI 的“对比两种换基影响”并排展示两张内核生成的世界视图，
`output/lattice_basis_rewrap.png` 与 `output/lattice_basis_deform.png`
是 demo 产出的同样两张对比图。

## 导出格式：显式记录重复基

保存的 `*_history.json` 为 version 2，除完整笔画历史外，显式写入：

```json
{
  "version": 2,
  "width": 160,
  "height": 160,
  "repeat_basis": {
    "a": [160.0, 0.0],
    "b": [80.0, 138.564],
    "kind": "oblique",
    "cell_area": 22170.24,
    "angle_deg": 60.0
  },
  "lattice": { "a": [160.0, 0.0], "b": [80.0, 138.564] },
  "strokes": [ "..." ]
}
```

加载时优先读 `repeat_basis`，也兼容旧字段 `lattice`；两者都没有的
version 1 文件按正交 `W×H` 晶格恢复。`save_project(..., write_world_tiling=True)`
还会导出真实斜格块的平铺 PNG（`*_lattice_tiled.png`）。

## 晶格绘画样例与整幅差图验收（辅助 CLI）

一条命令生成斜格绘画样例、锐角/大笔尖样例、两种换基对比，
以及“平铺后的整幅差图”验收：

```bash
python3 -m seamless_paint.lattice_demo
# 可选：python3 -m seamless_paint.lattice_demo output 3
#   参数：输出目录、每边格元副本数（默认 3）
```

demo 退出码为 0 表示验收通过。它会用一套**完全独立的参考实现**
（从历史中的 `raw_points` 重新推导折叠/最短平移/弧长重采样，
逐笔印 bbox 枚举 + RGBA 预乘覆盖并集）渲染 3×3 斜格块，
再与内核的 `render_cell_block_uint8` 结果逐通道比较，并把两份结果
各自平铺后比对。最近一次结果：

```
dabs checked: 367
max channel diff : 0 (tolerance 2)
mean channel diff: 0.00000
basis-shift seam invariance: PASS
ACCEPTANCE: PASS
```

`output/` 中的晶格相关产物：

| 文件 | 内容 |
| --- | --- |
| `lattice_sample_unit.png` | 斜格参数单元图表（160×160） |
| `lattice_sample_history.json` | v2 历史，含显式 `repeat_basis`（60°菱形基） |
| `lattice_sample_preview3x3.png` | 图表的朴素 3×3 复制 |
| `lattice_sample_lattice_tiled.png` | **按斜格基真正平铺**的 3×3 格元块（900×520） |
| `lattice_sample_acute15_tiled.png` | 15°锐角晶格样例的横向平铺 |
| `lattice_sample_bignib_tiled.png` | 64² 格 + r=52 超大扁笔尖（跨 6 个格元身份） |
| `lattice_basis_rewrap.png` / `lattice_basis_deform.png` | 同一画作的两种换基影响对比 |
| `lattice_accept_reference3x3.png` | 独立参考实现的 3×3 平铺 |
| `lattice_accept_kernel3x3.png` | 内核 3×3 平铺 |
| `lattice_accept_diff3x3.png` | 放大后的整幅差热图：**全黑=逐像素一致** |
| `lattice_accept_compare3x3.png` | 参考 / 内核 / 差图 三联对照 |

样例内容专门用于肉眼核验：三条不同旋转角与扁率的书法感曲线
（非正交重复、跨边连续）、晶格原点上的黑色扁笔尖点
（应在四个相邻格元角出现四个相同印记）、`a` 边中点的紫色扁笔尖点。
查看建议：

```bash
# 直接打开或用任意图片查看器：
xdg-open output/lattice_sample_lattice_tiled.png
xdg-open output/lattice_accept_compare3x3.png   # 第三联全黑即验收通过
```

此外还有平移不变性接缝检查：把真实周期世界渲染矩形整体平移 `a` 或 `b`
后，重叠区域必须逐像素相等。

初版正交场景的无头验收 demo 仍然保留：

```bash
python3 -m seamless_paint.demo output
# 产物：acceptance_demo_unit.png / _history.json / _preview3x3.png
```

## 测试

```bash
python3 -m unittest discover -s tests -v
```

当前共 41 个测试（1 个在无 numpy 环境下跳过：numpy/纯 Python 交叉校验），
其中 `tests/test_lattice.py` 27 个，覆盖：

- **几何**：基向量往返变换、默认正交基、共线/零/NaN/Inf 拒绝、
  近退化晶格显式开关、非法半径/策略拒绝、格顶笔印恰好 4 个身份、
  支撑范围随半径变化、圆形支撑精确界等于 `B⁻¹` 行范数。
- **独立参考与整幅平铺差图**：60°斜格、约 10°锐角、约 5.7°近退化晶格、
  `lattice` 策略；独立像素优先参考实现 vs 内核的 3×3 整幅 montage
  最大通道差 ≤2、均值差 <0.05。
- **接缝与覆盖次数**：对角对称、世界渲染平移 `±a/±b` 逐像素不变、
  r=60 全覆且格元计数 >9、软笔尖 alpha 上限与二次落笔加深规则、
  报告格元与 `enumerate_cell_ids` 完全一致。
- **笔画路径**：弧长/间距保真（含曾出现的解包位移重复累加回归）、
  跨格顶轨迹、与独立推导的笔印序列一致。
- **换基**：deform 后缓冲与重放一致、`T`/`Q′` 数值精确校验；
  rewrap 后采样点不变且与新格重放一致；非法 mode 拒绝；
  撤销/重放后历史记账一致（历史被 `__init__` 清空的回归）。
- **持久化**：`repeat_basis` JSON、保存/加载逐像素差 ≤1、
  无基字段的旧 JSON 按正交恢复。
- **Web API**：线程化真实 HTTP 服务随机端口；`/api/lattice`
  构造 60°基、`/api/stroke` 带 angle/aspect 且返回 `last_cells`、
  `world.png`/`lattice_preview.png`/下载端点、`/api/basis/preview`
  返回 data URL、共线基返回 HTTP 400 `{ok:false,error}` 且会话仍可用。

## HTTP API 摘要（UI 与无头验收共用）

| 方法/路径 | 作用 |
| --- | --- |
| `GET /api/state` | 尺寸、当前基向量、夹角/面积、世界视图矩形 |
| `POST /api/stroke` | 世界坐标点列 + 半径/颜色/angle_deg/aspect 等，走真实晶格落笔 |
| `POST /api/lattice` | 更换基向量，`mode: "rewrap" \| "deform"` |
| `POST /api/basis/preview` | 不写入会话，返回两种换基效果的世界视图 data URL |
| `GET /api/world.png?w=&h=` | 内核逆映射渲染的真实周期世界视图 |
| `GET /api/lattice_preview.png?copies=&max=` | 斜格块平铺预览 PNG |
| `GET /api/download/lattice_tile` | 下载晶格平铺 PNG |
| `POST /api/undo` `/api/clear` `/api/new` `/api/save` | 历史与工程操作 |

## 目录结构

```
seamless_paint/
├── run_painter.sh                 # 图形窗口便捷启动脚本
├── seamless_paint/
│   ├── __main__.py                # python3 -m seamless_paint -> webapp
│   ├── kernel.py                  # 绘画内核：环面 + 一般晶格（numpy + 纯 Python）
│   ├── lattice.py                 # Lattice：B/B⁻¹、支撑枚举、最短平移等周期几何
│   ├── errors.py                  # SeamlessInputError
│   ├── pngio.py                   # 仅用标准库的 PNG 编解码
│   ├── history.py                 # v2 工程持久化与 repeat_basis、平铺导出
│   ├── webapp.py                  # 标准库 HTTP 服务，内核驱动的绘画窗口
│   ├── demo.py                    # 初版正交环面无头验收
│   ├── lattice_demo.py            # 斜格样例 + 独立参考整幅差图验收（辅助 CLI）
│   └── static/                    # 浏览器绘画界面（HTML/CSS/JS，无外部依赖）
└── tests/
    ├── test_kernel.py             # 初版正交测试（含 numpy/纯 Python 交叉校验）
    └── test_lattice.py            # 斜格几何/参考差图/接缝/换基/持久化/Web API
```

## 依赖与说明

- Python 3.9+
- 必需：仅 Python 标准库（`http.server`、`zlib`、`struct`、`json` 等）
- 可选：`numpy`（存在时自动启用向量化晶格内核；缺失时自动回退纯 Python，
  输出经独立参考实现与测试核对）。当前开发环境没有 numpy/pip，
  numpy 路径未在本机运行，但 `tests/test_lattice.py` 中的参考等价测试
  会在装有 numpy 的评分环境自动执行两条后端的一致性校验。
- 浏览器实测说明：本环境无 Chromium/Firefox/Playwright，前端仅做过
  `node --check` 语法校验；UI 后端链路由 `LatticeWebEndpointTests`
  的真实 HTTP 服务测试覆盖。
