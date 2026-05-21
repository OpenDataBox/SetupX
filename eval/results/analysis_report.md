# Ours vs R2R 系统性对比分析报告

> 生成时间: 2026-03-16
> 数据来源: results/ours_92_final.jsonl, results/r2r_44_eval.json
> 评估标准: Phase 2 诉讼模型（Prosecutor + Judge）

---

## 1. 整体通过率

| 系统 | 仓库数 | not_guilty | guilty | 其他 | 通过率 |
|------|--------|-----------|--------|------|--------|
| Ours | 92 | 72* | 13* | 7 (setup_failed=4, no_result=3) | 78.3%* |
| R2R  | 44 | 29 | 10 | 5 (构建失败) | 65.9% |

*修正后数据：kedro-org/kedro 由 guilty 翻转为 not_guilty（见第5节）

### Ours 92 仓库 verdict 分布

- not_guilty: 71 → 修正后 72
- not_guilty_after_prosecution: 1（检察官起诉但法官判无罪）
- guilty: 13 → 修正后 12
- setup_failed: 4（Agent 未在 50 步内完成）
- no_result: 3（astropy/reproject, skmin-lab/unixmd, tefra/xsdata 无结果）

### R2R 44 仓库 verdict 分布

- not_guilty: 29
- guilty: 10
- 构建失败（Dockerfile build 失败）: 5

---

## 2. 同一 44 仓库 Head-to-Head 对比

R2R 的 44 个仓库是 Ours 92 的子集，可以直接对比。

| 对比类别 | 数量 | 占比 |
|----------|------|------|
| 双方都通过 (both_pass) | 23 | 52.3% |
| 仅 Ours 通过 (ours_pass_r2r_fail) | 14 | 31.8% |
| 仅 R2R 通过 (ours_fail_r2r_pass) | 6 → 修正后 5 | 11.4% |
| 双方都失败 (both_fail) | 1 | 2.3% |

**在同一 44 仓库上:**
- Ours: 37/44 = 84.1% → 修正后 38/44 = 86.4%
- R2R: 29/44 = 65.9%
- **Ours 净领先: +9 个仓库 (+20.5 percentage points)**

### 仅 Ours 通过的 14 个仓库（Ours 优势所在）

| 仓库 | R2R 失败原因 |
|------|-------------|
| dj-stripe/dj-stripe | R2R guilty |
| python-poetry/poetry | R2R guilty |
| castagnait/plugin.video.netflix | R2R guilty |
| embeddings-benchmark/mteb | R2R guilty |
| guardrails-ai/guardrails | R2R guilty |
| jacebrowning/memegen | R2R guilty |
| mampfes/hacs_waste_collection_schedule | R2R guilty |
| piccolo-orm/piccolo | R2R guilty |
| posit-dev/great-tables | R2R guilty |
| spec-first/connexion | R2R guilty |
| nonebot/nonebot2 | R2R Dockerfile 构建失败 |
| huggingface/datasets | R2R Dockerfile 构建失败 |
| mopidy/mopidy | R2R Dockerfile 构建失败 |
| roboflow/supervision | R2R Dockerfile 构建失败 |

> 分析：14 个中有 4 个 R2R Dockerfile 构建直接失败（说明一次性生成 Dockerfile 的脆弱性），
> 10 个 R2R guilty 说明 R2R 的 Dockerfile 虽然能构建但装的不完整。

### 仅 R2R 通过的 6 个仓库（Ours 劣势/需分析）

| 仓库 | Ours 失败原因 | 说明 |
|------|-------------|------|
| **kedro-org/kedro** | guilty: import gitpython | **已确认误判**（应为 import git） |
| amperser/proselint | guilty: 未安装 google-re2 | google-re2 需要系统级 re2 库，pip install 常失败 |
| beeware/briefcase | guilty: GitPython 未装 + setuptools 不兼容 | GitPython 指控误判，setuptools 指控可能真实 |
| getsentry/sentry-python | guilty: 未以 editable 模式安装 | Agent 策略问题 |
| piskvorky/smart_open | guilty: 缺 backports.lzma 等 | 可选依赖相关 |
| platformio/platformio-core | guilty: 缺 pyserial | 依赖声明在非标准位置 |

---

## 3. XPU 经验知识库效果

### 3.1 基本统计

| 指标 | 数值 |
|------|------|
| 数据库 XPU 条目 | 163 条（手工标注 + 在线提取） |
| 离线提取 XPU | 118 条（来自本轮 92 仓库轨迹） |
| TRY_XPU_SUGGESTION 总执行次数 | 548 次 |
| 推测执行成功率 | 99.6%（546/548，仅 2 次 rollback） |
| 使用过 XPU 建议的仓库 | 78/92 = 84.8% |

### 3.2 XPU 使用强度与通过率

| XPU 使用次数 | 仓库数 | 通过率 |
|-------------|--------|--------|
| 0 次 | 14 | 85.7% |
| 1-3 次 | 39 | 79.5% |
| 4-10 次 | 30 | 76.7% |
| 11+ 次 | 9 | 66.7% |

### 3.3 步数统计

| 分组 | 平均步数 |
|------|----------|
| 使用 XPU | 25.3 步 |
| 未使用 XPU | 14.6 步 |

### 3.4 关于 XPU 效果的讨论

**表面矛盾**：使用 XPU 越多，通过率反而越低。

**解释**：这是典型的 confounding variable（混淆变量）问题。XPU 查询由 last_error 触发——
仓库越难（依赖复杂、构建错误多）→ 触发越多错误 → 越多 XPU 查询 → 但仍然难以通过。
因果关系是「难度 → 多 XPU + 低通过率」，不是「多 XPU → 低通过率」。

**99.6% 推测执行成功率的含义**：XPU 给出的建议几乎都是可执行且不需要 rollback 的，
说明 XPU 建议质量高。但「执行成功」不等于「解决了当前问题」——可能建议是正确的操作但
不足以解决全部问题。

**缺失的 ablation study**：要严格证明 XPU 的因果贡献，需要在同一批仓库上做
XPU-enabled vs XPU-disabled 对比实验。当前数据只能证明 XPU 建议质量高（低 rollback），
不能证明通过率提升是 XPU 带来的。

> **建议**：如果要在论文中声称 XPU 的贡献，应当补充 ablation study，
> 或者至少谨慎表述为"XPU 提供了高质量的可行建议（99.6% 成功率），
> 覆盖了 84.8% 的仓库"，而不直接声称提升了通过率。

---

## 4. envBench 评估（进行中，数据不完整）

已完成 61/100 个仓库，但发现两个系统性 bug 导致结果不可信，已停止：

### Bug 1: 环境变量不持久（envBench 独有）

envBench 的 bootstrap 脚本通过 `source` 在单个 shell 里执行，
`conda activate`、`pyenv global`、`export PATH=...` 的效果不持久到检察官的 `docker exec`。
检察官用的是系统默认 python3，不是 bootstrap 配置好的那个。

- 影响范围：60/100 仓库使用 pyenv，7/100 使用 conda
- 14/61 已完成仓库因 bootstrap 脚本超时（600s）无结果

**为什么 Ours 和 R2R 不受影响**：
- Ours: Agent 用系统 python + pip install，不涉及 conda/pyenv
- R2R: Dockerfile 把依赖固化到镜像层，容器启动即最终状态

### Bug 2: pip 包名 ≠ import 名（通用问题）

检察官从 pyproject.toml 读到 pip 包名后直接做 `import`，但很多包的
pip 安装名和 Python import 名不同：

| pip 包名 | 正确 import 名 |
|-----------|----------------|
| beautifulsoup4 | bs4 |
| GitPython | git |
| attrs | attr |
| Pillow | PIL |
| PyYAML | yaml |
| SecretStorage | secretstorage |

- envBench: 确认影响 9 个仓库
- R2R: 确认影响 1 个仓库 (mampfes/hacs_waste_collection_schedule)，但有其他真实指控，verdict 不变
- Ours: 确认影响 4 个仓库，其中 **kedro-org/kedro 需要翻转**（唯一指控为误判），其余 3 个有其他真实指控

---

## 5. 已确认的修正

### 5.1 Ours: kedro-org/kedro guilty → not_guilty

- 原指控：「未安装核心依赖 gitpython」（唯一指控）
- 日志证据：检察官执行 `python3 -c "import gitpython"` 失败
- 实际情况：pip 包名 GitPython 的 import 名是 `git`，不是 `gitpython`
- 检察官日志原文："所有核心依赖验证完毕，只有 gitpython 不可导入"
- 修正：verdict 应为 not_guilty

修正后 Ours: 72/92 = 78.3%（原 71/92 = 77.2%）

### 5.2 R2R: 无需修正

mampfes/hacs_waste_collection_schedule 有 beautifulsoup4 和 PyYAML 两条误判指控，
但去除后仍有 3 条真实指控，guilty verdict 不变。

---

## 6. 待办事项

1. **修复 Bug 2**：在检察官 prompt 中加入 pip 包名 → import 名映射提醒，三家统一重新评估
2. **修复 envBench Bug 1**：bootstrap 执行后捕获环境变量，注入后续 exec_run
3. **增加 bootstrap 超时**：600s 对 PyTorch/flash-attn 等重型安装不够，考虑 1200s
4. **envBench 重跑**：修复后重跑 100 个仓库
5. **XPU Ablation Study**：关闭 XPU 在同一批仓库上重跑，对比通过率差异（代价大，待决策）
6. **结果文件修正**：更新 ours_92_final.jsonl 中 kedro 的 verdict

---

## 附录: 数据文件说明

| 文件 | 内容 | 仓库数 |
|------|------|--------|
| results/ours_92_final.jsonl | 我们系统的 92 仓库评估结果 | 92 |
| results/r2r_44_eval.json | R2R baseline 的 44 仓库重新评估结果 | 44 |
| results/envbench_100_eval.json | envBench baseline 评估（不完整，有 bug） | 61/100 |
| data/offline_xpu_extracted.jsonl | 从 92 仓库轨迹离线提取的 XPU 经验 | 118 条 |
