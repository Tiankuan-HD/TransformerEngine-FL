# vendor NPU 后端两处修复说明（`test/all-v2`）

本分支 = `test/all` @ `2371a595` **原样** + 两个提交，只动一个文件：
`transformer_engine/plugin/core/backends/vendor/npu/npu.py`，共 **+4 / −42**。

| 提交 | 改什么 |
|---|---|
| `fix(vendor/npu): pass state_steps as float32 to fused_adamw` | `multi_tensor_adam` 的 `state_steps` dtype |
| `fix(vendor/npu): delegate multi_tensor_scale_tensor to the vendor op` | `multi_tensor_scale_tensor` 改为调用厂商原生算子 |

## 怎么发现的

把七个算子用 `TE_FL_PER_OP` 钉死到 `vendor:NPU`，**不留回退余地**：

```
TE_FL_PER_OP='multi_tensor_adam=vendor:NPU;rmsnorm_fwd=vendor:NPU;rmsnorm_bwd=vendor:NPU;
              get_flash_attention_class=vendor:NPU;multi_tensor_l2norm=vendor:NPU;
              generic_gemm=vendor:NPU;multi_tensor_scale_tensor=vendor:NPU'
```

默认的 `flagos → vendor → reference` 自动回退链会把这两个失败掩盖掉，**常规跑法碰不到**。
所以「平时能跑通」是正常的 —— 这两个 vendor 实现平时根本不会被调用。

> ⚠️ 判断「配置生效」还是「失败回退」**不能只看 `using` 行**。首次调用就失败时
> `last_impl_id` 被 `_invalidate_cache` 清成 `None`，不会打印 `fallback to`，
> 两种情况的 `Op 'xxx' using 'vendor.npu'` 长得一模一样。要看前面有没有
> `Implementation '...' failed for op ...` 这行 WARNING。

---

## 修复一 · `multi_tensor_adam` 的 `state_steps` 传了 `int64`

**改前**（`npu.py:575`）：

```python
step_tensor = torch.tensor(
    step,
    dtype=torch.int64,          # ← torch_npu 要 float32
    device=group["params"][0].device,
)
fused_adamw(..., [step_tensor] * len(group["params"]), ...)
```

**报错**：

```
_fused_adamw_: state_steps[0] must be a one-element float32 tensor
```

**改后**：`dtype=torch.int64` → `dtype=torch.float32`。就这一行。

元素个数本来就是对的（0 维张量 `numel() == 1`），**只有 dtype 不对**。

> 厂商包里**没有** `multi_tensor_adam`，所以这里走 torch_npu 的 `fused_adamw` 是对的，
> `del chunk_size` 对 adam 而言也成立。只是那句注释
> `# CUDA launch partitioning has no NPU equivalent.` 说法不准确 ——
> 昇腾是有的，而且更硬（见修复二）。本分支没有动那行注释。

---

## 修复二 · `multi_tensor_scale_tensor` 绕开了厂商原生算子

**改前**：`npu.py:369` 的函数体没有调用厂商算子，而是用
`torch._amp_foreach_non_finite_check_and_unscale_` 手工实现了一遍，
**`chunk_size` 只出现在签名里、函数体内从未使用** —— 整个梯度列表一次下发。

**报错**（每个走到这个算子的 rank 都有）：

```
Last implementation 'vendor.npu' failed for op 'multi_tensor_scale_tensor':
  ... the current working operator name is
      _amp_foreach_non_finite_check_and_unscale_kernel
KernelLaunch failed because value 310400 for parameter coreDim is invalid.
Expected value: less than or equal to 65535.
```

调用栈：

```
Megatron optimizer.step()
  → clip_grads.clip_grad_by_total_norm_fp32  (clip_grads.py:215)
    → multi_tensor_applier                   (multi_tensor_apply.py:21)
      → vendor NPU multi_tensor_scale_tensor
```

**改后**：

```python
def multi_tensor_scale_tensor(
    self, chunk_size, noop_flag, tensor_lists, scale,
) -> None:
    """Multi-tensor scale with a tensor-valued scaling factor."""
    opt = _get_tenpu_optimizers()
    return opt.multi_tensor_scale_tensor(chunk_size, noop_flag, tensor_lists, scale)
```

### 为什么是「委托」而不是「自己分批」

**厂商包里本来就有这个算子**，签名和 `chunk_size` 的用法与 `multi_tensor_l2norm` 一致。
容器内 introspection（910C / CANN 9.0.1 / torch_npu 2.7.1.post8）：

```
transformer_engine_npu.pytorch.optimizers:
    multi_tensor_scale_tensor    ✅  ← 原生就有，收 chunk_size
    multi_tensor_scale           ✅
    multi_tensor_l2norm          ✅
    multi_tensor_adam            ❌
```

同一个文件里的三个实现，对 `chunk_size` 的处理不同，结果也不同：

| 实现 | 对 `chunk_size` 的处理 | 实测 |
|---|---|---|
| `multi_tensor_l2norm`（`:418`） | `opt.multi_tensor_l2norm(chunk_size, ...)` | ✅ 16/16 |
| `multi_tensor_scale`（`:363`） | `opt.multi_tensor_scale(chunk_size, ...)` | ✅ |
| `multi_tensor_scale_tensor`（`:369`） | 手工实现，从未使用 | 🔴 coreDim 越限 |

`multi_tensor_l2norm` 在**同一批 rank 上、失败前 6 秒**还是正常的
（某 rank：`15:03:30` l2norm 成功 → `15:03:36` scale_tensor 失败）。

### grid 的算法：`元素数 / 1024`，没有封顶

```
输出层梯度分片 = vocab 248320 × hidden 5120 / TP 4 = 317,849,600
310400 × 1024                                      = 317,849,600   ← 精确吻合
```

每个 program 负责 1024 个元素，换算成元素数的上限即 `65535 × 1024 = 67,107,840`，
而**输出层那一个梯度张量自己就有 3.18 亿元素，超限 4.7 倍**。

所以「在**张量之间**按元素数分批」是救不了的 —— 单个张量就超限，分批后它仍要
单独下发一次。要救只能在**张量内部**切块，那正是 `chunk_size` 的用途，
也正是厂商原生算子已经做好的事。我们试过张量间分批的补丁，coreDim 值一个没变。

### ⚠️ 普通梯度桶是擦着上限过去的

本配置 Megatron 侧 8 个梯度桶、`bucket_size=40000000`，最大的一个：

```
桶 1 = 66,846,720 元素 → 65,280 个 program
上限                   → 65,535 个
余量                   → 255 个（0.39%）
```

**不只是「输出层特殊」，整条路径都贴着天花板在跑。** `bucket_size` 稍微调大一点，
非输出层那条路径也会一起崩。建议：vendor 实现里凡是把整批张量交给底层 foreach
的地方都按 `chunk_size` 切，不要依赖上游恰好没超。

### 尚未解释清楚的一处

三个 coreDim 值里，`310400` 已与单个输出层张量精确对上；另外两个
（`141050` → 1.44 亿元素、`169351` → 1.73 亿元素）**超过任何单个梯度桶**（最大 6684 万），
按理应当来自多桶合并下发 —— 那样张量间分批本该对它们有效，但实测没变。
**此处矛盾我们没有解决**，列出来供参考，**不作为结论**。修法本身不依赖它。

---

## 实测验证

910C 单机 16 卡，Qwen3.5-27B 8 层 TP4 PP2 DP2 bf16，普通异步模式
（未加 `ASCEND_LAUNCH_BLOCKING`）。每一格都实跑过。

| TE-FL \ FlagGems | 工作分支（较早） | 工作分支（最新） | 上游 master + 屏蔽已知坏算子 |
|---|---|---|---|
| **原样 `test/all`** | 🔴 两个算子全挂 | 🔴 **逐项相同**，一个数字没变 | — |
| **本分支** | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 |

原样 `test/all` 的失败形态（两种 FlagGems 下完全一致）：

```
 4  All 1 implementation(s) failed for op='multi_tensor_adam'
12  All 1 implementation(s) failed for op='multi_tensor_scale_tensor'
12  value 141050 for parameter coreDim        ← rank 0–3
12  value 310400 for parameter coreDim        ← rank 4–7
12  value 169351 for parameter coreDim        ← rank 8–11
36  current working operator name is _amp_foreach_non_finite_check_and_unscale_kernel
12  state_steps[0] must be a one-element float32 tensor
```

**一步都没跑起来。** 三个 coreDim 值按 rank 排开是整齐的 **4/4/4，正好对上 TP4 分组**
—— 同一个 kernel、三个并行组规模不同，不是三个不同的 kernel（36 条报错里算子名
**全部相同、无一例外**）。

本分支跑完 8 步并正常收尾（`after training is done` / `saving checkpoint`），
全程 `skipped iterations: 0` / `nan iterations: 0`：

| iter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| lm loss | 13.751 | 13.766 | 10.837 | 10.252 | 10.189 | 10.078 | 10.074 | 10.191 |
| grad norm | 180.894 | 184.815 | 133.940 | 72.661 | 69.207 | 50.550 | 45.043 | 37.812 |

七个钉死的算子全部 **16/16 `using 'vendor.npu'`**，零失败、零 coreDim 报错。

> 本轮是 8 层调试规模，`elapsed time per iteration` 在 70–280 s 之间抖动，
> **不是性能数据**，不要引用其吞吐。

## 和 `codex/npu-vendor-compat` 的关系

`codex/npu-vendor-compat` 的 `be80f8e1`（10-14，未合进 `test/all`）里
`multi_tensor_scale_tensor` **已经是同样的委托写法**，并带了单测
`tests/plugin/backend/npu/test_multi_tensor_scale_tensor.py`，
还补了 `get_cudnn_version`（本次日志里该算子走的是 `default.flagos`）。

两边择一合入即可。本分支只动 `npu.py`，不含单测。

## 环境

```
Ascend 910C (Ascend910_9382) / 驱动 25.5.0 / 固件 7.8.0.5.216 / CANN 9.0.1
Python 3.12.13 / torch 2.7.1+cpu / torch_npu 2.7.1.post8
FlagTree 0.7.0+ascend (Triton 3.5.1)
FlagScale + Megatron-LM-FL
```
