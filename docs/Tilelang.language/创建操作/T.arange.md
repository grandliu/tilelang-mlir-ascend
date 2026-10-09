# Tilelang.language.arange

# 1. OP概述

简介：`tilelang.language.arange` 根据步长（strides）和偏移量（offset），向向量中填充从 0、1、2…… 开始的连续序列。

```markup
T.arange(dst, strides: Union[list, tuple], offset=0)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `dst` | `tensor` | 输出tensor |
| `strides`                         | `Union[list,tuple]` | 输入步长 |
| `offset`                          | `int` | 输入偏移量 |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | √ | × | √ | × | √ | √ | √ | × | × |

#### 2.2.2 Shape支持

结论：在shape方面，arange无特殊要求；

### 2.3 特殊限制说明

#### 2.3.1 问题说明

已知限制：当二维 stride-0 `T.arange` 的输出被紧邻的 `T.vbrc` 读取时，两者之间可能没有建立正确的同步依赖。结果是 `T.vbrc` 已开始读取并广播，而 `T.arange` 对行尾部分 lane 的写入尚未完全可见，从而在广播结果中留下 stale UB 数据。

该问题实测于 CANN 8.5.0 / Ascend910B2C；上述同步问题与工具链版本相关，升级后需重新验证。

#### 2.3.2 触发条件

当 `T.arange` 写入二维 `(1, N)` float32 缓冲，`strides` 含 0（如 `[0, 1]`），且输出紧接着作为 `T.vbrc` 的 src 时，同时满足以下条件可能触发竞态：

1. 缓冲数据类型为 float32（实测域，其他 dtype 未验证）；
2. `N mod 16 ∈ [9,12]`（行 stride 按 32B 补齐后 pad ≥ 4 个元素；对齐 N 或 pad ≤ 3 时实测正常）；
3. 广播目标行数 `bm ≥ 2`（`bm=1` 免疫；`bm=2` 低概率仍可触发）。

#### 2.3.3 竞态表现与验证注意事项

- 仅 dst 的**块首行**（`row % bm == 0`）尾部约 4 个有效 lane 保持 stale UB 残留，例如 N=300 时为列 296..299，N=268 时为列 264..267；
- `T.arange` 写入的 src 数据本身正确，其余行、其余列也正确；错误发生在二维 stride-0 arange 写入与后续向量广播读取之间的同步衔接；
- 该问题具有 Heisenbug 特征：在 `T.arange` 与 `T.vbrc` 之间插入读取依赖（例如用 `T.print` 打印 src 尾部）会引入额外依赖或等待，使症状完全消失。实测对照为无 print 10/10 失败、有 print 0/10 失败，因此排查和验证必须以无 print 版本为准。

#### 2.3.4 规避写法

使用一维 `T.arange` 生成连续序列，再通过 `T.reshape` 构造 `(1, N)` 视图。该写法语义等价，不经过 stride-0 二维写入路径，实测无竞态：

```python
idx_src = T.alloc_shared((N,), "float32")
idx_row = T.alloc_shared((1, N), "float32")
T.arange(idx_src, [1], 0)      # 一维连续填充
T.reshape(idx_src, idx_row)     # (N,) -> (1, N)，纯元数据零拷贝
T.vbrc(idx_row, idx_j)
```

当 `bm == 1` 时，直接执行 `T.reshape(idx_src, idx_j)`，不要调用同形状 `(1, N) -> (1, N)` 的 `T.vbrc`；后者会产生空的 `broadcast_dims`，触发 MLIR verify 失败。

### 2.4 使用方法

以下示例实现了一个形状为(M,N)的tensor的arange功能

```python
@tilelang.jit(target="npuir")
def vec_arange(M, N, block_M, block_N, src_dtype="float32", dst_dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dst_dtype),
        B: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), dst_dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dst_dtype)
            strides = [1, 2]
            T.arange(A_VEC, strides, offset=1)
            T.arange(B_VEC, strides)
            T.copy(A_VEC, A[bx, by])
            T.copy(B_VEC, B[bx, by])

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::arangeOp**将被转换为hivm::VArangeOp
