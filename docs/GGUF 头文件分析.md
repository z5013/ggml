# GGUF 头文件分析

## 1. 文件概述

`gguf.h` 是 ggml 库中的一个头文件，定义了 GGUF (GGML Universal Format) 文件格式的相关功能。GGUF 是一种二进制文件格式，用于存储和加载机器学习模型，特别是大型语言模型。

### 文件结构

GGUF 文件具有以下结构：
1. 文件魔数 "GGUF" (4 字节)
2. 文件版本 (uint32_t)
3. 文件中 ggml 张量的数量 (int64_t)
4. 文件中键值对的数量 (int64_t)
5. 每个键值对：
   - 键 (字符串)
   - 值类型 (gguf_type)
   - 如果值类型是 GGUF_TYPE_ARRAY：
     - 数组的类型 (gguf_type)
     - 数组中元素的数量 (uint64_t)
     - 数组中每个元素的二进制表示
   - 否则：
     - 值的二进制表示
6. 每个 ggml 张量：
   - 张量名称 (字符串)
   - 张量的维度数量 (uint32_t)
   - 每个维度：
     - 张量在该维度的大小 (int64_t)
   - 张量数据类型 (ggml_type)
   - 张量数据在张量数据二进制块中的偏移量 (uint64_t)
7. 张量数据二进制块 (可选，对齐)

### 字符串序列化

字符串被序列化为字符串长度 (uint64_t) 后跟不带空终止符的 C 字符串。

### 对齐方式

如果定义了特殊键 "general.alignment" (uint32_t)，则使用它进行对齐，否则使用 GGUF_DEFAULT_ALIGNMENT。

## 2. 核心数据结构

### 2.1 枚举类型

#### gguf_type

定义了可以存储为 GGUF KV 数据的类型：

| 枚举值 | 类型 | 描述 |
|--------|------|------|
| GGUF_TYPE_UINT8 | 0 | 无符号 8 位整数 |
| GGUF_TYPE_INT8 | 1 | 有符号 8 位整数 |
| GGUF_TYPE_UINT16 | 2 | 无符号 16 位整数 |
| GGUF_TYPE_INT16 | 3 | 有符号 16 位整数 |
| GGUF_TYPE_UINT32 | 4 | 无符号 32 位整数 |
| GGUF_TYPE_INT32 | 5 | 有符号 32 位整数 |
| GGUF_TYPE_FLOAT32 | 6 | 32 位浮点数 |
| GGUF_TYPE_BOOL | 7 | 布尔值 (存储为 int8_t) |
| GGUF_TYPE_STRING | 8 | 字符串 |
| GGUF_TYPE_ARRAY | 9 | 数组 |
| GGUF_TYPE_UINT64 | 10 | 无符号 64 位整数 |
| GGUF_TYPE_INT64 | 11 | 有符号 64 位整数 |
| GGUF_TYPE_FLOAT64 | 12 | 64 位浮点数 |
| GGUF_TYPE_COUNT | - | 标记枚举结束 |

### 2.2 结构体

#### gguf_context

GGUF 上下文结构体，用于操作 GGUF 文件。具体实现细节未在头文件中公开，仅声明为不完整类型。

#### gguf_init_params

```c
struct gguf_init_params {
    bool no_alloc;

    // if not NULL, create a ggml_context and allocate the tensor data in it
    struct ggml_context ** ctx;
};
```

- `no_alloc`：是否不分配内存
- `ctx`：如果不为 NULL，创建一个 ggml_context 并在其中分配张量数据

## 3. 函数接口

### 3.1 初始化和释放函数

| 函数名 | 描述 | 参数 | 返回值 |
|--------|------|------|--------|
| `gguf_init_empty` | 初始化空的 GGUF 上下文 | 无 | `struct gguf_context *` |
| `gguf_init_from_file` | 从文件初始化 GGUF 上下文 | `const char * fname`：文件名<br>`struct gguf_init_params params`：初始化参数 | `struct gguf_context *` |
| `gguf_free` | 释放 GGUF 上下文 | `struct gguf_context * ctx`：要释放的上下文 | 无 |

### 3.2 类型和版本相关函数

| 函数名 | 描述 | 参数 | 返回值 |
|--------|------|------|--------|
| `gguf_type_name` | 获取类型名称 | `enum gguf_type type`：类型 | `const char *`：类型名称 |
| `gguf_get_version` | 获取 GGUF 版本 | `const struct gguf_context * ctx`：上下文 | `uint32_t`：版本号 |
| `gguf_get_alignment` | 获取对齐方式 | `const struct gguf_context * ctx`：上下文 | `size_t`：对齐值 |
| `gguf_get_data_offset` | 获取数据偏移量 | `const struct gguf_context * ctx`：上下文 | `size_t`：偏移量 |

### 3.3 键值对操作函数

| 函数名 | 描述 | 参数 | 返回值 |
|--------|------|------|--------|
| `gguf_get_n_kv` | 获取键值对数量 | `const struct gguf_context * ctx`：上下文 | `int64_t`：键值对数量 |
| `gguf_find_key` | 查找键 | `const struct gguf_context * ctx`：上下文<br>`const char * key`：键名 | `int64_t`：键 ID，未找到返回 -1 |
| `gguf_get_key` | 获取键名 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `const char *`：键名 |
| `gguf_get_kv_type` | 获取键值对类型 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `enum gguf_type`：类型 |
| `gguf_get_arr_type` | 获取数组类型 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `enum gguf_type`：类型 |
| `gguf_remove_key` | 移除键 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名 | `int64_t`：键 ID，未找到返回 -1 |

### 3.4 值获取函数

| 函数名 | 描述 | 参数 | 返回值 |
|--------|------|------|--------|
| `gguf_get_val_u8` | 获取 uint8_t 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `uint8_t`：值 |
| `gguf_get_val_i8` | 获取 int8_t 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `int8_t`：值 |
| `gguf_get_val_u16` | 获取 uint16_t 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `uint16_t`：值 |
| `gguf_get_val_i16` | 获取 int16_t 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `int16_t`：值 |
| `gguf_get_val_u32` | 获取 uint32_t 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `uint32_t`：值 |
| `gguf_get_val_i32` | 获取 int32_t 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `int32_t`：值 |
| `gguf_get_val_f32` | 获取 float 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `float`：值 |
| `gguf_get_val_u64` | 获取 uint64_t 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `uint64_t`：值 |
| `gguf_get_val_i64` | 获取 int64_t 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `int64_t`：值 |
| `gguf_get_val_f64` | 获取 double 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `double`：值 |
| `gguf_get_val_bool` | 获取 bool 值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `bool`：值 |
| `gguf_get_val_str` | 获取字符串值 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `const char *`：值 |
| `gguf_get_val_data` | 获取原始数据 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `const void *`：数据指针 |

### 3.5 数组操作函数

| 函数名 | 描述 | 参数 | 返回值 |
|--------|------|------|--------|
| `gguf_get_arr_n` | 获取数组元素数量 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `size_t`：元素数量 |
| `gguf_get_arr_data` | 获取数组数据 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID | `const void *`：数据指针 |
| `gguf_get_arr_str` | 获取数组中的字符串 | `const struct gguf_context * ctx`：上下文<br>`int64_t key_id`：键 ID<br>`size_t i`：索引 | `const char *`：字符串 |

### 3.6 张量操作函数

| 函数名 | 描述 | 参数 | 返回值 |
|--------|------|------|--------|
| `gguf_get_n_tensors` | 获取张量数量 | `const struct gguf_context * ctx`：上下文 | `int64_t`：张量数量 |
| `gguf_find_tensor` | 查找张量 | `const struct gguf_context * ctx`：上下文<br>`const char * name`：张量名称 | `int64_t`：张量 ID，未找到返回 -1 |
| `gguf_get_tensor_offset` | 获取张量偏移量 | `const struct gguf_context * ctx`：上下文<br>`int64_t tensor_id`：张量 ID | `size_t`：偏移量 |
| `gguf_get_tensor_name` | 获取张量名称 | `const struct gguf_context * ctx`：上下文<br>`int64_t tensor_id`：张量 ID | `const char *`：张量名称 |
| `gguf_get_tensor_type` | 获取张量类型 | `const struct gguf_context * ctx`：上下文<br>`int64_t tensor_id`：张量 ID | `enum ggml_type`：类型 |
| `gguf_get_tensor_size` | 获取张量大小 | `const struct gguf_context * ctx`：上下文<br>`int64_t tensor_id`：张量 ID | `size_t`：大小 |
| `gguf_add_tensor` | 添加张量 | `struct gguf_context * ctx`：上下文<br>`const struct ggml_tensor * tensor`：张量 | 无 |
| `gguf_set_tensor_type` | 设置张量类型 | `struct gguf_context * ctx`：上下文<br>`const char * name`：张量名称<br>`enum ggml_type type`：类型 | 无 |
| `gguf_set_tensor_data` | 设置张量数据 | `struct gguf_context * ctx`：上下文<br>`const char * name`：张量名称<br>`const void * data`：数据 | 无 |

### 3.7 值设置函数

| 函数名 | 描述 | 参数 | 返回值 |
|--------|------|------|--------|
| `gguf_set_val_u8` | 设置 uint8_t 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`uint8_t val`：值 | 无 |
| `gguf_set_val_i8` | 设置 int8_t 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`int8_t val`：值 | 无 |
| `gguf_set_val_u16` | 设置 uint16_t 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`uint16_t val`：值 | 无 |
| `gguf_set_val_i16` | 设置 int16_t 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`int16_t val`：值 | 无 |
| `gguf_set_val_u32` | 设置 uint32_t 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`uint32_t val`：值 | 无 |
| `gguf_set_val_i32` | 设置 int32_t 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`int32_t val`：值 | 无 |
| `gguf_set_val_f32` | 设置 float 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`float val`：值 | 无 |
| `gguf_set_val_u64` | 设置 uint64_t 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`uint64_t val`：值 | 无 |
| `gguf_set_val_i64` | 设置 int64_t 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`int64_t val`：值 | 无 |
| `gguf_set_val_f64` | 设置 double 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`double val`：值 | 无 |
| `gguf_set_val_bool` | 设置 bool 值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`bool val`：值 | 无 |
| `gguf_set_val_str` | 设置字符串值 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`const char * val`：值 | 无 |
| `gguf_set_arr_data` | 设置数组数据 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`enum gguf_type type`：类型<br>`const void * data`：数据<br>`size_t n`：元素数量 | 无 |
| `gguf_set_arr_str` | 设置字符串数组 | `struct gguf_context * ctx`：上下文<br>`const char * key`：键名<br>`const char ** data`：字符串数组<br>`size_t n`：元素数量 | 无 |
| `gguf_set_kv` | 从另一个上下文设置或添加键值对 | `struct gguf_context * ctx`：目标上下文<br>`const struct gguf_context * src`：源上下文 | 无 |

### 3.8 文件写入函数

| 函数名 | 描述 | 参数 | 返回值 |
|--------|------|------|--------|
| `gguf_write_to_file` | 写入上下文到文件 | `const struct gguf_context * ctx`：上下文<br>`const char * fname`：文件名<br>`bool only_meta`：是否只写入元数据 | `bool`：是否成功 |
| `gguf_get_meta_size` | 获取元数据大小 | `const struct gguf_context * ctx`：上下文 | `size_t`：元数据大小 |
| `gguf_get_meta_data` | 获取元数据 | `const struct gguf_context * ctx`：上下文<br>`void * data`：数据指针 | 无 |

## 4. 使用注意事项

1. **类型安全**：使用值获取函数时，必须确保使用正确的类型函数，否则会导致程序中止。例如，对于存储为 GGUF_TYPE_INT32 的值，必须使用 `gguf_get_val_i32` 函数获取。

2. **内存管理**：使用 `gguf_init_from_file` 函数时，需要注意内存分配。如果设置了 `ctx` 参数，会创建一个 ggml_context 并在其中分配张量数据。

3. **文件格式**：GGUF 文件格式有特定的结构，必须按照规定的格式读写，否则会导致文件损坏或无法读取。

4. **对齐要求**：张量数据需要对齐，默认对齐值为 32 字节，可以通过 "general.alignment" 键自定义。

5. **字符串处理**：字符串在 GGUF 文件中存储为长度前缀的形式，读取时需要注意处理。

6. **数组处理**：对于数组类型，需要先获取数组类型和元素数量，然后再获取数组数据。

7. **张量操作**：添加张量时，张量名称必须唯一。更改张量类型后，所有具有更高索引的张量的偏移量会立即重新计算。

`gguf.h` 依赖于 `ggml.h` 头文件，以及标准库中的 `stdbool.h` 和 `stdint.h`。

## 6. 总结

`gguf.h` 是 ggml 库中用于处理 GGUF 文件格式的核心头文件，提供了完整的 API 用于读写 GGUF 文件。它定义了清晰的数据结构和函数接口，支持各种数据类型和数组操作，为机器学习模型的存储和加载提供了可靠的基础。

通过合理使用这些 API，可以有效地处理大型机器学习模型，实现模型的序列化和反序列化，为模型的部署和使用提供便利。

## 7. 代码示例

### 7.1 从文件加载 GGUF 上下文

```c
#include "gguf.h"

int main() {
    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = NULL,
    };
    
    struct gguf_context * ctx = gguf_init_from_file("model.gguf", params);
    if (!ctx) {
        fprintf(stderr, "Failed to load GGUF file\n");
        return 1;
    }
    
    // 使用上下文...
    
    gguf_free(ctx);
    return 0;
}
```

### 7.2 读取键值对

```c
// 查找键
int64_t key_id = gguf_find_key(ctx, "general.architecture");
if (key_id >= 0) {
    // 获取值类型
    enum gguf_type type = gguf_get_kv_type(ctx, key_id);
    if (type == GGUF_TYPE_STRING) {
        const char * value = gguf_get_val_str(ctx, key_id);
        printf("Architecture: %s\n", value);
    }
}
```

### 7.3 写入 GGUF 文件

```c
// 创建空上下文
struct gguf_context * ctx = gguf_init_empty();

// 添加键值对
GGUF_set_val_str(ctx, "general.architecture", "gpt2");
GGUF_set_val_i32(ctx, "gpt2.n_embd", 768);
GGUF_set_val_i32(ctx, "gpt2.n_layer", 12);

// 添加张量
// ...

// 写入文件
bool success = gguf_write_to_file(ctx, "model.gguf", false);
if (!success) {
    fprintf(stderr, "Failed to write GGUF file\n");
}

gguf_free(ctx);
```

