# URDF 导出的 USD 修复指南

## 问题

URDF export 生成的 USD 文件包含多层 robot 配置，直接作为 `RigidObject` 加载到 Isaac Lab 会报错：

```
Simulation view object is invalidated and cannot be used again to call updateArticulationsKinematic
Failed to get DOF velocities from backend
```

## 根因

URDF 导出文件包含 4 个配置层，其中 3 个与 RigidObject 冲突：

| 文件 (在 `urdf/<name>/configuration/` 下) | 内容 | 是否可用 |
|------|------|:---:|
| `<name>_base.usd` | mesh、visual、collider 结构 | ✅ 可用 |
| `<name>_robot.usd` | IsaacRobotAPI, robotLinks, robotJoints | ❌ 污染 |
| `<name>_physics.usd` | PhysicsFixedJoint, PhysicsCollisionGroup | ❌ 污染 |
| `<name>_sensor.usd` | sensor payload | ❌ 多余 |

**完全不加载 robot/physics/sensor 这三个配置层**。

## 修复步骤

### 1. 找到 `<name>_base.usd`

```
usd/<name>/urdf/<name>/configuration/<name>_base.usd
```

导出为 USDA 确认结构：

```bash
./isaaclab.sh -p scripts/tools/dump_usda.py --headless \
  usd/<name>/urdf/<name>/configuration/<name>_base.usd \
  /tmp/<name>_base.usda
```

关键信息：默认 prim 名称、base_link 路径、colliders/visuals/meshes 结构。

### 2. 创建最小化 USDA

模板如下（把 `<NAME>` 替换为实际名称，`<DEFAULT_PRIM>` 为 base.usd 的 defaultPrim）：

```usda
#usda 1.0
(
    defaultPrim = "<DEFAULT_PRIM>"
    metersPerUnit = 1
    subLayers = [
        @./urdf/<NAME>/configuration/<NAME>_base.usd@
    ]
    upAxis = "Z"
)

over "<DEFAULT_PRIM>"
{
    over "base_link" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        # 从原 <name>_physics.usd 的 USDA 导出中复制 mass/inertia 数据
        point3f physics:centerOfMass = (...)
        float3 physics:diagonalInertia = (...)
        float physics:mass = ...
        quatf physics:principalAxes = (...)

        def Xform "collisions" (
            prepend references = </colliders/base_link>
        )
        {
        }
    }
}

over "colliders"
{
    over "base_link"
    {
        over "base_link"
        {
            over "node_STL_BINARY_" (
                prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
            )
            {
                uniform token physics:approximation = "convexHull"
            }
        }
    }
}

def PhysicsScene "physicsScene" (
    prepend apiSchemas = ["PhysxSceneAPI"]
)
{
    vector3f physics:gravityDirection = (0, 0, -1)
    float physics:gravityMagnitude = 9.81
}
```

### 3. 关键数据来源

| 数据 | 来源 |
|------|------|
| `mass`, `centerOfMass`, `diagonalInertia`, `principalAxes` | 从原 `<name>_physics.usd` 导出 USDA 中复制 |
| `node_STL_BINARY_` 路径 | 检查 `<name>_base.usd` 中 meshes/colliders 的实际结构 |
| `colliders/base_link` 引用路径 | 从 `<name>_base.usd` 的 colliders Scope 中确认 |

### 4. 备份并替换

```bash
# 备份
cp usd/<name>/<name>.usd usd/<name>/<name>.usd.bak

# 转换 USDA -> USDC 并替换
./isaaclab.sh -p scripts/tools/convert_usda.py --headless \
  /tmp/<name>_minimal.usda \
  usd/<name>/<name>.usd
```

### 5. 验证

检查 prim 结构是否正确：

```bash
./isaaclab.sh -p scripts/tools/dump_usda.py --headless \
  usd/<name>/<name>.usd /tmp/verify.usda
```

确认：
- ✅ `base_link` 有 `PhysicsRigidBodyAPI`, `PhysicsMassAPI`
- ✅ `collisions/.../node_STL_BINARY_` 有 `PhysicsCollisionAPI`, `PhysicsMeshCollisionAPI`
- ✅ 有 `PhysicsScene`
- ❌ 无 `IsaacRobotAPI`, `IsaacLinkAPI`
- ❌ 无 `PhysicsFixedJoint`, `PhysicsCollisionGroup`
- ❌ 无 `robotLinks`, `robotJoints`

### 6. 辅助脚本

两个工具脚本位于 `scripts/tools/`：

```bash
# 导出 USD -> USDA（文本格式，可读）
./isaaclab.sh -p scripts/tools/dump_usda.py --headless <input.usd> <output.usda>

# 转换 USDA -> USDC（二进制）
./isaaclab.sh -p scripts/tools/convert_usda.py --headless <input.usda> <output.usdc>
```

## 实际案例：bottle.usd

- 备份：`usd/bottle/bottle.usd.bak`
- 模板：`/tmp/bottle_minimal.usda`
- base 文件：`usd/bottle/urdf/bottle/configuration/bottle_base.usd`
- 默认 prim：`bottle`
- 碰撞节点：`colliders/base_link/base_link/node_STL_BINARY_`
