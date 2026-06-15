# 应县木塔结构抗风抗震仿真与健康监测系统

## 项目概述

本系统为**应县木塔**古建筑保护提供完整的结构健康监测解决方案，集成传感器数据采集、有限元结构仿真、神经网络损伤识别、实时告警推送等功能，实现对木塔结构状态的全方位监测与评估。

### 核心功能

- 📊 **实时数据监测** - 5层40个传感器，每10分钟采集位移、加速度、温湿度、含水率
- 🏗️ **结构仿真分析** - 基于有限元法和木材各向异性本构模型，计算风/地震动力响应
- 🔍 **智能损伤识别** - 基于SSI/FDD模态分析和神经网络，精确定位结构损伤位置和程度
- ⚠️ **智能告警系统** - 层间位移角超限、固有频率异常等6种告警类型，WebSocket实时推送
- 🌐 **三维可视化** - Three.js高精度木塔模型，振动模态动画，损伤位置红色标记

---

## 技术架构

### 后端技术栈
| 技术 | 用途 |
|------|------|
| **FastAPI 0.104** | Python高性能Web框架，异步API |
| **PostgreSQL + TimescaleDB** | 关系数据 + 时序数据存储 |
| **SQLAlchemy 2.0** | ORM，支持异步会话 |
| **PyTorch 2.1** | 神经网络损伤识别模型 |
| **NumPy / SciPy** | 数值计算，有限元求解 |
| **WebSocket** | 实时数据和告警推送 |
| **JWT** | 认证授权 |

### 前端技术栈
| 技术 | 用途 |
|------|------|
| **React 18 + TypeScript** | 前端框架 |
| **Three.js** | 木塔三维模型渲染 |
| **Ant Design 5** | UI组件库 |
| **ECharts 5** | 数据可视化图表 |
| **Zustand** | 状态管理 |
| **Vite** | 构建工具 |

### 核心算法
1. **木材正交各向异性本构模型** - 9个弹性参数(E_L, E_R, E_T, G_LR, G_LT, G_RT, v_LR, v_LT, v_RT)
2. **铁木辛柯梁单元** - 考虑剪切变形的梁单元刚度矩阵
3. **Newmark-β法** - 动力时程分析求解器
4. **模态叠加法** - 基于模态坐标的高效动力求解
5. **Davenport风速谱** - 脉动风荷载模拟
6. **GB50011设计反应谱** - 地震荷载模拟
7. **SSI/FDD** - 随机子空间法/频域分解法 模态参数识别
8. **BP神经网络** - 损伤定位与程度评估

---

## 目录结构

```
AI_solo_coder_task_A_125/
├── database/                          # 数据库
│   └── 001_init_schema.sql           # TimescaleDB初始化脚本
├── backend/                           # Python后端
│   ├── main.py                       # FastAPI主应用入口
│   ├── config.py                     # 配置管理
│   ├── requirements.txt              # Python依赖
│   ├── .env                          # 环境变量
│   ├── core/                         # 核心模块
│   │   ├── database.py              # 数据库连接
│   │   ├── models.py                # SQLAlchemy ORM模型
│   │   └── schemas.py               # Pydantic数据模型
│   ├── api/                          # API路由
│   │   ├── auth_routes.py           # 认证授权API
│   │   ├── sensor_routes.py         # 传感器数据API
│   │   ├── simulation_routes.py     # 结构仿真API
│   │   ├── damage_routes.py         # 损伤识别API
│   │   └── alert_routes.py          # 告警系统API
│   ├── simulation/                   # 结构仿真模块
│   │   ├── timber_constitutive.py   # 木材本构模型
│   │   ├── load_generator.py        # 风/地震荷载生成
│   │   ├── finite_element_solver.py # 有限元求解器
│   │   └── simulation_service.py    # 仿真服务层
│   ├── damage_detection/             # 损伤识别模块
│   │   ├── modal_analysis.py        # SSI/FDD模态分析
│   │   ├── neural_network.py        # 神经网络模型
│   │   └── damage_service.py        # 损伤识别服务
│   ├── alerts/                       # 告警模块
│   │   ├── alert_engine.py          # 告警规则引擎
│   │   └── websocket_manager.py     # WebSocket连接管理
│   └── sensor_simulator.py          # 传感器模拟器脚本
├── frontend/                          # React前端
│   ├── package.json                  # 前端依赖
│   ├── vite.config.ts                # Vite配置
│   ├── tsconfig.json                 # TypeScript配置
│   └── src/
│       ├── main.tsx                  # 应用入口
│       ├── App.tsx                   # 主组件，路由配置
│       ├── types/index.ts            # TypeScript类型定义
│       ├── services/
│       │   ├── api.ts                # Axios API封装
│       │   └── websocket.ts          # WebSocket服务
│       ├── store/useStore.ts         # Zustand状态管理
│       ├── components/
│       │   └── PagodaModel/          # 木塔三维模型组件
│       │       ├── PagodaModel.tsx   # Three.js木塔模型
│       │       └── PagodaModel.scss  # 样式文件
│       ├── layouts/
│       │   └── MainLayout.tsx        # 主布局组件
│       ├── pages/                    # 页面组件
│       │   ├── Login.tsx             # 登录页
│       │   ├── Dashboard.tsx         # 总览仪表盘
│       │   ├── RealtimeMonitor.tsx   # 实时监测页
│       │   ├── Simulation.tsx        # 结构仿真页
│       │   ├── DamageDetection.tsx   # 损伤识别页
│       │   ├── AlertCenter.tsx       # 告警中心页
│       │   ├── DataAnalysis.tsx      # 数据分析页
│       │   └── SystemSettings.tsx    # 系统设置页
│       └── styles/
│           └── global.scss           # 全局样式
└── .trae/documents/                  # 项目文档
    ├── PRD_应县木塔健康监测系统.md   # 产品需求文档
    └── ARCH_技术架构文档.md          # 技术架构文档
```

---

## 快速开始

### 1. 环境要求

- **数据库**: PostgreSQL 14+ with TimescaleDB 2.10+
- **Python**: 3.10+
- **Node.js**: 18+
- **内存**: 推荐4GB以上
- **磁盘**: 推荐20GB以上

### 2. 数据库初始化

#### 安装TimescaleDB
```bash
# Ubuntu/Debian
sudo apt-get install timescaledb-2-postgresql-14

# Docker
docker run -d --name timescaledb -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  timescale/timescaledb:latest-pg14
```

#### 创建数据库并执行初始化脚本
```sql
-- 创建数据库
CREATE DATABASE pagoda_monitor;

-- 连接数据库
\c pagoda_monitor

-- 执行初始化脚本
\i database/001_init_schema.sql
```

**初始化内容**:
- 12个关系表（楼层、传感器、用户、告警、仿真、损伤识别等）
- 2个时序超表（sensor_data）
- 2个连续聚合视图（10分钟、1小时聚合）
- 40个传感器（5层 × 8个）
- 5个DTU设备
- 3个默认用户

### 3. 后端启动

```bash
# 进入后端目录
cd backend

# 安装依赖
pip install -r requirements.txt

# 配置环境变量（.env文件）
# DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/pagoda_monitor
# SECRET_KEY=your-secret-key-here

# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**后端服务地址**: http://localhost:8000
**API文档**: http://localhost:8000/docs (Swagger UI)

### 4. 前端启动

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

**前端地址**: http://localhost:5173

### 5. 启动传感器模拟器

```bash
cd backend

# 连续运行模式（每10分钟上报一次）
python sensor_simulator.py --mode continuous

# 回溯填充模式（填充过去7天的数据）
python sensor_simulator.py --mode backfill --days 7

# 单次模拟
python sensor_simulator.py --mode single
```

---

## 默认账号

| 用户名 | 密码 | 角色 | 权限 |
|--------|------|------|------|
| **admin** | admin123 | 系统管理员 | 全部权限 |
| **monitor** | monitor123 | 监测人员 | 数据查看、告警处理 |
| **researcher** | research123 | 研究人员 | 数据分析、仿真计算 |

---

## API接口概览

### 认证授权
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/login` | 登录获取令牌 |
| GET | `/api/auth/me` | 获取当前用户信息 |
| POST | `/api/auth/change-password` | 修改密码 |
| POST | `/api/auth/logout` | 退出登录 |

### 传感器数据
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/sensors/data` | 传感器数据上报（DTU） |
| GET | `/api/sensors/data` | 查询历史数据 |
| GET | `/api/sensors/realtime/{floor}` | 获取楼层实时数据 |
| GET | `/api/sensors` | 获取传感器列表 |
| GET | `/api/sensors/floors` | 获取楼层信息 |

### 结构仿真
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/simulation/run` | 运行结构仿真（风/地震） |
| POST | `/api/simulation/modal-analysis` | 运行模态分析 |
| GET | `/api/simulation/{id}` | 获取仿真信息 |
| GET | `/api/simulation/{id}/results` | 获取仿真结果 |

### 损伤识别
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/damage/analyze` | 启动损伤识别分析 |
| GET | `/api/damage/{id}/results` | 获取损伤识别结果 |
| GET | `/api/damage/modal-parameters` | 获取模态参数 |
| GET | `/api/damage/health/assessment` | 获取健康评估 |

### 告警系统
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/alerts` | 查询告警列表 |
| PUT | `/api/alerts/{id}/acknowledge` | 确认告警 |
| PUT | `/api/alerts/{id}/resolve` | 处理告警 |
| GET | `/api/alerts/thresholds` | 获取告警阈值 |
| POST | `/api/alerts/thresholds` | 设置告警阈值 |

### WebSocket实时推送
| 端点 | 房间 | 说明 |
|------|------|------|
| `/ws/monitoring` | monitoring | 实时监测数据推送 |
| `/ws/alerts` | alerts | 告警实时推送 |
| `/ws/simulation` | simulation | 仿真进度推送 |
| `/ws/damage` | damage | 损伤识别进度推送 |

---

## 木塔结构参数

应县木塔（佛宫寺释迦塔）建于辽清宁二年（公元1056年），是中国现存最高最古的木结构塔式建筑。

### 几何参数
| 楼层 | 高度(m) | 直径(m) | 立柱数 |
|------|---------|---------|--------|
| 1层 | 6.59 | 30.27 | 24 |
| 2层 | 5.49 | 22.65 | 24 |
| 3层 | 4.99 | 18.46 | 24 |
| 4层 | 4.59 | 15.28 | 24 |
| 5层 | 4.09 | 12.10 | 24 |

总高度: **67.31米** (含塔刹)

### 传感器配置
| 类型 | 数量/层 | 单位 | 说明 |
|------|---------|------|------|
| 位移X | 1 | mm | 水平X向位移 |
| 位移Y | 1 | mm | 水平Y向位移 |
| 加速度X | 1 | m/s² | 水平X向加速度 |
| 加速度Y | 1 | m/s² | 水平Y向加速度 |
| 温度 | 1 | °C | 环境温度 |
| 湿度 | 1 | % | 环境相对湿度 |
| 含水率 | 2 | % | 木材含水率 |

总计: **40个传感器** + **5个DTU设备**

---

## 告警规则

### 告警类型
| 类型 | 说明 | 预警阈值 | 严重阈值 |
|------|------|----------|----------|
| 层间位移角 | 相对位移/层高 | 0.25% | 0.5% |
| 固有频率下降 | 相对基准值下降 | 5% | 10% |
| X向位移 | 绝对位移 | 15mm | 30mm |
| Y向位移 | 绝对位移 | 15mm | 30mm |
| 加速度 | 振动加速度 | 0.25g | 0.50g |
| 温度 | 环境温度 | 45°C | 55°C |
| 木材含水率 | 木材含水率 | 25% | 35% |

### 告警级别
- **warning (警告)**: 超过预警阈值，需关注
- **critical (严重)**: 超过严重阈值，需立即处理

---

## 有限元模型说明

### 木材本构模型
采用**正交各向异性**本构模型，考虑三个正交方向的不同力学性能：
- **L方向** (顺纹): 沿木材纤维方向
- **R方向** (径向): 垂直于年轮方向
- **T方向** (弦向): 平行于年轮方向

默认木材参数（落叶松）:
```
E_L = 10000 MPa, E_R = 1200 MPa, E_T = 600 MPa
G_LR = 900 MPa, G_LT = 750 MPa, G_RT = 300 MPa
v_LR = 0.42, v_LT = 0.42, v_RT = 0.45
密度 ρ = 450 kg/m³
```

### 单元类型
- **铁木辛柯梁单元** (Timoshenko Beam): 考虑剪切变形和转动惯量
- 每层24根立柱，共120个梁单元
- 节点自由度: 6个 (ux, uy, uz, θx, θy, θz)

### 求解方法
1. **模态分析**: 特征值求解，提取前10阶模态
2. **动力时程分析**: Newmark-β法 (γ=0.5, β=0.25)
3. **阻尼模型**: 瑞利阻尼 [C] = α[M] + β[K]

---

## 神经网络损伤识别

### 模型架构
```
输入层 (50个特征)
   ↓
隐藏层1 (256, ReLU, Dropout=0.3)
   ↓
隐藏层2 (128, ReLU, Dropout=0.3)
   ↓
隐藏层3 (64, ReLU, Dropout=0.2)
   ↓
┌───────────────┬───────────────┐
│ 位置头        │ 程度头        │
│ (15, Softmax) │ (15, Sigmoid) │
└───────────────┴───────────────┘
   ↓               ↓
损伤位置概率    损伤程度指数
```

### 特征提取
从SSI/FDD提取的模态参数中提取50维特征向量：
- 前10阶固有频率
- 前10阶阻尼比
- 前10阶模态振型
- 频率变化率
- 模态置信准则(MAC)

### 损伤定位
输出15个单元的损伤概率（5层 × 3个代表性单元/层），使用Softmax归一化。

### 损伤程度
输出15个单元的损伤程度指数(0~1)，0表示无损伤，1表示完全破坏。

---

## 监测流程

```
传感器采集 → 4G DTU传输 → FastAPI接收 → TimescaleDB存储
    ↓
定时分析任务 ←───────────┘
    ↓
┌────────────────────────────────────┐
│ 模态参数识别(SSI/FDD)              │
│ 神经网络损伤识别                    │
│ 结构健康评估                        │
└────────────────────────────────────┘
    ↓
阈值检测 → 触发告警 → WebSocket推送 → 前端实时显示
    ↓
历史数据分析 ←──────────────────────┘
```

---

## 开发说明

### 运行测试
```bash
# 后端测试
cd backend
python -m pytest tests/

# 前端构建
cd frontend
npm run build
```

### 数据备份
```bash
# 备份数据库
pg_dump -U username pagoda_monitor > backup.sql

# 备份时序数据
pg_dump -U username -t sensor_data pagoda_monitor > sensor_data_backup.sql
```

### 常见问题

**Q: 数据库连接失败?**
A: 检查`.env`中的`DATABASE_URL`，确保PostgreSQL服务运行，TimescaleDB扩展已启用。

**Q: 前端无法连接后端?**
A: 检查`vite.config.ts`中的代理配置，确保后端服务在8000端口运行。

**Q: 传感器模拟器不产生数据?**
A: 检查数据库连接，确保sensor表已有初始化的40个传感器。

---

## 技术文档

详细文档请查看:
- [产品需求文档](.trae/documents/PRD_应县木塔健康监测系统.md)
- [技术架构文档](.trae/documents/ARCH_技术架构文档.md)
- [API在线文档](http://localhost:8000/docs)

---

## 版权声明

本系统用于应县木塔的结构健康监测研究与保护，所有监测数据仅供研究参考使用。

© 2024 古建筑保护研究中心

---

## 联系方式

- 技术支持: support@example.com
- 项目地址: https://github.com/example/pagoda-monitor
