#!/bin/bash
# Docker Compose 环境快速设置脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Qlib MCP Docker Compose 环境设置${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1. 检查必要的工具
echo -e "\n${BLUE}[1/5] 检查必要工具...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装${NC}"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker 已安装: $(docker --version)${NC}"

if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}错误: Docker Compose 未安装${NC}"
    echo "请先安装 Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose 已安装: $(docker compose --version)${NC}"

# 检查 nvidia-docker（可选）
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA 驱动已安装${NC}"
    if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA Docker 支持已启用${NC}"
    else
        echo -e "${YELLOW}⚠ NVIDIA Docker 支持未启用，将无法使用 GPU${NC}"
        echo -e "${YELLOW}  安装方法: sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker${NC}"
    fi
else
    echo -e "${YELLOW}⚠ NVIDIA 驱动未安装，将在 CPU 模式下运行${NC}"
fi

# 2. 创建 .env 文件
echo -e "\n${BLUE}[2/5] 配置环境变量...${NC}"

if [ -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env 文件已存在，是否覆盖? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm .env
    else
        echo -e "${YELLOW}跳过 .env 文件创建${NC}"
    fi
fi

if [ ! -f ".env" ]; then
    # 检测 qlib 数据路径
    QLIB_DATA_PATH=""
    if [ -d "$HOME/.qlib/qlib_data" ]; then
        QLIB_DATA_PATH="$HOME/.qlib/qlib_data"
    elif [ -d "/root/.qlib/qlib_data" ]; then
        QLIB_DATA_PATH="/root/.qlib/qlib_data"
    fi
    
    if [ -z "$QLIB_DATA_PATH" ]; then
        echo -e "${YELLOW}⚠ 未检测到 Qlib 数据目录${NC}"
        echo -e "${YELLOW}  请输入 Qlib 数据路径（默认: $HOME/.qlib/qlib_data）:${NC}"
        read -r user_input
        QLIB_DATA_PATH=${user_input:-"$HOME/.qlib/qlib_data"}
    fi
    
    # 检测 CUDA 设备
    CUDA_DEVICE="0"
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        echo -e "${GREEN}检测到 ${GPU_COUNT} 个 GPU${NC}"
        echo -e "${YELLOW}请输入默认使用的 CUDA 设备编号（0-$((GPU_COUNT-1))，默认: 0）:${NC}"
        read -r user_cuda
        CUDA_DEVICE=${user_cuda:-"0"}
    fi
    
    # 创建 .env 文件
    cat > .env << EOF
# Qlib MCP Docker Compose 环境变量
# 自动生成于 $(date)

# Qlib 数据路径（宿主机）
QLIB_DATA_PATH=${QLIB_DATA_PATH}

# 默认使用的 CUDA 设备
CUDA_DEVICE=${CUDA_DEVICE}

# qlib-benchmark 专用 CUDA 设备（可选，默认使用 CUDA_DEVICE）
CUDA_DEVICE_BENCHMARK=${CUDA_DEVICE}

# Docker 镜像版本标签
IMAGE_TAG=latest
EOF
    
    echo -e "${GREEN}✓ .env 文件已创建${NC}"
    cat .env
else
    echo -e "${GREEN}✓ .env 文件已存在${NC}"
fi

# 3. 检查必要的目录和文件
echo -e "\n${BLUE}[3/5] 检查项目文件...${NC}"

if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}错误: docker-compose.yml 不存在${NC}"
    exit 1
fi
echo -e "${GREEN}✓ docker-compose.yml 存在${NC}"

if [ ! -d "workspace/AlphaSAGE" ]; then
    echo -e "${RED}错误: workspace/AlphaSAGE 目录不存在${NC}"
    exit 1
fi
echo -e "${GREEN}✓ workspace/AlphaSAGE 存在${NC}"

if [ ! -f "workspace/AlphaSAGE/Dockerfile" ]; then
    echo -e "${RED}错误: workspace/AlphaSAGE/Dockerfile 不存在${NC}"
    exit 1
fi
echo -e "${GREEN}✓ AlphaSAGE Dockerfile 存在${NC}"

# 添加 qlib_benchmark 检查
if [ ! -d "workspace/qlib_benchmark" ]; then
    echo -e "${RED}错误: workspace/qlib_benchmark 目录不存在${NC}"
    exit 1
fi
echo -e "${GREEN}✓ workspace/qlib_benchmark 存在${NC}"

if [ ! -f "workspace/qlib_benchmark/Dockerfile" ]; then
    echo -e "${RED}错误: workspace/qlib_benchmark/Dockerfile 不存在${NC}"
    exit 1
fi
echo -e "${GREEN}✓ qlib_benchmark Dockerfile 存在${NC}"

# 4. 构建 Docker 镜像
echo -e "\n${BLUE}[4/5] 构建 Docker 镜像...${NC}"
echo -e "${YELLOW}这可能需要 5-10 分钟，请耐心等待...${NC}"

if ./build_all.sh; then
    echo -e "${GREEN}✓ Docker 镜像构建成功${NC}"
else
    echo -e "${RED}✗ Docker 镜像构建失败${NC}"
    exit 1
fi

# 5. 测试运行
echo -e "\n${BLUE}[5/5] 测试 Docker Compose 环境...${NC}"

echo -e "${YELLOW}测试 AlphaSAGE 容器...${NC}"
if docker compose run --rm alphasage python3 --version; then
    echo -e "${GREEN}✓ AlphaSAGE 容器测试成功${NC}"
else
    echo -e "${RED}✗ AlphaSAGE 容器测试失败${NC}"
    exit 1
fi

echo -e "${YELLOW}测试 qlib-benchmark 容器...${NC}"
if docker compose run --rm qlib-benchmark python3 --version; then
    echo -e "${GREEN}✓ qlib-benchmark 容器测试成功${NC}"
else
    echo -e "${RED}✗ qlib-benchmark 容器测试失败${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker Compose 环境测试成功${NC}"

# 完成
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Docker Compose 环境设置完成！${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}下一步:${NC}"
echo -e "1. 运行 AlphaSAGE 训练:"
echo -e "   ${BLUE}docker compose run --rm alphasage python3 train_GP.py --instruments csi300${NC}"
echo -e ""
echo -e "2. 运行 qlib-benchmark 基准测试:"
echo -e "   ${BLUE}docker compose run --rm qlib-benchmark python3 train_with_custom_factors.py${NC}"
echo -e ""
echo -e "3. 查看完整文档:"
echo -e "   ${BLUE}cat README_DOCKER_COMPOSE.md${NC}"
echo -e ""
echo -e "4. 通过 MCP 调用（自动使用 Docker Compose）:"
echo -e "   ${BLUE}python mcp_server_inline.py${NC}"
echo -e ""
echo -e "${GREEN}Happy Training! 🚀${NC}"

