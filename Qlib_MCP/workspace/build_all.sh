#!/bin/bash
# 使用 Docker Compose 构建所有镜像

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 获取脚本所在目录（Qlib_MCP 根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}使用 Docker Compose 构建所有镜像${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查 docker-compose.yml 是否存在
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}错误: docker-compose.yml 不存在${NC}"
    exit 1
fi

# 检查 .env 文件，如果不存在则提示创建
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}警告: .env 文件不存在，将使用默认配置${NC}"
    echo -e "${YELLOW}建议复制 .env.example 为 .env 并根据实际情况修改${NC}"
    echo ""
fi

echo -e "${YELLOW}开始构建镜像...${NC}"

# 使用 docker compose 构建所有服务
docker compose build

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}所有 Docker 镜像构建成功！${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # 显示镜像信息
    echo -e "\n${YELLOW}已构建的镜像:${NC}"
    docker compose images
    
    echo -e "\n${GREEN}构建完成！现在可以通过以下方式使用:${NC}"
    echo -e "${YELLOW}1. 测试运行:${NC} ./run_training.sh --workspace alphasage --instruments csi300"
    echo -e "${YELLOW}2. MCP 调用:${NC} 通过 MCP 服务自动调用"
    echo -e "${YELLOW}3. 手动运行:${NC} docker compose run --rm alphasage python3 train_GP.py --instruments csi300"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}Docker 镜像构建失败！${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

