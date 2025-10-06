#!/usr/bin/env python3
import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from real_time_comparator import main

if __name__ == "__main__":
    sys.exit(main())
