#!/usr/bin/env python3
"""
重命名脚本：将文件夹中所有 lr(0.01)_rot1_* 格式的子文件夹
重命名为 lr(0.01)_struct(cka)_rot1_*
"""

import os
import sys
import re


def rename_folders(parent_dir):
    """
    重命名指定文件夹下的子文件夹

    Args:
        parent_dir: 父文件夹路径
    """
    if not os.path.exists(parent_dir):
        print(f"错误: 路径 '{parent_dir}' 不存在")
        return

    if not os.path.isdir(parent_dir):
        print(f"错误: '{parent_dir}' 不是一个文件夹")
        return

    # 获取所有子文件夹
    try:
        entries = os.listdir(parent_dir)
    except PermissionError:
        print(f"错误: 没有权限访问 '{parent_dir}'")
        return

    renamed_count = 0
    skipped_count = 0

    for entry in entries:
        full_path = os.path.join(parent_dir, entry)

        # 只处理文件夹
        if not os.path.isdir(full_path):
            continue

        # 检查是否匹配 lr(0.01)_rot1 模式（但不包含 struct(cka)）
        # 使用正则表达式确保精确匹配
        if 'lr(0.01)_rot1' in entry and 'struct(cka)' not in entry:
            # 生成新名称：将 lr(0.01)_rot1 替换为 lr(0.01)_struct(cka)_rot1
            new_name = entry.replace('lr(0.01)_rot1', 'lr(0.01)_struct(cka)_rot1')
            new_path = os.path.join(parent_dir, new_name)

            # 检查新名称是否已存在
            if os.path.exists(new_path):
                print(f"跳过: '{entry}' -> 目标路径已存在: '{new_name}'")
                skipped_count += 1
                continue

            try:
                os.rename(full_path, new_path)
                print(f"已重命名: '{entry}' -> '{new_name}'")
                renamed_count += 1
            except Exception as e:
                print(f"重命名失败: '{entry}' - 错误: {e}")
        elif 'lr(0.01)_struct(cka)_rot1' in entry:
            # 已经是正确格式，跳过
            pass

    print(f"\n总结:")
    print(f"  成功重命名: {renamed_count} 个文件夹")
    print(f"  跳过: {skipped_count} 个文件夹")


def main():
    if len(sys.argv) != 2:
        print("用法: python rename_folders.py <文件夹路径>")
        print("示例: python rename_folders.py /path/to/your/folder")
        sys.exit(1)

    folder_path = sys.argv[1]

    # 显示将要处理的路径
    print(f"正在处理文件夹: {folder_path}")
    print("-" * 60)

    rename_folders(folder_path)


if __name__ == "__main__":
    main()
