import argparse
import re
from typing import List, Dict, Any
import pandas as pd


def parse_value(s: str) -> Any:
    s = s.strip()
    # 尝试转成 int 或 float
    for fn in (int, float):
        try:
            return fn(s)
        except ValueError:
            continue
    return s


def parse_summary_file(path: str) -> List[Dict[str, Any]]:
    """
    解析类似如下格式的 summary.txt：

    Hyperparameters:
    lamda : 10.0
    gamma : 25.0
    ...
    Results:
    Best accuracy : 0.2545
    Best epoch : 75
    ==================================================
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 每个实验块用 ==== 分隔
    blocks = text.split("==================================================")
    experiments: List[Dict[str, Any]] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        exp: Dict[str, Any] = {}
        lines = block.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 跳过只有 "Hyperparameters:"、"Results:" 的行
            if re.match(r"^[A-Za-z ]+:$", line):
                continue

            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if val == "":
                    continue
                exp[key] = parse_value(val)

        if exp:
            experiments.append(exp)

    return experiments


def make_pivot(
    experiments: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    value_key: str
) -> pd.DataFrame:
    """
    构造二维表：行是 y_key，列是 x_key，值是 value_key。
    """
    if not experiments:
        raise ValueError("No experiments parsed from summary.txt")

    # 收集所有不同的 x 和 y 取值
    x_vals = sorted({exp[x_key] for exp in experiments if x_key in exp})
    y_vals = sorted({exp[y_key] for exp in experiments if y_key in exp})

    if not x_vals or not y_vals:
        raise ValueError(f"Cannot find x_key={x_key} or y_key={y_key} in experiments")

    # 初始化 DataFrame
    pivot_df = pd.DataFrame(index=y_vals, columns=x_vals, dtype=float)

    # 填值：若有重复 (同 x,y 多条)，后写的会覆盖前面的
    for exp in experiments:
        if x_key not in exp or y_key not in exp or value_key not in exp:
            continue
        x_val = exp[x_key]
        y_val = exp[y_key]
        val = exp[value_key]
        pivot_df.at[y_val, x_val] = val

    pivot_df.index.name = y_key
    pivot_df.columns.name = x_key
    return pivot_df


def collect_other_settings(
    experiments: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    value_key: str
) -> str:
    """
    把除了 x、y 和 value 以外的字段整理成一个字符串。
    返回格式化的设置信息字符串,用于在 Excel 中显示。
    """
    if not experiments:
        return ""

    all_keys = set().union(*[set(exp.keys()) for exp in experiments])
    other_keys = [k for k in all_keys if k not in {x_key, y_key, value_key}]

    lines = []
    for k in sorted(other_keys):
        values = {exp.get(k) for exp in experiments if k in exp}
        if len(values) == 1:
            # 如果该字段只有一个值,说明是固定设置
            lines.append(f"{k}: {list(values)[0]}")
        else:
            # 如果有多个值,列出所有值
            lines.append(f"{k}: {', '.join(str(v) for v in sorted(values))}")

    return "\n".join(lines)


def collect_other_settings_with_exclusion(
    experiments: List[Dict[str, Any]],
    excluded_keys: set
) -> str:
    """
    把除了指定排除键以外的字段整理成一个字符串。
    返回格式化的设置信息字符串,用于在 Excel 中显示。

    Args:
        experiments: 实验数据列表
        excluded_keys: 需要排除的键的集合
    """
    if not experiments:
        return ""

    all_keys = set().union(*[set(exp.keys()) for exp in experiments])
    other_keys = [k for k in all_keys if k not in excluded_keys]

    lines = []
    for k in sorted(other_keys):
        values = {exp.get(k) for exp in experiments if k in exp}
        if len(values) == 1:
            # 如果该字段只有一个值,说明是固定设置
            lines.append(f"{k}: {list(values)[0]}")
        else:
            # 如果有多个值,列出所有值
            lines.append(f"{k}: {', '.join(str(v) for v in sorted(values))}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse summary.txt to Excel pivot table.")
    parser.add_argument("--input", type=str, default="/mnt/cache/wuxinghao/FL/FedTSP/system/temp/summary.txt",
                        help="Path to summary.txt")
    parser.add_argument("--output", type=str, default="/mnt/cache/wuxinghao/FL/FedTSP/system/temp/summary.xlsx",
                        help="Output Excel file name")
    parser.add_argument("--x", type=str, default="lamda",
                        help="Field name for x-axis (columns), e.g., lamda")
    parser.add_argument("--y", type=str, default="gamma",
                        help="Field name for y-axis (rows), e.g., gamma")
    parser.add_argument("--value", type=str, default="Best accuracy",
                        help="Field to use as cell value, e.g., 'Best accuracy'")
    parser.add_argument("--group_by", type=str, default=None,
                        help="Field name(s) to group by (comma-separated). Each unique combination will create a separate table.")
    args = parser.parse_args()

    experiments = parse_summary_file(args.input)

    if not experiments:
        raise RuntimeError("No experiments found in input file.")

    # 如果指定了 group_by，按该字段(s)分组
    if args.group_by:
        # 解析 group_by 字段列表（支持逗号分隔的多个字段）
        group_by_fields = [f.strip() for f in args.group_by.split(',')]

        # 验证所有 group_by 字段都存在
        for field in group_by_fields:
            if not any(field in exp for exp in experiments):
                raise ValueError(f"Cannot find group_by field '{field}' in experiments")

        # 收集所有不同的 group_by 组合
        group_combinations = set()
        for exp in experiments:
            group_tuple = tuple(exp.get(field) for field in group_by_fields)
            group_combinations.add(group_tuple)

        group_combinations = sorted(group_combinations)

        # 按 group_by 组合分组实验
        grouped_experiments = {}
        for group_tuple in group_combinations:
            # 创建分组键：将字段名和值配对
            group_key = tuple(zip(group_by_fields, group_tuple))
            grouped_experiments[group_key] = [
                exp for exp in experiments
                if all(exp.get(field) == value for field, value in zip(group_by_fields, group_tuple))
            ]
    else:
        # 如果没有指定 group_by，所有实验作为一组
        grouped_experiments = {"all": experiments}
        group_by_fields = []

    raw_df = pd.DataFrame(experiments)

    # 写 Excel
    with pd.ExcelWriter(args.output, engine="xlsxwriter") as writer:
        workbook = writer.book

        # 创建格式
        settings_format = workbook.add_format({
            'text_wrap': True,
            'valign': 'top',
            'align': 'left',
            'border': 1
        })

        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D3D3D3',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })

        cell_format = workbook.add_format({
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })

        # 为每个分组创建一个sheet
        sheet_counter = 0
        for group_key, group_exps in grouped_experiments.items():
            # 确定sheet名称
            if args.group_by:
                # 生成sheet名称：使用序号，确保唯一且不超过31字符
                sheet_name = f"Group_{sheet_counter}"
                sheet_counter += 1
            else:
                sheet_name = "Results"

            # 创建透视表
            pivot_df = make_pivot(
                group_exps,
                x_key=args.x,
                y_key=args.y,
                value_key=args.value,
            )

            # 收集其他设置信息（排除 group_by 字段）
            excluded_keys = {args.x, args.y, args.value}
            if args.group_by:
                excluded_keys.update(group_by_fields)

            settings_text = collect_other_settings_with_exclusion(
                group_exps,
                excluded_keys
            )

            # 创建worksheet
            worksheet = workbook.add_worksheet(sheet_name)

            # 写入设置信息
            current_row = 0
            if args.group_by:
                # 如果是分组模式，显示当前组的值
                group_info_parts = [f"{field}={value}" for field, value in group_key]
                group_info = ", ".join(group_info_parts)
                worksheet.write(current_row, 0, f"Group: {group_info}", header_format)
                current_row += 1

            if settings_text:
                worksheet.write(current_row, 0, "Experiment Settings:", header_format)
                current_row += 1
                num_lines = settings_text.count('\n')
                if num_lines > 0:
                    worksheet.merge_range(current_row, 0, current_row + num_lines, 0, settings_text, settings_format)
                    current_row += num_lines + 1
                else:
                    worksheet.write(current_row, 0, settings_text, settings_format)
                    current_row += 1
                worksheet.set_column(0, 0, 30)  # 设置列宽

            # 透视表从设置信息下方开始
            start_row = current_row + 1

            # 写入透视表标题
            worksheet.write(start_row, 0, f"{args.y} \\ {args.x}", header_format)

            # 写入列标题 (x轴)
            for col_idx, x_val in enumerate(pivot_df.columns):
                worksheet.write(start_row, col_idx + 1, x_val, header_format)

            # 写入行标题 (y轴) 和数据
            for row_idx, y_val in enumerate(pivot_df.index):
                worksheet.write(start_row + row_idx + 1, 0, y_val, header_format)
                for col_idx, x_val in enumerate(pivot_df.columns):
                    value = pivot_df.at[y_val, x_val]
                    if pd.notna(value):
                        worksheet.write(start_row + row_idx + 1, col_idx + 1, value, cell_format)
                    else:
                        worksheet.write(start_row + row_idx + 1, col_idx + 1, "", cell_format)

            # 设置数据列宽
            for col_idx in range(len(pivot_df.columns)):
                worksheet.set_column(col_idx + 1, col_idx + 1, 12)

        # 原始数据sheet
        raw_df.to_excel(writer, sheet_name="raw_data", index=False)

    if args.group_by:
        print(f"Done. Wrote Excel to {args.output}")
        print(f"  - Created {len(grouped_experiments)} sheets grouped by '{args.group_by}'")
        print(f"  - raw_data sheet: all original data")
    else:
        print(f"Done. Wrote Excel to {args.output}")
        print(f"  - Results sheet: pivot table with settings")
        print(f"  - raw_data sheet: all original data")


if __name__ == "__main__":
    main()
