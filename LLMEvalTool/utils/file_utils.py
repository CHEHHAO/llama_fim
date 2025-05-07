import os
import json

def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def json_list_to_jsonl(json_list, output_file):
    """
    将JSON列表转换为JSONL文件

    :param json_list: 包含JSON对象的列表
    :param output_file: 输出的JSONL文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in json_list:
                # 将每个JSON对象写入文件，并在末尾添加换行符
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"成功将JSON列表转换为JSONL文件，输出文件: {output_file}")
    except Exception as e:
        print(f"发生错误: {e}")