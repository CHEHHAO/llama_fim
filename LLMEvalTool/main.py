import sys
import os

sys.path.append(os.path.join(__file__, "../../../"))
from data_process import evaluation
import json
import csv
import time
from model import model_invoke

error_count = 0


def generate_one_completion_by_request(prompt, suffix, model_name):
    all_len = 4096
    prompt = prompt[-4096:]

    rest_len = all_len - len(prompt)
    suffix = suffix[0:rest_len]

    inputs_str = f"<｜fim▁begin｜>{prompt}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"
    #print("input:" + inputs_str)
    result = model_invoke.request_models(inputs_str, model_name)
    return result


def main(task, input, checkpoint):
    with open(input, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    output_lines = []
    completion_time = 0
    count = 0
    for line in lines:
        json_obj = json.loads(line)
        # 当前时间
        start_time = time.time()
        completion = generate_one_completion_by_request(prompt=json_obj["prefix"], suffix=json_obj["suffix"],
                                                        model_name=checkpoint)
        if completion == '':
            continue
        print(completion) 
        end_time = time.time()
        count = count + 1
        completion_time = completion_time + end_time - start_time
        print(completion_time)
        print(count)
        json_obj["generated_code"] = completion
        json_obj["model_name"] = checkpoint
        output_lines.append(json.dumps(json_obj, ensure_ascii=False))
        print(1)
    file_basename = os.path.join("./task", task_name)
    print(file_basename)
    output_file = file_basename + '/generated.jsonl'
    print("输出到："+output_file)
    # 生成一个新jsonl
    with open(output_file, 'w', encoding="utf-8") as file:
        file.write('\n'.join(output_lines))

    # 评估
    print(input)
    em_scores = evaluation.evaluation(output_file)
    with open(file_basename + '/output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(em_scores.keys())   # 写入表头（键）
        writer.writerow(em_scores.values()) # 写入数据（值）


# 扫描当前目录下文件，并把输出为一个列表
def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


import os
import shutil
import getopt

if __name__ == "__main__":
    inputfile = ''
    outputfile = ''
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hm:t:", ["model=", "task="])
    except getopt.GetoptError:
        print
        'test.py -m <model> -t <task>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print
            'test.py -m <model> -t <task>'
            sys.exit()
        elif opt in ("-m", "--model"):
            model_name = arg
        elif opt in ("-t", "--task"):
            task_name = arg
    print("模型名称为："+model_name)
    print("任务名称为："+task_name)

    base_case_path = "./case_info/java_4_blocks_60_lines_block_demo.jsonl"
    task_path = task_name
    folder_path = os.path.join("./task", task_path)
    checkpoint = model_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        shutil.copy(base_case_path, folder_path)
    file_array = list_files(folder_path)
    for file_path in file_array:
        try:
            main(task_name, file_path, checkpoint)
        except Exception as e:
            print(e)
