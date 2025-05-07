import json
from typing import List, Tuple, Dict
from collections import defaultdict
from LLMEvalTool.evaluator.exact_match import exact_match
from LLMEvalTool.utils import file_utils

def parse_cmb_jsonl(file_name: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    grouped_reference_codes = defaultdict(list)
    grouped_generated_codes = defaultdict(list)

    with open(file_name, 'r', encoding="utf-8") as file:
        # 逐行解析json数据
        for line in file:
            json_obj = json.loads(line)
            case_type = json_obj["case_type"]
            reference_code = json_obj["reference_code"]
            reference_code = reference_code.strip()
            generated_code = json_obj["generated_code"]
            generated_code = generated_code.strip()

            if any(c != ' ' for c in reference_code):
                if case_type == "block":
                    grouped_reference_codes[case_type].append(reference_code)
                    grouped_generated_codes[case_type].append(generated_code)
                    lines = len(reference_code.split("\n"))
                    if lines > 0:
                        case_type, ref_first_line, gen_first_line = gen_block_n_result(reference_code, generated_code,
                                                                                       1)
                        grouped_reference_codes[case_type].append(ref_first_line)
                        grouped_generated_codes[case_type].append(gen_first_line)
                    if lines > 1:
                        case_type, ref_first_line, gen_first_line = gen_block_n_result(reference_code, generated_code,
                                                                                       2)
                        grouped_reference_codes[case_type].append(ref_first_line)
                        grouped_generated_codes[case_type].append(gen_first_line)
                    if lines > 2:
                        case_type, ref_first_line, gen_first_line = gen_block_n_result(reference_code, generated_code,
                                                                                       3)
                        grouped_reference_codes[case_type].append(ref_first_line)
                        grouped_generated_codes[case_type].append(gen_first_line)
                    if lines > 3:
                        case_type, ref_first_line, gen_first_line = gen_block_n_result(reference_code, generated_code,
                                                                                       4)
                        grouped_reference_codes[case_type].append(ref_first_line)
                        grouped_generated_codes[case_type].append(gen_first_line)
                    if lines > 4:
                        case_type, ref_first_line, gen_first_line = gen_block_n_result(reference_code, generated_code,
                                                                                       5)
                        grouped_reference_codes[case_type].append(ref_first_line)
                        grouped_generated_codes[case_type].append(gen_first_line)
                    if lines > 5:
                        case_type, ref_first_line, gen_first_line = gen_block_n_result(reference_code, generated_code,
                                                                                       6)
                        grouped_reference_codes[case_type].append(ref_first_line)
                        grouped_generated_codes[case_type].append(gen_first_line)
                    if lines > 6:
                        case_type = "block_over6"
                        grouped_reference_codes[case_type].append(ref_first_line)
                        grouped_generated_codes[case_type].append(gen_first_line)
                else:
                    ref_first_line = reference_code
                    gen_first_line = generated_code
                    grouped_reference_codes[case_type].append(ref_first_line)
                    grouped_generated_codes[case_type].append(gen_first_line)
        return grouped_reference_codes, grouped_generated_codes

def gen_block_n_result(reference_code: str, generated_code: str, block_line):
    case_type = "block_" + str(block_line)
    ref_first_line = get_line(reference_code, block_line)
    gen_first_line = get_line(generated_code, block_line)
    return case_type, ref_first_line, gen_first_line

def parse_one_line_jsonl(file_name: str, block_line) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    grouped_reference_codes = defaultdict(list)
    grouped_generated_codes = defaultdict(list)

    with open(file_name, 'r', encoding="utf-8") as file:
        # 逐行解析json数据
        for line in file:
            json_obj = json.loads(line)
            case_type = json_obj["case_type"]
            reference_code = json_obj["reference_code"]
            generated_code = json_obj["generated_code"]

            if any(c != ' ' for c in reference_code):
                ref_first_line = get_line(reference_code, block_line)
                gen_first_line = get_line(generated_code, block_line)

                grouped_reference_codes[case_type].append(ref_first_line)
                grouped_generated_codes[case_type].append(gen_first_line)

    return grouped_reference_codes, grouped_generated_codes


def get_line(text: str, count):
    lines = text.split("\n")
    result = ""
    if len(lines) < count + 1:  # 如果只有一个换行符
        result = text
    else:
        result = "\n".join(lines[:count])  # 获取第二个换行符前的行
    return result

def parse_jsonl(file_name: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    grouped_reference_codes = defaultdict(list)
    grouped_generated_codes = defaultdict(list)

    with open(file_name, 'r', encoding="utf-8") as file:
        # 逐行解析json数据
        for line in file:
            json_obj = json.loads(line)
            case_type = json_obj["case_type"]
            reference_code = json_obj["reference_code"]
            generated_code = json_obj["generated_code"]

            if any(c != ' ' for c in reference_code):
                grouped_reference_codes[case_type].append(reference_code)
                grouped_generated_codes[case_type].append(generated_code)

    return grouped_reference_codes, grouped_generated_codes


def evaluation_em(generate_codes: List[str], reference_codes: List[str]) -> Dict[str, float]:
    metric = exact_match.ExactMatch()
    results_em = metric.compute(predictions=generate_codes, references=reference_codes)
    return results_em


def evaluation(file: str):
    grouped_reference_codes, grouped_generated_codes = parse_cmb_jsonl(file)
    bleu_scores = {}
    em_scores = {}

    for case_type in grouped_generated_codes:
        em_scores[case_type] = evaluation_em(grouped_generated_codes[case_type], grouped_reference_codes[case_type])
    for case_type in em_scores:
        print(f"{case_type}的EM为{em_scores[case_type]}")
    return em_scores

def evaluation_directory(directory: str):
    result_list = []
    for file in file_utils.list_files(directory):
        if file.endswith("generated.jsonl"):
            result = evaluation_and_output(file)
            result_list.append(result)
    return result_list

def evaluation_and_output(file: str):
    grouped_reference_codes, grouped_generated_codes = parse_cmb_jsonl(file)
    em_scores = {}

    for case_type in grouped_generated_codes:
        em_scores[case_type] = evaluation_em(grouped_generated_codes[case_type], grouped_reference_codes[case_type])
    em_result_list = []
    for case_type in em_scores:
        print(f"{case_type}的EM为{em_scores[case_type]}")
        em_result_list.append(em_scores[case_type]["exact_match"])
    result = {"em": em_result_list}
    return result

if __name__ == "__main__":
    # 需要更改为对应的文件路径
    evaluation_directories = "abc"
    for directory in evaluation_directories:
        result_list = evaluation_directory(directory)
