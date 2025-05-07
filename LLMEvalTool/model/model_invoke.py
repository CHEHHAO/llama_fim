import requests
import json

error_count = 0


def request_models(input, model_name):
    result = request_models_with_token(input, model_name)
    return result


def request_models_with_token(input, model_name):
    global error_count
    url = "http://127.0.0.1:8088/v1/completions"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "model": model_name,
        "prompt": input,
        "temperature": 0.1,
        "max_tokens": 256,
        "top_p": 1,
        "n": 1,
        "stream": False
    }
    # 增加重试
    error_count = 0
    while error_count < 5:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # 如果状态码不是200，会抛出HTTPError异常
            # 处理返回结果
            result = response.json()
            completion = result['choices'][0]['text']
            print(completion)
            return completion
        except requests.exceptions.RequestException as e:
            print("发生异常：", e)
            error_count = error_count + 1
    return ''

