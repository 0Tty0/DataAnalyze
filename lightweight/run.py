import os

from dotenv import load_dotenv

from deepanalyze_langgraph import DeepAnalyzeLangGraph


def main():
    load_dotenv()

    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    api_base = os.getenv("API_BASE")
    api_key = os.getenv("API_KEY")
    workspace = os.getenv("WORKSPACE", "./workspace")
    max_rounds = int(os.getenv("MAX_ROUNDS", "20"))
    max_api_retries = int(os.getenv("MAX_API_RETRIES", "3"))
    max_exec_retries = int(os.getenv("MAX_EXEC_RETRIES", "2"))
    log_level = os.getenv("LOG_LEVEL", "INFO")

    prompt = """# Instruction
请读取并分析给定销售数据，生成一份简洁的中文数据科学报告。报告至少包含：
1. 总销售额、总销量、平均折扣；
2. 各区域销售额对比与Top区域；
3. 各产品销售额Top3；
4. 线上/线下渠道对比；
5. 关键发现与建议。
如需计算，请在 <Code>...</Code> 中输出可执行 Python 代码，最终用 <Answer>...</Answer> 给出结论。

# Data
File 1:
{"name": "sample_sales_200.csv", "size": "约20KB"}
"""

    agent = DeepAnalyzeLangGraph(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        max_rounds=max_rounds,
        max_api_retries=max_api_retries,
        max_exec_retries=max_exec_retries,
        log_level=log_level,
    )
    result = agent.generate(prompt=prompt, workspace=workspace)
    print(result["reasoning"])
    print(f"\n报告已保存到: {result['report_path']}")


if __name__ == "__main__":
    main()
