"""LLM 回答模組互動式測試腳本

用法：
    python LLM_prompt_test.py

功能：驗證 system prompt + session 記憶的多輪對話行為。

CLI 指令：
    /reset    清空對話歷史（system prompt 保留）
    /history  顯示目前記憶
    /system   顯示目前的 system prompt
    /setsys   進入多行輸入模式更換 system prompt（單獨一行輸入 END 結束）
    /q        離開
"""

##test 記憶邏輯
#三個輪胎的車通常叫甚麼名字?
#如果這樣的車拆了一個輪子，還剩幾個輪子? 
#如果我再拆了一個輪子，還剩幾個輪子?

import os

from dotenv import load_dotenv
from ollama import Client

from llm_chat import ChatBot


def init_client():
    load_dotenv()
    ollama_host = os.getenv("OLLAMA_HOST")
    model_name = os.getenv("LLM_MODEL_NAME")
    if not ollama_host or not model_name:
        raise RuntimeError("請確認 .env 中 OLLAMA_HOST 與 LLM_MODEL_NAME 已設定")
    print("連線 Ollama:", ollama_host, "/", model_name)
    return Client(host=ollama_host), model_name


def read_multiline(prompt: str) -> str:
    print(prompt)
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    ollama_client, model_name = init_client()
    bot = ChatBot(ollama_client, model_name)

    print("\n" + "=" * 60)
    print("就緒。輸入問題開始對話。指令：/reset /history /system /setsys /q")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n對話結束")
            break

        if not user_input:
            continue
        if user_input == "/q":
            print("對話結束")
            break
        if user_input == "/reset":
            bot.reset()
            print("已清空對話歷史")
            continue
        if user_input == "/history":
            hist = bot.get_history()
            if not hist:
                print("（目前無歷史）")
            for i, m in enumerate(hist):
                preview = m["content"].replace("\n", " ")[:80]
                print(f"  [{i}] {m['role']}: {preview}...")
            continue
        if user_input == "/system":
            print("--- system prompt ---")
            print(bot.system_prompt)
            print("---------------------")
            continue
        if user_input == "/setsys":
            new_prompt = read_multiline("輸入新的 system prompt，單獨一行輸入 END 結束：")
            if new_prompt.strip():
                bot.set_system_prompt(new_prompt)
                print("已更新 system prompt 並清空歷史")
            else:
                print("輸入為空，未更新")
            continue

        print("生成中...")
        answer = bot.chat(user_input)
        print(f"\n助理：{answer}")


if __name__ == "__main__":
    main()
