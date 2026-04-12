# main.py
from agent import HelloAgent
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    console.print(Panel(
        "[bold]🤖 Hello Agent 已启动！[/bold]\n"
        "我可以帮你：计算数学、查询时间、搜索维基百科、记录笔记\n"
        "输入 'quit' 退出，输入 'reset' 重置对话",
        title="Agent 启动",
        border_style="green"
    ))
    
    agent = HelloAgent()
    
    while True:
        user_input = input("\n你：").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            console.print("[bold]再见！[/bold]")
            break
        
        if user_input.lower() == "reset":
            agent.reset()
            continue
        
        agent.chat(user_input)

if __name__ == "__main__":
    main()
