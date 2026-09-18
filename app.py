import os
import gradio as gr
import spaces
from smolagents import CodeAgent, InferenceClientModel
from tools import search_arxiv

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.environ.get("HF_TOKEN"))

agent = CodeAgent(
    tools=[search_arxiv],
    model=model,
    add_base_tools=False,
    max_steps=5)


@spaces.GPU
def agent_chat(user_prompt):
    return agent.run(user_prompt)


demo = gr.Interface(
    fn=agent_chat,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt..."),
    outputs=gr.Markdown(label="Agent Output"),
    title="Autonomous arXiv Research Agent",
    description="Powered by smolagents and Qwen 2.5 Coder")

demo.launch()