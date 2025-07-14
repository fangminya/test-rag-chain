# LangChain封装ZhipuAI LLM
# 导入必要的库
import os
import time
import uuid
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage
from pydantic import Field

# 定义一个ZhipuAI LLM类，继承自LLM类
class ZhipuaiLLM(LLM):
    model_name: str = "glm-4-plus"
    temperature: float = 0.1
    api_key: Optional[str] = None
    client: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.api_key is None:
            self.api_key = os.getenv("ZHIPUAI_API_KEY")
        
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=self.api_key)
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "zhipuai"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """调用智谱AI模型"""
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                **kwargs
            )
            # 检查响应结构
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                end_time = time.time()
                
                # 创建AIMessage格式的响应
                ai_message = AIMessage(
                    content=content if content else "无响应内容",
                    additional_kwargs={},
                    response_metadata={
                        'time_in_seconds': round(end_time - start_time, 2)
                    },
                    id=f'run-{uuid.uuid4()}-0',
                    usage_metadata={
                        'input_tokens': response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                        'output_tokens': response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                        'total_tokens': response.usage.total_tokens if hasattr(response, 'usage') else 0
                    }
                )
                return str(ai_message)
            else:
                return "响应格式错误"
        except Exception as e:
            print(f"调用智谱AI模型时出错: {e}")
            return f"错误: {str(e)}"
    
    @property
    def _identifying_params(self) -> dict[str, Any]:
        """返回模型参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
        } 