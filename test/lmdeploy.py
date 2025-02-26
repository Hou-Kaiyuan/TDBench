from openai import OpenAI

# lmdeploy serve api_server --api-keys ICSL123ICSL --server-port 23333 internlm/internlm2-chat-1_8b
client = OpenAI(
    api_key='sk-ICSL123ICSL',
    base_url="http://l.icsl.cc:23333/v1"
)
model_name = client.models.list().data[0].id
print(model_name)