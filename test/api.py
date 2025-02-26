# import sys
# sys.path.append('..')
# from vlmeval.config import supported_VLM
# # print(supported_VLM)
# model = supported_VLM['llava_v1.5_7b']()
# # Forward Single Image
# ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
# print(ret)  # The image features a red apple with a leaf on it.
# # Forward Multiple Images
# ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
# print(ret)  # There are two apples in the provided images.


from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('liuhaotian/llava-v1.5-7b', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
# lmdeploy serve api_server --api-keys sk-ICSL123ICSL --server-port 23333 liuhaotian/llava-v1.5-7b