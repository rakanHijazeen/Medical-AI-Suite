from app.rag.engine import MedicalRAGEngine


def fake_stream(**kwargs):
    class Delta:
        def __init__(self, content):
            self.content = content

    class Choice:
        def __init__(self, content):
            self.delta = Delta(content)
            self.message = None

    class Chunk:
        def __init__(self, content):
            self.choices = [Choice(content)]

    yield Chunk('hello ')
    yield Chunk('world')


class FakeEngine(MedicalRAGEngine):
    def __init__(self):
        self.api_key = 'fake'
        self.client = None
        self.model = 'llama-3.1-8b-instant'


engine = FakeEngine()
collected = []


def cb(chunk):
    collected.append(chunk)

result = engine._stream_and_collect(lambda stream=False: fake_stream(), stream_callback=cb)
print('result:', result)
print('collected:', collected)
