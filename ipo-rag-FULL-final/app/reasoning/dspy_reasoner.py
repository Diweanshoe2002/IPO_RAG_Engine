import dspy

class IPOAnalysis(dspy.Signature):
    context = dspy.InputField()
    question = dspy.InputField()
    reasoning = dspy.OutputField()
    answer = dspy.OutputField()

class IPOReasoner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(IPOAnalysis)

    def forward(self, context, question):
        return self.cot(context=context, question=question)

def run_reasoning(context_chunks, question):
    text = "\n\n".join(context_chunks)
    model = IPOReasoner()
    result = model(context=text, question=question)
    return result.answer
