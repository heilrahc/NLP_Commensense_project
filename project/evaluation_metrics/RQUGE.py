from evaluate import load

class RQUGE:
    def __init__(self):
        # Load the RQUGE model
        self.model = load("alirezamsh/rquge")

    def get_score(self, generated_questions, contexts, answers):
        # Compute the RQUGE score
        results = self.model.compute(generated_questions=generated_questions, contexts=contexts, answers=answers)
        return results["mean_score"]